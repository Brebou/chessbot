"""Per-ply soft-label fine-tuning (every position of every game is a training signal).

  python -m plan.train --data data/r1/shards --init base_model --out out/r1 --epochs 3 --lr 1e-4 --device cuda
Several --data dirs allowed (e.g. SF shards + beam winmoves shards). Holds out the LAST --val-frac of the
games of the first dir for validation. Saves out/ep{N}/ and out/last/ (EMA weights, same keys as base).
"""
import argparse, copy, glob, math, os, random, time
import numpy as np, torch
import torch.nn.functional as F
from plan.common import load_model, logits_of, save_checkpoint


def load_shards(d):
    g = []
    for f in sorted(glob.glob(f"{d}/shard_*.pt")):
        g += torch.load(f, weights_only=False)
    return g


def collate(games, bos, K, V, dev):
    B, L = len(games), max(len(g["ids"]) for g in games)
    X = np.zeros((B, L), np.int64); I = np.zeros((B, L, K), np.int64)
    P = np.zeros((B, L, K), np.float32); W = np.zeros((B, L), np.float32)
    for b, g in enumerate(games):
        k = len(g["ids"]); X[b, 0] = bos; X[b, 1:k] = g["ids"][:k - 1]
        idx = g["idx"].astype(np.int64); pp = g["p"].astype(np.float32).copy(); pp[idx < 0] = 0
        I[b, :k] = np.maximum(idx, 0); P[b, :k] = pp; W[b, :k] = g["w"]
    LM = np.zeros((B, L), np.float32); mask = None
    if "lg_flat" in games[0]:
        mask = np.zeros((B, L, V), bool)
        for b, g in enumerate(games):
            cnt = np.diff(g["lg_off"]); k = len(cnt)
            mask[b, np.repeat(np.arange(k), cnt), g["lg_flat"].astype(np.int64)] = True; LM[b, :k] = cnt > 0
    out = [torch.from_numpy(x).to(dev) for x in (X, I, P, W, LM)]
    return out + [None if mask is None else torch.from_numpy(mask).to(dev)]


def batches(games, bs, bos, K, V, dev, shuffle=True):
    order = list(range(len(games)))
    if shuffle:
        random.shuffle(order)
    chunks = [order[i:i + bs * 50] for i in range(0, len(order), bs * 50)]
    bl = []
    for c in chunks:
        c.sort(key=lambda i: len(games[i]["ids"]))
        bl += [c[i:i + bs] for i in range(0, len(c), bs)]
    if shuffle:
        random.shuffle(bl)
    for b in bl:
        yield collate([games[i] for i in b], bos, K, V, dev)


def loss_fn(logits, I, P, W, LM, mask, legal_w=0.0, ref_logits=None, kl=0.0):
    lf = logits.float(); logp = lf.log_softmax(-1)
    ce = -(P * logp.gather(-1, I)).sum(-1)
    wsum = W.sum().clamp(min=1e-6)
    loss = (ce * W).sum() / wsum
    if kl > 0 and ref_logits is not None:
        rp = ref_logits.float().softmax(-1)
        kld = (rp * (rp.clamp_min(1e-9).log() - logp)).sum(-1)
        loss = loss + kl * (kld * (W > 0)).sum() / (W > 0).sum().clamp(min=1)
    illegal = 0.0
    if mask is not None:
        ll = -(lf.masked_fill(~mask, float("-inf")).logsumexp(-1) - lf.logsumexp(-1))
        ll = torch.where(LM > 0, ll, torch.zeros_like(ll))
        if legal_w > 0:
            loss = loss + legal_w * ll.sum() / LM.sum().clamp(min=1)
        ok = mask.gather(-1, lf.topk(3, -1).indices).any(-1)
        illegal = ((~ok) & (LM > 0)).sum().item()
    top1 = (logits.argmax(-1) == I[..., 0]) & (W > 0)
    return loss, top1.sum().item(), (W > 0).sum().item(), wsum.item(), illegal, LM.sum().item()


@torch.no_grad()
def validate(model, val, bos, K, V, dev):
    model.eval(); tot = acc = n = ws = ill = lm = 0.0
    for X, I, P, W, LM, mask in batches(val, 64, bos, K, V, dev, shuffle=False):
        l, c, m, w, i, q = loss_fn(logits_of(model, X), I, P, W, LM, mask)
        tot += l.item() * w; acc += c; n += m; ws += w; ill += i; lm += q
    return tot / max(ws, 1e-6), acc / max(n, 1), ill / max(lm, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True); ap.add_argument("--init", default="base_model")
    ap.add_argument("--base", default="base_model", help="dir whose config/tokenizer/key names are reused; also KL ref")
    ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--ema", type=float, default=0.999); ap.add_argument("--kl", type=float, default=0.0)
    ap.add_argument("--val-frac", type=float, default=0.02); ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-steps", type=int)
    ap.add_argument("--legal-weight", type=float, default=1.0, help="weight of -log(prob mass on legal moves)")
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    dev = a.device
    model, vocab = load_model(a.init, dev); model.train()
    ref = load_model(a.base, dev)[0] if a.kl > 0 else None
    sets = [load_shards(d) for d in a.data]
    K = max(s[0]["idx"].shape[1] for s in sets if s)
    nval = int(len(sets[0]) * a.val_frac)
    val, train = sets[0][len(sets[0]) - nval:], sets[0][:len(sets[0]) - nval]
    for s in sets[1:]:
        train += s
    V = model.config.vocab_size
    if any("lg_flat" not in g for g in train + val):
        print("WARNING: shards lack legal labels (run plan.addlegal) -> legality loss disabled")
        for g in train + val:
            g.pop("lg_flat", None); g.pop("lg_off", None)
    print(f"train games {len(train)}, val games {len(val)}, K={K}, V={V}")
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95))
    spe = math.ceil(len(train) / a.bs); total = a.max_steps or spe * a.epochs
    lr_at = lambda s: a.lr * (s + 1) / a.warmup if s < a.warmup else \
        a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, total - a.warmup))))
    if val:
        print("val before: ce %.4f top1 %.4f illegal_top3 %.4f" % validate(ema, val, vocab.bos, K, V, dev))
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        model.train(); run = []
        for X, I, P, W, LM, mask in batches(train, a.bs, vocab.bos, K, V, dev):
            for gp in opt.param_groups:
                gp["lr"] = lr_at(step)
            lg = logits_of(model, X)
            rl = None
            if ref is not None:
                with torch.no_grad():
                    rl = logits_of(ref, X)
            loss, *_ = loss_fn(lg, I, P, W, LM, mask, a.legal_weight, rl, a.kl)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            d = min(a.ema, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            run.append(loss.item()); step += 1
            if step % 100 == 0:
                print(f"ep {ep} step {step}/{total} loss {np.mean(run[-100:]):.4f} lr {lr_at(step):.2e} "
                      f"{time.time()-t0:.0f}s", flush=True)
            if step >= total:
                break
        if val:
            print("ep %d val (EMA): ce %.4f top1 %.4f illegal_top3 %.4f" % ((ep,) + validate(ema, val, vocab.bos, K, V, dev)))
        save_checkpoint(ema, a.base, f"{a.out}/ep{ep+1}"); save_checkpoint(ema, a.base, f"{a.out}/last")
        if step >= total:
            break


if __name__ == "__main__":
    main()
