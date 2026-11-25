"""Shared helpers for the plan/ pipeline.

ADAPT HERE if your repo differs:
  * load_model / load_vocab : how to load base_model (HF GPT2-style assumed)
  * engine_hook.py          : plug in the OFFICIAL level-1/2/3 engine (strongly recommended)
Run `python -m plan.selfcheck` first; it tells you what (if anything) to adapt.
"""
import json, os, re, shutil
from pathlib import Path

import chess
import chess.engine
import torch

UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")
MAX_PLIES = 250  # ply cap used by the judge (per your notes)


# ----------------------------------------------------------------- vocab / model
class Vocab:
    def __init__(self, tok2id):
        self.tok2id = dict(tok2id)
        self.move2id = {t: i for t, i in self.tok2id.items() if UCI_RE.match(t)}
        self.id2move = {i: t for t, i in self.move2id.items()}
        self.bos = None
        for k in ("[BOS]", "<bos>", "<BOS>", "<s>", "BOS"):
            if k in self.tok2id:
                self.bos = self.tok2id[k]
                break
        if len(self.move2id) < 1000 or self.bos is None:
            raise RuntimeError(
                f"Vocab not recognised ({len(self.move2id)} UCI tokens, bos={self.bos}). "
                "Edit common.load_vocab() to build {token: id} for your tokenizer.")


def load_vocab(path):
    path = str(path)
    tok2id = {}
    try:
        from transformers import AutoTokenizer
        tok2id = AutoTokenizer.from_pretrained(path, trust_remote_code=True).get_vocab()
    except Exception:
        pass
    if not any(UCI_RE.match(t) for t in tok2id):
        vj = Path(path) / "vocab.json"
        if vj.exists():
            tok2id = json.load(open(vj))
    return Vocab(tok2id)


def load_model(path, device="cpu"):
    path = str(path)
    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    except Exception:
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(path)
    return model.to(device).eval(), load_vocab(path)


def logits_of(model, ids):
    out = model(input_ids=ids)
    return out.logits if hasattr(out, "logits") else out[0]


@torch.no_grad()
def choose_move(model, vocab, board, hist_ids, device):
    """Judge decoding: top-1 -> top-2 -> top-3, first legal. Returns (uci|None, rank)."""
    ids = torch.tensor([[vocab.bos] + hist_ids], device=device)
    top = torch.topk(logits_of(model, ids)[0, -1], 3).indices.tolist()
    legal = {m.uci() for m in board.legal_moves}
    for r, t in enumerate(top):
        u = vocab.id2move.get(t)
        if u is not None and u in legal:
            return u, r
    return None, 3


def save_checkpoint(model, base_dir, out_dir):
    """Copy base_dir (config, tokenizer...) and write model.safetensors with the SAME keys as base."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    out_dir = Path(out_dir)
    shutil.copytree(base_dir, out_dir, dirs_exist_ok=True)
    with safe_open(str(Path(base_dir) / "model.safetensors"), "pt") as f:
        keys = list(f.keys())
    sd = model.state_dict()
    missing = [k for k in keys if k not in sd]
    if missing:
        raise RuntimeError(f"keys in base file not in model.state_dict(): {missing[:5]}")
    save_file({k: sd[k].detach().cpu().clone().contiguous() for k in keys},
              str(out_dir / "model.safetensors"), metadata={"format": "pt"})


# ----------------------------------------------------------------- stockfish
def open_sf(path=None, hash_mb=16):
    path = (path or os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")
            or os.path.expanduser("~/.local/bin/stockfish"))
    eng = chess.engine.SimpleEngine.popen_uci(path)
    eng.configure({"Threads": 1, "Hash": hash_mb})
    return eng


# ----------------------------------------------------------------- opponent engine
_VAL = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500,
        chess.QUEEN: 900, chess.KING: 0}
_MATE, _INF = 100000, 10**9
_warned = False


def _material(b):
    return sum(v * (len(b.pieces(pt, b.turn)) - len(b.pieces(pt, not b.turn))) for pt, v in _VAL.items())


def _order(b, moves):
    def key(m):
        if b.is_capture(m):
            victim = 100 if b.is_en_passant(m) else _VAL.get(b.piece_type_at(m.to_square), 100)
            return (0, -victim, m.uci())
        return (1, 0, m.uci())
    return sorted(moves, key=key)


def _qs(b, a, beta):
    stand = _material(b)
    if stand >= beta:
        return beta
    a = max(a, stand)
    for m in _order(b, [m for m in b.legal_moves if b.is_capture(m)]):
        b.push(m); s = -_qs(b, -beta, -a); b.pop()
        if s >= beta:
            return beta
        a = max(a, s)
    return a


def _ab(b, d, a, beta, ply):
    moves = list(b.legal_moves)
    if not moves:
        return -_MATE + ply if b.is_check() else 0
    if d == 0:
        return _qs(b, a, beta)
    for m in _order(b, moves):
        b.push(m); s = -_ab(b, d - 1, -beta, -a, ply + 1); b.pop()
        if s >= beta:
            return beta
        a = max(a, s)
    return a


def builtin_engine_move(b, depth):
    """APPROXIMATION of the official engine (material alpha-beta + quiescence). Slow in pure Python."""
    best, a = None, -_INF
    for m in _order(b, list(b.legal_moves)):
        b.push(m); s = -_ab(b, depth - 1, -_INF, -a, 1); b.pop()
        if best is None or s > a:
            best, a = m, s
    return best


def engine_move(board, level):
    global _warned
    mv = None
    try:
        from plan import engine_hook
        mv = engine_hook.engine_move(board, level)
    except ImportError:
        pass
    if mv is None:
        if not _warned:
            print("[warn] using built-in approximate engine; wire plan/engine_hook.py to the official one")
            _warned = True
        mv = builtin_engine_move(board, level)
    return mv


# ----------------------------------------------------------------- game status
def status(board, our_color, max_plies=MAX_PLIES):
    """None if game continues, else (result_for_us, reason)."""
    if board.is_checkmate():
        return (0.0 if board.turn == our_color else 1.0), "checkmate"
    if board.is_game_over(claim_draw=True):
        return 0.5, "draw"
    if len(board.move_stack) >= max_plies:
        return 0.5, "cap"
    return None


# ----------------------------------------------------------------- openings
def _parse_line(line):
    b = chess.Board(); out = []
    toks = [t for t in re.split(r"\s+", line.strip()) if t and not re.match(r"^\d+\.+$", t)]
    for t in toks:
        t = re.sub(r"^\d+\.+", "", t)
        try:
            m = chess.Move.from_uci(t)
            if m not in b.legal_moves:
                raise ValueError
        except ValueError:
            m = b.parse_san(t)
        b.push(m); out.append(m.uci())
    return out


def load_openings(path, n, seed, random_plies=6):
    import random
    rng = random.Random(seed)
    if path:
        lines = []
        for l in open(path):
            l = l.strip()
            if not l:
                continue
            if l.startswith("{"):
                d = json.loads(l); mv = d.get("moves", d.get("opening"))
                lines.append(mv if isinstance(mv, list) else _parse_line(mv))
            else:
                lines.append(_parse_line(l))
        return rng.sample(lines, min(n, len(lines)))
    res = []
    while len(res) < n:
        b = chess.Board(); mv = []
        for _ in range(random_plies):
            m = rng.choice(list(b.legal_moves)); b.push(m); mv.append(m.uci())
            if b.is_game_over():
                break
        else:
            res.append(mv)
    return res
