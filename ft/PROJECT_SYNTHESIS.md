# ChessBot post-training — full project synthesis

*Handover document. Written for a fresh contributor (human or model) with GitHub
access who needs to take over the post-training campaign. Everything below is
verified against the repo and `ft/EXPERIMENTS.md` (the running ledger).*

---

## 1. The task

Academic assignment (Part 2 = everything under `ft/`): improve a **fixed,
frozen-architecture ChessGPT via post-training only** to beat (a) a
deterministic alpha-beta + quiescence engine at levels 1–3, and (b) other
students' models in an anonymous Elo tournament. A fresh pre-training run is
**forbidden** — we may only fine-tune / post-train the provided base model.

Deliverable: a single `model.safetensors` + a write-up.

**The only two metrics:**

1. **Official score** `S = (S1 + S2 + S3)/3` where `Sn` = mean win score
   (1 = win, ½ draw, 0 loss) of our model vs the class engine at depth `n`
   (n = 1, 2, 3). This is measurable exactly in our eval harness.
2. **Elo vs other students' models.** Not directly measurable (other models
   private). Proxy: paired self-play vs base. Rule of the campaign: a change
   that helps the engine metric and doesn't hurt the self-play/Elo proxy is
   considered transferable; `balanced = mean(engine_delta, proxy_delta)` is the
   selection statistic used everywhere.

The base model (submission candidate as of this document):

- ~947k params: 4 layers, `n_embd=128`, 4 heads, `n_inner=384`, `n_ctx=256`,
  tied embeddings.
- Vocabulary: 1971 UCI move tokens (+ specials). Input encoding: `[BOS] m1 … mk`
  (the full game history since move 1 is fed at every decision point — there is
  no "position" representation, only move history).
- Decoding at play time: **greedy, then top-1 → top-2 → top-3 by temperature
  order, first legal move kept**. Max 250 plies per game.

Base-model reference numbers (reproducible via the eval harness): paired score
**0.231** overall; per-level engine win-rate roughly L1 ≈ 0.17, L2 ≈ 0.07,
L3 ≈ 0.03 in collector cells.

`chess_hw/` is the **judged, sacred directory**: never modify it. Its
`__pycache__/*.pyc` files are tracked, and pytest re-dirties them, so before
**every** `git add` run `git checkout origin/main -- chess_hw/__pycache__`.

---

## 2. Evaluation protocol (binding — do not work around it)

All comparisons in the campaign use the paired harness in `ft/eval/paired_eval.py`:

- **Paired engine games**: model vs the class engine (SVG/alpha-beta+quiescence,
  `chess_hw` FastEngine for L1–3, move-identical to the official one, ~4×
  faster), starting from a fixed pool of ~31,369 book openings. A *cell* =
  (level, opening_id, model_color). Both base and candidate play the identical
  cells → paired delta, cell-bootstrap sigma.
- **Paired self-play proxy**: 200 cells vs base, both colors — tournament proxy
  (stand-in for the Elo metric).
- Noise measured across the campaign:
  - 10-opening sandbox cells: σ ≈ 0.030–0.035
  - 40-opening box cells: σ ≈ 0.015–0.02
  - 80-opening cells help but cost ~2× wall-clock.
- **Qualification rule**: `delta − 2σ > 0`, and never worse than −0.03 on any
  single level.
- **3-seed confirmation gate** (`ft/research/confirm.py`): a candidate is only
  "banked" if it repeats `delta − 2σ > 0` on ALL of eval-seeds 777/888/999 at
  40 openings, with `balanced > 0` on ≥ 2 of 3. Run:
  `uv run python -m ft.research.confirm --model-path <cand> --base-path base_model/model.safetensors --seeds 777 888 999 --openings 40 --eval-workers 22`
- **Baseline determinism**: the eval is bitwise reproducible — eval workers
  always load models on CPU (`chess_hw/match.py::_get_player`), the class
  engine is deterministic, weights are byte-identical. No device round-off.
  *Never* compare delta numbers across different eval seeds or opening counts
  (this caused one whole spurious-hope episode; see history §5, run13 Arm-3).

The official-metric line printed by the trainer/arm evals is exactly
`(S1+S2+S3)/3` for candidate and base plus their delta — that is the number to
report to the user.

---

## 3. Components and file map

```
ft/
  common/          serialization, tiny helpers
  data/            annotation pipeline (StockfishAnnotator, score_to_mover),
                   bucket building, game selection, tablebase glue
  eval/            paired_eval.py (engine + selfplay paired instruments),
                   metrics.py (frozen fixtures + build_fixtures),
                   summarize.py
  sft/             train_sft.py (FTSFTConfig/FTSFTTrainer — the one trainer
                   everything calls), losses.py (CE + contrastive hinge +
                   anchor-KL), dagger.py (DAgger collection, PoC classifier,
                   corrections builder), poc_loop.py (toy self-improvement loop),
                   dataset.py
  research/        arc2.py (REINFORCE-family arm runner + config registry),
                   sweep.py (train_candidate wrapper + corpus builders),
                   grpo.py (RLOO/GRPO arm — j-batch), soup.py (weight-average
                   arms), confirm.py (3-seed gate), confirm_d1.py,
                   see_arm.py (x1 recipe), iterate2.py / iterate3.py (chain
                   iterations), results.jsonl (all arm rows), EXPERIMENTS.md
  pipeline.py      Part-1-style judged flow (bucket ratios 45/20/15/15/5,
                   BETA=1.0 LAMBDA=0.5 softplus contrastive, cap 8.0)
base_model/        the frozen base (0.231)
tests/             pytest suite (151 passed / 7 skipped as of j-batch)
ft/EXPERIMENTS.md  the ledger — ALWAYS update it with each verdict
ft/research/results.jsonl  machine-readable arm rows
```

`FTSFTTrainer` (`ft/sft/train_sft.py`) supports: normal-CE rows, contrastive
pairs, KL anchor to a frozen reference (`kl_lambda > 0`, real forward
KL(base‖policy) at the prediction position — user explicitly rejected
corpus-row pseudo-KL), `last_k_layers` partial freezing (0 = tune all),
`eval_seed` passthrough (aligns trainer epoch evals with the formal arm cells),
unconditional `peak/` banking of the best balanced-so-far checkpoint
(`peak/` + `best/` under the run dir; `_prune_checkpoints` never touches them),
plateau stop, rollback lr, ≤5 checkpoints, paired + fixture evals.

---

## 4. Campaign matrix (what was tried and what happened)

Scores below are paired-vs-base deltas at the eval resolution of their run;
**only the "confirm" lines had 3-seed status; nothing ever qualified.**

| Batch / run | Idea | Verdict |
|---|---|---|
| run1–4 (pre-history, mixes) | off-policy SF-annotated mixes, eligibility buckets | flat to negative; mate1 4.8→10% real but never converts to score; every elite-mix costs 4–6 pts captures-late |
| run2a | mate weight 2.0 | ↑mate1, ↓captures — the metaphor for single-metric over-optimization |
| run3 | mix_it2 (5.8% on-policy) | best trend −0.042→−0.0156, killed by epoch budget |
| run4 | mix_it2b (+DAgger pairs to 15%) | declined to −0.047; on-policy negative pairs directionally harmful (~1–2σ, noisy) |
| run5 | re-run mix_it2, 5 epochs, KL 0.1 | plateau-killed at −0.035; J-curve hypothesis unconfirmed at this dose |
| run6 | PoC corrections + real KL | designed; superseded by poc loop |
| run7 (poc loop) | continuous self-play→blunder-corrections→KL micro-FT | flat-negative steady state over 16 iterations — corrections-only at low dose doesn't move the score; GPU/eval bottlenecks fixed (shared base cache, eval cadence knobs) |
| run10 (x1, see_arm) | 5-gen chain, big corpus, KL 0.1, lr 5e-5 | all gens ≤ −0.001 engine, no qualification; second-best L1 +0.100 spike at gen2 epoch4 (`best_score: None` — the banking bug) |
| run13 Arm-1 | fresh 300k off-policy corpus, 5 epochs | KILLED (no time-to-signal) |
| run13 Arm-2 (REINFORCE, d/f family) | 3 seeds × 600 games, outcome-weighted correction rows | ZERO (mean +0.006); signal starvation: ~35 winning games out of 2400 |
| run13 Arm-3 (chain, 2 gen) | gen1 d1 +0.033, gen2 +0.033 in-seed | **CONFIRMED DEAD (Phase-2 box confirm, 3 seeds × 40):** engine +0.021/+0.006/+0.002 (≪ 2σ on all seeds), proxy negative all seeds, balanced +0.004/−0.022/−0.016 → qualifies=false ×3. The in-run +0.033 numbers were seed luck. Family closed — 4th spike-is-noise replication |
| d-batch dose | games 400/600/800 dose-response | games-dose dead — d1 at 400 doesn't replicate |
| e1 | curriculum | dead |
| f-batch | corrections-only, loss weights | no arm survives |
| g-batch | trajectory (blunder-path) corrections | g1 −0.008, g2 −0.025 — DEAD, replicated |
| h-batch (soup, soup.py) | weight-average 4 REINFORCE finals; WiSE-FT 0.5 blend | zero: mean +0.005 over 3 seeds, proxy negative everywhere; averaging flat arms gives flat. DEAD |
| i-batch (merged, untested at scale) | last-k-layer-localized FT (`last_k_layers`), i1/i2/i3 configs | **OPEN** — the untested macro |
| run14 | depth-12 SF labels for correction quality | **sandbox cannot fit** (4 GB cgroup OOM ×3 at 22/12/6 workers); box-only |
| j-batch (this PR) | RLOO/GRPO group-relative RL (grpo.py, j1/j2) | **OPEN** — sandbox prototype |

### Robust, replicated facts (don't re-litigate these)

1. Checkpoint qualification `balanced − 2σ > 0` has never fired in ~10 post-training runs, ~60 evals.
2. Chains (fine-tune gen N on gen N−1) **decay past generation 2** — three independent replications.
3. Single-cell / single-seed evals are seed luck; all arbitration via confirm.py.
4. Trainer epoch evals were seed-mismatched with formal arm evals — fixed with `FTSFTConfig.eval_seed` (PR #58). All pre-fix epoch-eval numbers (seed 42) are not comparable.
5. Mate-in-1 rate 4.8→10% is bankable via `mate` rows but **never converts** to engine score.
6. Every off-policy "elite" mix costs captures-late.
7. Legality is already banked at 97–100% top-1 per the fixtures.
8. The contrastive hinge is inert at these scales.
9. Greedy collector = deterministic = identical games per opening → fine for paired eval, fatal for group-relative RL (need sampling — see `SamplingProbe`).
10. Win-rate of the base at collector cells ≈1.7% → positive signal is starved; any outcome-only reward must be *relative within a group* to matter.
11. Sandbox: **ONE heavy job at a time**; depth-12 SF scoring is box-only.

---

## 5. Failure archaeology — why the campaign keeps finding zeros

- **Capacity, not label quality** is the binding constraint (947k params, greedy decoding). Every recipe that pushed probability mass around without changing the regime moved the score by ≲ noise.
- **Sparse signal**: only ~1.7% of collector games are wins; outcome-weighted REINFORCE keeps ~35 positive rows per 2400 games. Both g-batch (trajectory) and h-batch (soup) showed there's no free structure in the corrections.
- **The "J-curve" / more-epochs hypothesis**: partially addressed with `peak/` banking — the +0.033..+0.0396 transient epoch peaks observed in run10/run13 were never stored because checkpoint banking was gated on `significant` (which never fired). Now every trainer run banks its best-balanced checkpoint into `peak/` regardless of significance; the next long/low-lr run (run14b) can harvest them. Longer runs are worth testing WITH banking, not bare.
- **Eval alignment**: seed mismatch manufactured fake signal in run13 Arm-3. Fixed and gated.
- **Phase-2 reality check (box, 3-seed confirm of the best-seeming candidate, run13 chain gen2)**: engine delta +0.021/+0.006/+0.002 across seeds 777/888/999 (mean +0.010, always below 2σ ≈ 0.033–0.037), proxy delta NEGATIVE on every seed (−0.013/−0.050/−0.035), qualifies=false ×3. Every historic +0.03..+0.04 "peak" traces to seed-42 epoch evals (pre-fix) or single-seed formal cells. Rule of thumb: an un-confirmed positive is noise until the gate says otherwise.

---

## 6. The j-batch (RLOO/GRPO) design — current best untested arm

`ft/research/grpo.py` implements group-relative RL for this exact setup:

- `SamplingProbe(FastProbePlayer)`: legal-masked softmax temperature sampling
  (T=0.9) for rollouts; per-game `rng = random.Random(f"{seed_tag}-{oid}-{lv}-{white}-{k}")`
  for replay. Greedy probe is useless here (identical games would give A=0).
- `collect_groups`: K=8 rollouts per (opening, level, color), states collected
  as (fen, history, model_move) like the REINFORCE collector.
- `rloo_advantages`: A_i = r_i − mean(others); constant-outcome groups → 0.
  This converts ALL games into gradient (the friend's insight, formalized).
- `build_grpo_mix`: positive games → own-move CE rows weighted `w_pos·Ahat`
  (Ahat = A/amax per group). j2 additionally scores **negative** games' unique
  states with SF (`poc_score`, `poc_classify` blunder rules) and turns
  blunders into corrections to `best_uci` at weight `w_neg·|Ahat|`
  (no negative CE weights — pushing down exact opponent-dependent moves
  degenerates into opening-avoidance / lose-fast drift).
- `run_grpo_arm` + `GRPO_CONFIGS` (`j1` = pure RLOO, no SF in loss →
  sandbox-runnable; `j2` = with negative corrections). Appends rows keyed
  `run: "grpo/jN"` to `ft/research/results.jsonl`.
- Box run: `uv run python -u -m ft.research.grpo --model-path base_model/model.safetensors --out-dir results/grpo-j1 --configs j1 --device cuda --n-openings 40 --k-rollouts 8 --temperature 0.9 --batch 512 --eval-openings 10 --selfplay-games 20 --eval-workers 22 --engine ~/.local/bin/stockfish`

---

## 7. Environment constraints and workflow

**Sandbox (where the assistant works):** CPU-only, 4 GB cgroup hard limit
(OOM-kills on heavy SF scoring pools — structural for depth-12), 6 GB RAM,
25 cores. Untracked files are lost on sandbox crashes (grpo.py was lost once
and recreated; keep working state committed early). Background jobs need
`setsid + nohup`, and logs should live under the workspace, not `/tmp`.

**GPU box (user runs, Onyxia):** Tesla T4, 25 cgroup cores → `--workers 22`,
`--eval-workers 22`; Stockfish 17.1 at `~/.local/bin/stockfish`; speeds:
2400 collector games ≈ 10 min, 47k-state SF scoring @ depth 8 ≈ 15 min,
train 2 epochs ≈ 10–15 min, eval ≈ 1–2 min → a whole arm ≈ 30–45 min.

**Workflow rules (learned the hard way):**

1. Every change on a fresh branch `vibe/<slug>-87e06f` off updated `origin/main` → draft PR.
2. The user merges PRs between turns → **always `gh pr view` the target PR's merge state immediately BEFORE every push**; late pushes to merged branches get stranded (happened twice; both were rescued by cherry-pick onto a fresh branch off main).
3. Update targeted PRs in place; never delete branches; never open replacement PRs.
4. `git checkout origin/main -- chess_hw/__pycache__` before every `git add`.
5. Multiline PR/issue bodies via quoted HEREDOC + `--body-file`.
6. The ledger `ft/EXPERIMENTS.md` and `ft/research/results.jsonl` are the single source of truth for verdicts; every experiment ends with a row in both.

---

## 8. Open threads and next moves (in priority order)

1. **i-batch** (merged, untested at scale): `for C in i1 i2; do uv run python -u -m ft.research.arc2 --model-path base_model/model.safetensors --anchor-path base_model/model.safetensors --out-dir results/i-batch/$C --reinforce-games 400 --reinforce-seed ibatch-$C --reinforce-device cuda --sf-depth 8 --sf-workers 22 --batch 512 --eval-openings 10 --selfplay-games 20 --eval-workers 22 --stage sweep --configs $C --engine ~/.local/bin/stockfish; done`
   (If corpus.jsonl is missing: pull the latest main — the unknown-config guard was added in #58; old checkouts crash with `FileNotFoundError: results/i-batch/i1/corpus.jsonl`.)
2. **j-batch sandbox prototype** (solo, background) then box j1/j2 (§6).
3. **run14**: depth-12 SF labels for the f2 recipe — box only.
4. **run14b**: 5-gen chain of the f2/REINFORCE recipe at lr 3e-5 + KL 0.1, harvesting `peak/` banks; confirm each peak.
5. **DONE — run13 chain gen2 Phase-2 confirm**: qualifies=false on 777/888/999; REINFORCE chain axis closed.
6. **Decision rule if everything stays flat**: submit `base_model` + the negative-result map. But the honest framing is "ceiling not established": i-batch, run14, and j-batch are untested macros.
7. Deferred: grounded Syzygy TB relabels, legal-masked CE, soft-target cp_second, `WRITEUP.md` authoring.

---

## 9. Submission-state snapshot

- Submission candidate: `base_model/model.safetensors` (0.231).
- Fallback: `results/sft/run1/final` (0.210).
- The one campaign candidate that ever reached the 3-seed gate for real
  (run13 chain gen2) failed it (see §5). Nothing qualifies — `base_model` still stands as the submission.
- Historic transient epoch peaks (+0.033..+0.0396) are now banked automatically
  into `peak/` so future long runs can be mined, but treat them as noise until
  confirmed.
