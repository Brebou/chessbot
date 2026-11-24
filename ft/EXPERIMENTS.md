# ft — experiment ledger and status

Progress tracker for the post-training work. Updated after every run or
merged PR. Judged spec: the mix ratios, loss, eval protocol, and stop
criteria implemented under `ft/`. Refined acceptance rules and the
instruments they use are in `ft/eval/` (paired eval + frozen fixtures).

## Status at a glance

| Item | State |
|---|---|
| Submission candidate | `base_model` (paired score 0.231) |
| Fallback | `results/sft/run1/final` (0.210) |
| Checkpoint qualification rule | `balanced − 2σ > 0`; nothing qualified (10 post-training runs, ~60 evals) |
| Local suite | 142 passed / 7 skipped (#41–#50 + dense-soft-loss tests; pending merge) |
| Current stage | run5 **killed**; poc/run7 flat-negative steady state over 16 iterations; decision pending: run8 (rehearsal-share amendment) or stop and submit `base_model` |

## How models are evaluated (3 instruments, all paired vs base)

1. **Paired engine games** (Elo-like): full games vs the official alpha-beta
   engine, levels 1–3, fixed openings/seeds both colors (~240 games);
   delta + cell-bootstrap sigma. FastEngine for L1–3 (move-identical, ~4×)
   [level 0 removed by decision mid-campaign (run7b clean restart): the
   official metric is (S1+S2+S3)/3, never used L0, and vs-random deltas
   sit at the score ceiling where cells are coin flips — pure noise;
   `--eval-levels` restores it if ever wanted].
2. **Paired self-play proxy** (tournament proxy): 200 openings × 2 colors
   vs base. `balanced = mean(engine, proxy)` drives selection/plateau.
3. **Frozen fixtures** (single forward pass, sub-second): 500 verified
   mate-in-1 + 500/bin (mid/late/end) from real games. Noise bands:
   mate1 ±1.2 pts, captures ±5 pts, legality ±1 pt.

## Iteration-1 experiment matrix (all runs `--mix` from data/ft8h)

| Config | mate1 (base 4.8%) | win-cap late (base 64.1%) | engine Δ | proxy Δ | qualified |
|---|---|---|---|---|---|
| mix, hinge on (run1b, 2 ep; run1c 3.5 ep) | ~8.5% sat. | 58.7–60.4% | ~−0.03 | ~−0.047 | none (×8) |
| mix_w2 (`--bucket-weight mate=2.0`), hinge on (run2a) | 10.2%, rising at abort | 55.6–57.0% | ~−0.033 | −0.058 | none |
| mix, hinge off (`--lambda-contrast 0`) (run2b) | 8.4–8.8% | 58.2–62.5% | ~−0.03 | −0.015…−0.054 | none |
| mix_w2, hinge off (run2c / B2) | 10.0% sat. | 55.7–58.5% | −0.03…−0.048 | −0.031…−0.056 | none |
| mix_it2 = mix + 5.8% DAgger rows, 18,049 on-policy states, engine-labeled (run3) | 8.4–9.2% | 59.0–59.5% (not recovered) | −0.042 → **−0.0156** (closest-to-zero of the campaign, ~1.1σ) | −0.039…−0.034 | none |
| mix_it2b = mix + 15%-capped DAgger-2 (32,114 CE rows → normal + 20,827 on-policy contrastive pairs; 87,860 states, depth 10), from base (run4) | 7.4–9.6% | 56.5–59.0% (not recovered) | −0.033 → −0.048 (monotone decline; best eval −0.0297) | −0.030…−0.046 | none |
| mix_it2 continued on the box well past the pre-registered 5 epochs (run5, ~27 epochs, 288-step eval cadence) | 6.6–9.6% | 58.2–62.8% winning-cap late (losing-cap late 20–26%) | −0.040 → −0.081 (monotone decline; epoch-2 local best −0.0247) | −0.034…−0.089 | none — killed per pre-registration |

Robust conclusions (946k params):
- mate1 dose-responds to CE weight (4.8 → 10%) then saturates; the
  contrastive hinge is inert on fixtures and its removal changes nothing.
- Every variant of the off-policy elite mix costs ~4–6 pts of late-bin
  capture play and ~2–5% head-to-head score → checkpoint selection
  correctly refuses everything; off-policy SFT is exhausted as an Elo lever.
- Legality is banked in the base (98% global / 95% end on our fixtures),
  preserved under all runs; monitor via guards only, never optimize.
- On-policy model-negative pairs are directionally negative at this
  capacity: run4 vs run3 at the matched step-1152 eval is engine −0.0453
  vs −0.0156 (~1–2σ — noisy, not conclusive), while `val_ce_contrastive`
  *improves* (3.66 → 3.30): the model fits its own visited states'
  negatives readily, but CE-pulling toward off-manifold SF-best at exactly
  the states its current policy produces does not verifiably help. The
  pure-CE on-policy dose question is NOT settled by run4 (run4 mixed in
  pairs and a 2.9× CE-share jump); run3 remains the only run with a
  monotone improving engine trend, stopped by epoch budget at its best
  eval.
- The one robust real improvement of the campaign is mate-in-1 (4.8% →
  7.4–10.2% across all variants); it never converts into engine score.
- poc/run7 (16 iterations, KL-anchored corrections-only) is the least-bad
  post-training update of the campaign: steady state −0.011 vs base,
  far better than any corpus SFT run (−0.03…−0.08), but still negative.
  The course's own on-policy baseline (examples/onpolicy_legality.py,
  0.225 → 0.256) reaches a positive delta only with on-policy games
  mixed with distillation AND Lichess replay. The remaining
  evidence-backed dose to test is therefore a rehearsal/distillation
  share inside the loop corrections (candidate run8); without it,
  corrections-only training is bounded flat-to-slightly-negative.

## Decision rules (pre-registered)

- Checkpoints qualify only via `balanced − 2σ > 0`; fixtures are diagnostics.
- Protocol amendment (user-directed, run7b clean restart): paired evals
  now cover levels 1–3 only (L0 dropped — never part of the official
  score, and its ceiling deltas only injected variance). Pre-registered
  pass/kill wording is unchanged; the first run7 block (16 iterations,
  pre-#41) remains ledgered under the old definition for the record.
- Protocol amendment (user-directed, run7b rollouts): game generation is
  no longer full self-play. Per iteration: 1/3 pure self-play, 2/3 model
  vs the FastEngine class-engine lookalike (half model-white, half
  model-black), engine level sampled per game at L1 10% / L2 30% /
  L3 60% — so the correction signal balances "don't regress vs a
  base-model-level opponent" with "learn to beat the class engine".
  Rollout mix is adjustable (`--engine-share`, `--opp-levels`,
  `--opp-level-probs`) and injectable (`game_fn`) for tests; eval
  protocol, seeds, and pass/kill wording unchanged. Effective from
  run7b; all earlier run7 iterations were pure self-play and stay
  ledgered as such.
- Protocol note (diagnostics cadence): fixture metrics are diagnostics
  only (never part of qualification); `--fixtures-every N` (default 1)
  thins them to every Nth iteration, always keeping the first — does
  not change the eval protocol or pass/kill criteria.
- Pre-registered **run8** (dense soft-policy loss, user-directed 2025):
  run7b recipe unchanged (mixed rollouts, 1 epoch/iter, KL 0.1,
  corrections >150 cp) **plus** a dense soft-CE bucket for states whose
  eval drop is in (35, 150] cp: targets = root multipv table softmax at
  T=100 cp (played move kept when present in the table, estimated
  cp_best − drop otherwise; per-state weights sum to 1); ALL such states
  selected; whole-bucket loss mass = (m/n)/8 of the correction states'
  mass (m = correction states, n = dense states; per-row weight =
  softmax × λ). Flags: `--dense-cp 35 --dense-temp 100
  --dense-mass-div 8` (dense off by default → run7b resumes stay
  comparable). Pass/kill wording unchanged: pass = balanced − 2σ > 0 or
  corrections/iteration shrinks; kill = captures < 55% or balanced
  ≤ −0.05 at iteration 4. New monitor: `val_ce_dense` (own bucket) plus
  the run5 signal `val_ce_normal` drift.
- Dense-loss selection rationale (signal design, user-directed):
  sub-optimal-but-not-blunder states = the sparse-gradient gap —
  corrections never touch them and single-target CE would over-pin one
  move where several are near-equal; soft CE redistributes mass between
  near-equal moves instead of punishing ("punish must also build").
  Cost note: all-state selection grows steps/iter ~3–5× (GPU-cheap,
  wall-clock visible).
- Fixtures deltas only count beyond their noise bands (see above).
- Note: contrastive rows already carry full CE toward the good move at
  weight 1.0 (`ft_loss` runs `weighted_ce` over the whole batch) — "punish
  must also build" is the implemented shape, no extra term needed.
- DAgger OOD detector is inert at the (pieces, material-balance)
  signature granularity (n_ood = 0 everywhere); harmless — the share cap
  never trims at 18k rows.
- Stage-2 (run3) started from `base_model`, *not* the failed SFT policies:
  no SFT checkpoint earned the chain, and re-exposure to the same mix
  doubles the drift the proxy metric penalizes.
- Escalation ladder if run3 is flat: (a) add CE-on-good to contrastive
  pairs + model-generated negatives from the same DAgger rollouts
  (“punish must also build”), (b) legal-masked CE, (c) soft targets from
  annotated `cp_second`. If none converts, submit `base_model`.
- Pre-registered **run5** (extend the only improving trend): mix_it2
  (run3's exact 5.8% DAgger blend), from `base_model`, epochs 5,
  `--eval-every-steps 288`, `--plateau-patience 3`, lr 1e-4. run3 was
  stopped by epoch budget at its best eval (−0.0156), never by plateau.
  Kill: if no improving engine trend beyond run3's by end of epoch 3.
  **Verdict: KILLED.** By end of epoch 3 the engine delta was −0.0484 and
  it declined monotonically to −0.081 by epoch 27; `val_ce_normal` rose
  2.63 → 3.55 (drift away from the elite mix), legality-end slipped
  0.95 → 0.91. The run3 trend did not extend; it was measurement noise
  around noise, not a real gradient.
- Pre-registered **run6** (full on-policy correction PoC, corpus-
  independent stabilization): score the model's move on every model-≠-SF
  state of dagger2 at depth 6 (`--build-poc-mix`); a state is a correction
  row iff the played move loses > 150 cp, misses a forced mate in ≤ 2, or
  fails to convert any forced mate with ≤ 10 pieces; everything else is
  dropped. Train corrections-only (`--lr 5e-5`) with a real KL anchor:
  forward KL(base ‖ policy) at the prediction position, λ = 0.1
  (`--kl-lambda`), frozen base reference computed from `--base-path`.
  Prediction: ~15–30% of the 87,860 states classify as blunders. Pass:
  first checkpoint with `balanced − 2σ > 0`, or a sustained trend with
  captures-late held ≥ 62%. Kill: KL fails to hold both by epoch 4 of a
  381-step/epoch schedule.
- Pre-registered **run7** (clean-room toy loop, `ft/sft/poc_loop.py`): no
  reuse of dagger2 or any collected data.
  **Interim verdict after 16 iterations (neither pass nor kill fired):**
  the box run deviated from the pre-registration in two ways — epochs
  defaulted to 2 per iteration (not 1) and fixtures were off, so the
  captures-late < 55% kill cannot fire. 31 paired evals vs base: mean
  balanced −0.0109 (sd 0.0110, SE 0.0020 → significantly below 0),
  end-of-iteration range −0.0281…+0.0031, sigma ≈ 0.015 →
  `balanced − 2σ > 0` never met; it4 balanced = −0.0056 → kill did not
  fire. Corrections stable at ~13.2k rows/iteration (~30% of ~44k states,
  0 failed scores) — no shrinkage over 16 iterations. Level-1 delta is
  the worst (mean ≈ −0.034, dips to −0.13). The KL anchor visibly holds
  the line — no downward drift over 16 iterations, unlike every corpus
  SFT run — but corrections-only training produced no gain, steady state
  slightly negative. Each iteration plays ~500 fresh
  full self-play games with the current policy (`stop_pieces=0`, openings
  8 plies, 250-ply cap), scores every visited state's played move at depth
  6 (multipv 5, before/after mover-POV), and keeps corrections only for
  the three failure classes: eval loss > 150 cp, missed mate in ≤ 2,
  ≤10-piece failure to convert a forced mate within +1 ply of the fastest
  (KRK-style multi-solution mates emit one row per equally-fast mating
  alternative, weight 1/n). Micro fine-tune on corrections only, lr 5e-5,
  forward-KL anchor λ 0.1 to frozen `base_model`; corrections discarded,
  fresh games resampled with a new seed each iteration. Pass: the loop's
  paired balance-vs-base becomes non-negative across iterations (or
  corrections/iteration shrinks); Kill: captures-late regresses below
  55% or balanced is still ≤ −0.05 at iteration 4. Iteration estimate:
  10–20 min on 22 workers.
- Ladder rungs (b) legal-masked CE and (c) soft-target `cp_second` remain
  declined: (b) legality is 97–100% top-1, no headroom; (c) reweights the
  same off-policy CE signal every run shows degrades engine play.
  `base_model` (0.231) stays the submission until run5/run6 qualify.

- run4 pre-flight crash: `rollout_positions`/`dagger_collect` wrapped the
  `progress` object as `show` instead of its `.show` method (fast path
  untested — CLI progress only); fixed with a stub-object regression test.

## Run log

| Run | Config | Outcome |
|---|---|---|
| sft/run1 (pre-ft) | SFT mix v1 | 0.210 -- kept as fallback |
| ft/run1 | mix, 2 ep | epoch bug (pre-fix), stale artifact, deletable |
| ft/run1b | mix, 3 ep, patience 3 | plateau abort, 4 flat evals |
| ft/run1c | mix, 7 ep, patience 7 | aborted at 3.5 ep; val CE turns over at den 3 |
| ft/run2a | mix_w2, patience 4 | mate1 10.2% rising; aborted by balanced plateau |
| ft/run2b (B1) | mix, no hinge | no change vs run1b |
| ft/run2c (B2) | mix_w2, no hinge | mate1 10% sat., captures not recovered |
| ft/run3 | DAgger from base (annotation pool) | 18,049 states collected+annotated (depth 10); 0 qualified; engine trend nearest zero of campaign; captures not recovered |
| ft/run4 | mix_it2b (DAgger-2 15% cap: CE→normal + on-policy pairs), from base, 3 ep | 1,728 steps, 6 evals; balanced declines −0.033 → −0.047; engine best −0.0297; ~1–2σ below run3 at matched step-1152; no qualification |
| ft/run5 | mix_it2, from base, 5 ep (pre-registered); extended to ~27 ep on the box; two identical launches appended to the same metrics.jsonl (GPU-float noise only) | **killed per pre-registration**: balanced −0.040 → −0.081 monotone decline; engine never better than run3's −0.0156 (local best −0.0247 at epoch 2); val_ce_normal 2.63 → 3.55; legality-end 0.95 → 0.91; captures-late held ~0.58–0.63 but under the 62% pass bar |
| ft/run6 | poc corrections-only + KL anchor (λ 0.1, lr 5e-5) | **pending** |
| poc/run7 | clean-room continuous loop (fresh games, corrections, KL); first 16 iterations on the pre-#41 loop, epochs defaulted to 2/iter, fixtures off | **flat-negative, inconclusive**: 31 evals, mean balanced −0.0109 (SE 0.0020, range −0.028…+0.009), 0 qualified, no kill trigger (it4 −0.0056); corrections stable ~13.2k/iter, no shrinkage; KL anchor prevents drift but yields no gain; L1 the worst level |

### run9 (auto-research sandbox sweep, CPU, run7-protocol corpus)

Corpus-first protocol: one SF-scored depth-4 corpus shared by all configs,
fixed eval seed, shared base records cache, every run in
`ft/research/results.jsonl`. Sweep A (auto9, 15,160 states, 5,204
corrections) + Sweep B + batch-u + auto10 (24,340/8,369, t5-policy states)
+ batch-w + batch-x + auto11 (19,674/6,269, engine-heavy 400 games) +
batch-y. ~24 arms total. Champions re-scored on independent cells
(confirm seed 999 / 555, openings 20).

Verdicts:
- **Recipe constants are a dead end at this scale**: rehearsal 0-5x,
  KL 0-0.5, lr 3e-5-4e-4, batch 256-512, epochs 2-12 (n1: 3x t5 dose,
  234 steps => -0.027), cp-threshold 100 (material curriculum), mate
  curriculum (proactive mate<=3/5 targets), decode-aware cheap-flip
  filtering (SF-best in base top-3: 58% of states), corrections-only vs
  corrections+distill, SF distill at volume (d1-d3: -0.027..-0.067
  despite 44-56% top-1 movement) -- all within noise or negative.
- **Generation-2 continuation (x1: t5 recipe from t5-final, KL/eval on
  original base) does not compound** (engine +0.008 in-sweep, < gen1).
- **Engine-coverage-dense corpus (auto11, 400 engine games) does not
  unlock the regime** (y2 in-sweep +0.042 confirmed -> +0.000 on seed 999).
- **Survivor: t5 recipe** (on-policy corrections + 1:1 rehearsal + KL
  0.1 + lr 5e-5 + 2-4 ep). Two independent engine cells: +0.029 (999) /
  +0.004 (555), avg +0.017; balanced +0.002 on both seeds. Positive but
  far below qualification (needs balanced - 2sigma > 0 at larger eval).
- Root cause hypothesis for the plateau: sandbox arms all move ~15k-24k
  depth-4-labelled states in <2,000 steps; the only demonstrated positive
  lever in the whole project remains scale (course-positive on-policy +
  distill + replay reached +0.03 at ~500k positions on GPU).

Pre-registration for run10 (GPU box): repeated-generation t5-recipe loop
at 10-20x sandbox scale (fresh 30k-50k games/gen, SF depth 8 labels,
KL 0.1, lr 5e-5, 2 ep, corrections+rehearsal 1:1, generation models
accumulating); qualification only on fresh-seed 40-opening + 200-game
cells. Sandbox confirms +0.017 is the floor to beat.

## ARC2 (open-loop, sandbox): book-aligned on-policy corrections — verdict so far
The official eval plays paired games from a 31,369-entry bundled opening
book; previous corpora sampled random opening plies (distribution
mismatch). Summary of the runs so far (sandbox 10-20 openings, 20
self-play games; sigma ~0.03):

| arm | recipe | engine | proxy | balanced | verdict |
| s1   | heuristic safety corpus + rehearsal | +0.0083 | −0.025 | −0.008 | flat, no engine gain |
| a1   | t5 on book corpus (full) | +0.0083 | +0.025 | +0.017 | marginal |
| a2   | cp_threshold 100 | +0.0083 | −0.050 | −0.021 | flat |
| a3   | kl 0.3, lr 1e-4 | +0.0083 | −0.050 | −0.021 | flat |
| y2   | mate_curric 3.0 + cp 100 + kl 0.3 | +0.0417 | −0.0625 | −0.010 | best engine seen, but noisy; confirm failed on fresh seed (−0.018) |
| t5   | (run9 survivor) | +0.0042 | 0.0 | +0.002 | confirm on fresh seed 555: +0.004/-0.008 — floor, not a win |
| b1   | + impact_weight by eval gap | −0.0167 | −0.075 | −0.046 | NEGATIVE — impact weighting hurts (overweights rare cascades) |

### Batch c (pre-registered, queued behind b-batch)
- **c1**: L1 correction upweight x3 (base headroom is at L1: 0.167 vs 0.069/0.031). Pass: engine delta ≥ +0.04. Kill: negative.
- **c2**: mirror/color-symmetry augmentation (every row duplicated on `Board.mirror()`). Free 2x data. Pass: delta ≥ +0.03 at same step count.
- **c3**: L1-gain x2 + rehearsal restricted to L1 states only.
- **c4**: mirror + L1-gain combo at cap 16k.
- **c5**: rehearsal dose 2.0 (tests dose–response of grounding rows).

### Batch d (pre-registered, REINFORCE-lite outcome weighting)
Fresh book games vs the class engine per arm (never reused): WINNING
games -> CE rows pushing UP the model's own played moves (w_win);
LOSING games -> CE rows to the SF best move when the played move was a
blunder (w_loss), else low-weight own-move rows; DRAWS -> w_draw.
- **d1**: win:loss:draw = 1 : 1 : 0.3, levels {1,2}; **d2**: levels {1} only, w_loss 1.5, w_draw 0.
- Pass: engine delta ≥ +0.04. Kill: negative engine delta.
Rationale: outcome selection is the policy-gradient surrogate — own
moves from wins find their way into future winning lines; losing
states get real corrections. b1's failure rules out arbitrary impact
weighting; outcome filtering is a much sharper axis (wins/losses, not
eval gaps).

### Batch c verdicts (sandbox, 10 openings, sigma ~0.03)
| arm | recipe | engine | proxy | balanced | verdict |
| c1  | L1 corrections x3 | −0.033 | −0.050 | −0.042 | NEGATIVE — simple L1 upweighting hurts |
| c2  | mirror augmentation 2x rows | 0.000 | −0.038 | −0.019 | flat |
| c3  | L1 x2 + L1-only rehearsal | +0.008 | −0.075 | −0.033 | flat engine |
| c4  | mirror + L1 x2, cap 16k | (OOM-killed at 31k rows — memcg 4 GB; rerun with cap ≤ 20k if ever retried) | | | |
| c5  | rehearsal dose 2.0 | −0.033 | −0.063 | −0.048 | NEGATIVE |
| e1  | curriculum: corrections epoch -> own-move rehearsal epoch | +0.033 | −0.088 | −0.027 | best engine delta of the batch family, but proxy crushed; |not qualifying|
Learning so far: on the book corpus every single leverage (dose,
weighting, targeting, augmentation, ordering) is within noise or
negative. Consistent with run9's lesson: capacity binds, recipe
details don't matter at this scale. The remaining macro question is
ITERATION (it2): do corrections compound across generations of
self-play, or decay? it2/genN rows answer directly (engine delta vs
base AND vs previous gen).

### Batch d verdicts (REINFORCE-lite outcome weighting)
- **d1 (win:loss:draw = 1:1:0.3, levels {1,2}): engine +0.058, proxy −0.013,
  balanced +0.023** — best engine delta of the whole arc2 campaign, first
  POSITIVE balanced score. Sigma 0.034 -> not yet qualifying (needs +0.068),
  but the axis works. Notable: base wins only 15/800 games (642 losses,
  143 draws) -> winning-game rows are scarce; the signal is dominated by
  corrections-on-losing-games (4,921 corrections from 15,923 losing
  states) + outcome selection. Hypothesis: dropping negative-free rows and
  focusing corrections on games the model would otherwise LOSE (rather
  than every blunder state) concentrates the gradient on decision-
  relevant positions. CONFIRMATION NEXT: fresh eval seed, more openings.
- d2 queued: L1-only, w_loss 1.5, w_draw 0 — outcome concentration test.

### d1 fresh-seed confirmation
- seed 777 (14 openings): engine **+0.042** (L1 +0.071 / L2 +0.054 / L3 0.0),
  proxy −0.069, balanced −0.014
- seed 555 (14 openings): engine +0.006 (L1 −0.018 / L2 +0.054 / L3 −0.018),
  proxy −0.063
- 3-seed mean engine delta ≈ **+0.035**, positive at L1+L2 in 2/3 seeds —
  the gains localize exactly where the recipe trains (levels 1-2).
  NOT seed-luck (y2 was). Proxy consistently slightly negative.

### it3 (pre-registered): REINFORCE-lite ITERATED
d1 recipe in a compounding loop (100 openings x levels {1,2} per gen,
fresh games each gen, corrections only from losing games, 1 epoch/gen).
- Pass: gen3 vs-base engine delta > gen1's AND per-level L1/L2 trend up.
- Kill: gen2 vs_prev <= 0 and gen3 vs_base <= gen1 (no compounding -> the
  recipe is a one-shot data-select, not an algorithm).

## RUN10 (GPU box, executed by user): iterated t5-recipe, 5 generations
Command: sweep --stage all --configs x1 --epochs 2 (4 eval epochs reported
per gen) --n-selfplay 1200 --n-engine 800 --sf-depth 8, anchor = true base;
each generation trains from the PREVIOUS generation's final model on a
FRESH corpus of on-policy games (corpus regenerated per gen, 65-70k rows),
paired-eval vs the TRUE BASE on a per-gen opening subset (seed varies).

| gen | trained from | rows | final engine d vs base | balanced | notes |
| 1 | base   | 70,359 | -0.0146 | -0.001 | flat; training KL drift 0.17->0.41 |
| 2 | gen1   | 65,350 | **+0.0396** | **+0.026** | epoch-4: L1 +0.100, official mean 0.1083 vs 0.0687; sigma 0.0199 -> missed qualification by **0.0002** |
| 3 | gen2   | 64,140 | -0.0250 | -0.015 | decay |
| 4 | gen3   | 65,095 | +0.0104 | -0.024 | proxy -0.058 |
| 5 | gen4   | 64,627 | -0.0521 | -0.039 | worst; L1 -0.131 |

Read:
- Generations STACK (chained weights) with fresh on-policy corpora; no
  independent re-measure of gen2's gain on other cells yet -> CONFIRM on
  2-3 fresh seeds (ft/research/confirm.py, eval-only, minutes).
- gen2 gain concentrated at L1 (+0.100) — same localization as the
  sandbox d1 REINFORCE arm (L1/L2 local +0.07/+0.05). Two independent
  setups now agree: the winnable headroom is at level 1-2.
- Decay after gen2 -> accumulated drift (train KL 0.4-0.53 by gen 4-5)
  outweighs fresh corrections. Cap chains at 2 generations unless a
  re-anchored consolidation gen is added.
- run7's rule stands: eval-seed variance between gens can inflate a
  single-cell delta; only multi-seed confirm counts.
NEXT (run11 proposal): REINFORCE-lite recipe at box scale
(corrections ONLY from losing games + own-move rows from winning games),
from base AND from run10/gen2 as starting points.

### it3 chain: does REINFORCE-iteration compound in the sandbox? NO.
- gen1 (vs base): engine +0.008 (L1 +0.075)
- gen2 (vs gen1): engine -0.0417 (vs_prev -0.0417) -> regression
Same shape as run10 on the box: one lucky step then decay. Standing
conclusion: ITERATION DOES NOT COMPOUND for chains > 2 gens. The recipe
family (corrections+KL) extracts a one-shot fix and then drift wins.
Remaining research axes with headroom:
1. f-batch (sandbox, dose-response of the d1 recipe at 400 games:
   f1 games-dose, f2 loss-weight, f3 corrections-only) -> informs run11/12
   starting-point choice.
2. run11 (box): d1 recipe at 600 games from run10/gen2.
3. run12 (box): it3 chain at 600 games/gen but STOP at gen2.
All gated on the confirm tool: no promotion without 3-seed confirmation.

### it3 FINAL (chain complete): no compounding, chain ends below base
gen1 (+0.008 w/ L1 +0.075) -> gen2 (-0.042) -> gen3 (vs prev +0.017,
vs base -0.025). Combined with the box's run10 (gen1 flat -> gen2 luck
-> gen3-5 decay): correcting on fresh on-policy games is a ONE-SHOT
extract; iterating the chain drifts below base by gen 2-3 in both
recipes (t5-iterated and REINFORCE-iterated). Rules out the whole
"self-improvement loop" family at this model scale.

### batch g (pre-registered): trajectory (blunder-path) corrections
For each losing game, locate the largest one-ply drop (cp_best -
cp_model) and correct up to the 3 preceding model moves as well
(target = SF best at each state, weight 0.25 x w_loss, deduped against
regular rows). g1 = f1 + trajectory; g2 = trajectory-only corrections.
Rationale: single-position CE corrects the TIPPING move; the setup
moves keep pulling the game back into the same losing line. Path rows
give a denser gradient on decisive games.
- Pass: engine delta >= +0.05 at 400 games (f-caliber N).
- Kill: < 0.

### f1 verdict (2x-game dose of d1): DEAD
1600 games at levels {1,2} (25 wins / 1249 losses / 326 draws), 9,307
corrections from 30,648 losing states, 1 epoch over 20k rows:
final engine 0.000, proxy -0.050, balanced -0.025.
The games-dose axis does NOT help; d1's +0.058 does not replicate at
2x games/1 epoch -> d1 was substantially noise + epoch-count luck.
Batch g (trajectory corrections) chained; f2 (loss-weight 1.5),
f3 (corrections-only) still running.

### run13 Arm 1 (run5: mix_it2 5 epochs): KILLED at box scale
Box T4, 308k rows x up to 5 epochs, plateau-abort fired at epoch 4
(balanced < 0.02 for 3 checkpoints): -0.031 / -0.008 / -0.031, L1 always
negative, best_score None. The run3 sandbox "improving trend" was censored
noise, not a trend. The off-policy SF-label corpus family (runs 1-5, 300k
rows) is now fully closed: dose does not rescue it.
Remaining bracket: Arm 2 (REINFORCE 3-seed x 600 games) and Arm 3
(2-gen chain); direction set by Phase-2 confirm gate.

### batch f verdicts (sandbox, 400 openings = 1600 games/level-pair)
| arm | recipe | engine | proxy | balanced | verdict |
| f1 | d1 @ 2x games, 1 ep | 0.000 | -0.050 | -0.025 | games-dose dead |
| f2 | d1 + w_loss 1.5 | +0.033 | -0.025 | **+0.004** | only non-negative balanced |
| f3 | corrections only (no win/draw rows) | +0.017 | -0.100 | -0.042 | corrections alone CRUSH self-play coherence |
Reading: the win-row leg (pushing up own moves in won games) is the anti-
collapse term; without it the model degrades its own play. d1's recipe
mean across doses is +0.01..+0.03 engine with sigma 0.034 -> resolvable
only at box scale with 3+ seeds (run13 Arm 2 does exactly this). g-batch
(trajectory rows) auto-launched.

### run13 Arm 2 (REINFORCE 3-seed x 600 games on box): ZERO
Per-seed engine deltas: +0.025 / -0.025 / +0.017 (sigmas .033-.036) ->
mean +0.006. The recipe is indistinguishable from base at 3x sandbox
dose; the L1/L2 localization does not survive seed spread (level deltas
scatter +/-0.1). Sandbox g-batch v1 was INVALID (dedupe bug ate every
trajectory row: "+0 trajectory rows"); fixed (dedupe only vs corrected
fens) and relaunched as results/arc2g.
Tally across the campaign: ~30 arms in 4 recipe families, cumulative
engine delta ~ 0. Structural facts that DID replicate: win-rows are the
anti-collapse term (f3: corrections-only => proxy -0.10); chains decay
after gen 2; eval-seed luck dominates single-cell results.

### run13 Arm 3 (2-gen REINFORCE chain, box): banks nothing real; family closed
- gen1: formal engine 0.000 (sigma 0.036), balanced 0.000. gen2 (on gen1):
  internal epoch evals +0.0333 both checkpoints (L1 +0.025->+0.050, L2
  +0.075 both) BUT formal post-eval engine -0.0083 (sigma 0.0314), proxy
  -0.0500, balanced -0.0292. Same weights, contradictory numbers.
- ROOT CAUSE of internal-vs-formal discrepancy (code-verified, not device):
  trainer-internal epoch evals ran with seed=42 cells (arc2 pass-through),
  the arm's formal eval with eval_seed=1234 cells, confirm with
  777/888/999. Different cell sets -> different games -> deltas scatter by
  pure sampling noise (sigma ~0.035 at 10 openings). There is NO device
  round-off: eval workers always load models on CPU (chess_hw/match.py
  _get_player, device="cpu"), the class engine is deterministic, and saved
  weights are byte-identical. Same-seed same-openings evals of the same
  weights are bitwise-reproducible.
- FIX merged: FTSFTConfig.eval_seed (None = cfg.seed, behavior unchanged);
  train_candidate passes it through, so trainer epoch evals now use the SAME
  cells as the arm formal eval -> epoch-eval numbers become directly
  comparable and no longer manufacture hope. NOTE: all run13 Arm-2/Arm-3
  internal epoch evals printed BEFORE this fix are seed-42 cells and are NOT
  comparable to their formal numbers.
- Chain verdict: capped at gen 2, nothing banked. Chain decay confirmed a
  3rd time (run10, it3, run13). REINFORCE chain axis closed pending the
  Phase-2 3-seed confirm of results/run13/chain/gen2/d1/final (still worth
  running: it is the only box-scale formal check of the family, and +0.033
  twice in-seed made it the best candidate alive).

### fixed g-batch verdicts (sandbox; dedupe bug fixed, rows actually emitted)
| arm | recipe | engine | proxy | balanced | trajectory rows |
| g1 | f1 + trajectory | -0.008 | -0.100 | -0.054 | +1626 |
| g2 | f3 + trajectory | -0.025 | +0.013 | -0.006 | +1601 |
The trajectory-correction axis (fix biggest 1-ply drops including last-3-plies
setup states) is flat-to-negative with rows actually present. DEAD, replicated.

### eval protocol note (binding, from the Arm-3 diagnosis)
Never compare deltas across different eval seeds or opening counts. Campaign
convention going forward: only confirm.py numbers (seeds 777/888/999 x 40
openings) arbitrate promotion; epoch/formal arm numbers only rank candidates
WITHIN the same (eval_seed, n_openings) cell set.

### soup arm (h-batch): weight-average of closed REINFORCE finals — ZERO
Soup s1 = uniform mean of {d1, f2, g1, g2} finals (all base-anchored, same
basin). 10-opening cells: engine -0.017 with L1 +0.10 spike (mirrors single
arms). WiSE-FT blend with base 0.5 (s2) flattens the spike: -0.008, dead.
3-seed 40-opening confirm of s1: +0.031 / -0.017 / +0.002 -> mean +0.005,
qualifies=false all seeds, proxy -0.035/-0.030/-0.020 on all seeds.
Verdict: averaging flat arms gives flat; the L1 spike was averaging luck,
not signal. Direction DEAD. (Code: ft/research/soup.py.)
Also: run14 sandbox prototype OOM-killed once (22 SF workers @ depth 12 +
concurrent soup evals breached the 4 GB cgroup); relaunched at 12 workers.
Lesson: sandbox concurrency budget = ONE heavy job at a time.
