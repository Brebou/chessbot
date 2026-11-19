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
| Local suite | 134 passed / 7 skipped (#41/#42 add tests; pending merge) |
| Current stage | run5 **killed**; poc/run7 flat-negative steady state over 16 iterations; decision pending: run8 (rehearsal-share amendment) or stop and submit `base_model` |

## How models are evaluated (3 instruments, all paired vs base)

1. **Paired engine games** (Elo-like): full games vs the official alpha-beta
   engine, levels 0–3, fixed openings/seeds both colors (~320 games);
   delta + cell-bootstrap sigma. FastEngine for L1–3 (move-identical, ~4×).
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
