"""Batched rollout generation: policy vs engine and policy vs itself.

All games are advanced simultaneously in lockstep rounds: every model turn
across every active game is batched into a single GPU sampling pass
(``PolicySampler``), while engine replies are computed in-process (or
dispatched to a process pool with ``engine_workers > 0``).

Two game sources, mixed with ``selfplay_ratio``:

- **engine games** (fraction ``1 - selfplay_ratio``): model vs
  ``FastEngine`` (the exact eval opponent) — maximizes the engine score;
- **self-play games** (fraction ``selfplay_ratio``): the policy plays both
  colors — keeps the training distribution tournament-like (the round-robin
  Elo is won against other LLM variants, not against the alpha-beta engine,
  and pure engine optimization is known to distort play style).

Each model decision becomes a :class:`Transition` storing the full move
history (token-level training data is rebuilt from it), the sampled token,
its sampling-time log-prob, its legality, the mover's color and the material
balance (White POV) at the decision position. Game termination follows
``chess_hw.match.play_game``: draw adjudication at ``max_plies``, seeded
random legal fallback when all sampled tokens are illegal.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import List, Optional, Sequence

import chess

from chess_hw.match import MAX_PLIES, load_openings
from diag.common import ProgressLine, material_diff
from diag.fast_engine import FastEngine, engine_best_move_task

from rl.policy import PolicySampler


@dataclass
class Transition:
    """One model decision inside a game."""

    history: List[str]          # UCI moves before the decision (mover to move)
    token_id: int               # sampled token
    legal: bool                 # whether the sampled token was a legal move
    logprob: float              # sampling-time log-prob (PPO "old" log-prob)
    ply: int                    # board ply at the decision
    mover: str = "white"        # "white" | "black" — side that made the decision
    material_white: float = 0.0  # material diff (White POV) at the decision position


@dataclass
class RolloutGame:
    """One finished game with its model-side transitions."""

    transitions: List[Transition] = field(default_factory=list)
    opening_idx: int = -1
    model_color: str = "white"   # "white" | "black" (engine games only)
    level: int = 0               # engine level (-1 for self-play games)
    selfplay: bool = False       # True = the policy played both colors
    result: str = "*"            # "1-0" | "0-1" | "1/2-1/2"
    termination: str = "unknown"
    model_score: float = 0.0     # 1/0.5/0, model POV (engine games) or White POV (self-play)
    final_material_white: float = 0.0  # material diff (White POV) at game end

    @property
    def group_key(self) -> tuple:
        if self.selfplay:
            return (self.opening_idx, "self", -1)
        return (self.opening_idx, self.model_color, self.level)


def score_for_color(result: str, color: str) -> float:
    """1.0 / 0.5 / 0.0 for "white" / "black" given a game result."""
    if result == "1/2-1/2":
        return 0.5
    white_won = result == "1-0"
    return 1.0 if white_won == (color == "white") else 0.0


def _finish_game(board: chess.Board, model_color: str, selfplay: bool) -> tuple:
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        result, termination = outcome.result(), outcome.termination.name.lower()
    else:
        result, termination = "1/2-1/2", "max_plies"
    if selfplay:
        score = score_for_color(result, "white")
    else:
        score = score_for_color(result, model_color)
    return result, termination, score, material_diff(board, chess.WHITE)


def generate_rollouts(
    model,
    tokenizer,
    *,
    n_games: int,
    levels: Sequence[int],
    group_size: int,
    seed: int,
    temperature: float = 1.0,
    batch_size: int = 1024,
    max_plies: int = MAX_PLIES,
    engine_workers: int = 0,
    selfplay_ratio: float = 0.0,
    device: Optional[str] = None,
    progress: bool = True,
) -> List[RolloutGame]:
    """Play ``n_games`` games (engine games + self-play) and return
    per-game transitions.

    Games are grouped by (opening, model color, level) — or (opening) for
    self-play — with ``group_size`` games per group sharing identical
    starting conditions, so GRPO group baselines compare like with like.
    """
    if not levels:
        raise ValueError("levels must not be empty")
    if not 0.0 <= selfplay_ratio <= 1.0:
        raise ValueError("selfplay_ratio must be in [0, 1]")
    openings = load_openings()
    rng = random.Random(f"rollout-{seed}")

    n_groups = -(-n_games // group_size)
    n_selfplay_groups = round(selfplay_ratio * n_groups)
    group_types = [True] * n_selfplay_groups + [False] * (n_groups - n_selfplay_groups)
    rng.shuffle(group_types)

    combos = [(lvl, color) for lvl in levels for color in ("white", "black")]
    games: List[RolloutGame] = []
    boards: List[chess.Board] = []
    engines: List[Optional[FastEngine]] = []
    combo_k = 0
    for g_idx, is_selfplay in enumerate(group_types):
        size = min(group_size, n_games - len(games))
        if is_selfplay:
            opening_idx = rng.randrange(len(openings))
            game_kw = dict(opening_idx=opening_idx, model_color="white",
                           level=-1, selfplay=True)
            engine = None
        else:
            lvl, color = combos[combo_k % len(combos)]
            combo_k += 1
            opening_idx = rng.randrange(len(openings))
            game_kw = dict(opening_idx=opening_idx, model_color=color,
                           level=lvl, selfplay=False)
            engine = FastEngine(
                level=lvl,
                seed=f"engine-rl-{seed}-{opening_idx}-{lvl}-{color}")
        for _ in range(size):
            games.append(RolloutGame(**game_kw))
            boards.append(chess.Board())
            for uci in openings[opening_idx]:
                boards[-1].push_uci(uci)
            engines.append(engine)

    sampler = PolicySampler(model, tokenizer, batch_size=batch_size, device=device,
                            temperature=temperature, seed=seed)
    pool = None
    if engine_workers > 0:
        pool = get_context("spawn").Pool(engine_workers)
    prog = ProgressLine(min_interval=2.0)

    def engine_seed_of(i: int) -> str:
        g = games[i]
        return f"engine-rl-{seed}-{g.opening_idx}-{g.level}-{g.model_color}"

    try:
        active = list(range(len(games)))
        while active:
            model_idx, engine_idx = [], []
            for i in active:
                b = boards[i]
                if b.outcome(claim_draw=True) is not None or b.ply() >= max_plies:
                    g = games[i]
                    (g.result, g.termination, g.model_score,
                     g.final_material_white) = _finish_game(
                        b, g.model_color, g.selfplay)
                    continue
                if games[i].selfplay or \
                        (b.turn == chess.WHITE) == (games[i].model_color == "white"):
                    model_idx.append(i)
                else:
                    engine_idx.append(i)

            engine_results = None
            if engine_idx:
                if pool is not None:
                    tasks = [(boards[i].fen(), games[i].level, engine_seed_of(i))
                             for i in engine_idx]
                    engine_results = list(pool.imap(engine_best_move_task, tasks))
                else:
                    engine_results = []
                    for i in engine_idx:
                        mv = engines[i].best_move(boards[i])
                        engine_results.append(mv.uci() if mv else None)

            if model_idx:
                choices = sampler.sample(
                    [[m.uci() for m in boards[i].move_stack] for i in model_idx],
                    [boards[i] for i in model_idx],
                )
                for i, choice in zip(model_idx, choices):
                    g, b = games[i], boards[i]
                    mover = "white" if b.turn == chess.WHITE else "black"
                    mat = material_diff(b, chess.WHITE)
                    if choice.legal:
                        b.push(choice.move)
                    else:
                        b.push(sampler.seeded_fallback(b, g.opening_idx))
                    g.transitions.append(Transition(
                        history=[m.uci() for m in b.move_stack[:-1]],
                        token_id=choice.token_id, legal=choice.legal,
                        logprob=choice.logprob, ply=b.ply() - 1,
                        mover=mover, material_white=mat,
                    ))

            if engine_results is not None:
                for i, uci in zip(engine_idx, engine_results):
                    if uci:
                        boards[i].push_uci(uci)

            still = []
            for i in model_idx + engine_idx:
                b = boards[i]
                if b.outcome(claim_draw=True) is not None or b.ply() >= max_plies:
                    g = games[i]
                    (g.result, g.termination, g.model_score,
                     g.final_material_white) = _finish_game(
                        b, g.model_color, g.selfplay)
                else:
                    still.append(i)
            active = still

            done = sum(1 for g in games if g.result != "*")
            prog.show(f"rollouts: {done}/{len(games)} games | "
                      f"{prog.rate(done):.1f}/s | {prog.eta(done, len(games))}")
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        prog.close()

    for g in games:
        if g.result == "*":
            raise RuntimeError("rollout finished with an unfinished game")
    return games
