"""Graphical interface for testing the ChessGPT model.

This provides a visual way to:
- Manipulate the board and play moves
- See the model's predictions with probabilities
- Get evaluation from the built-in engine
- Auto-play to a position using the engine
- Have the model play against itself

Usage:
    uv run diag/gui.py --model base_model --device cpu

Note: Stockfish is optional. If installed, it will be used for stronger evaluation.
The built-in engine is always available.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import chess
import chess.svg
import gradio as gr
import numpy as np

# Add parent directory to path to import chess_hw
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_hw import load_model, ChessTokenizer, Engine
from chess_hw.play import ModelPlayer, MoveChoice

from .common import ProbePlayer, replay, material_diff, material_value, phase


# Try to import stockfish if available
try:
    import stockfish
    STOCKFISH_AVAILABLE = True
except ImportError:
    STOCKFISH_AVAILABLE = False


class ChessGUI:
    """Main class for the chess testing interface."""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """Initialize the GUI.
        
        Args:
            model_path: Path to the model directory or .safetensors file
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_path = Path(model_path)
        self.device = device
        
        # Load model and tokenizer
        self.model = load_model(str(model_path), device=device)
        self.tokenizer = ChessTokenizer()
        
        # Create players
        self.model_player = ModelPlayer(self.model, name="model")
        self.probe_player = ProbePlayer(self.model, self.tokenizer, device=device)
        
        # Built-in engine for evaluation
        self.engine = Engine(level=3, seed=42)
        
        # Optional Stockfish
        self.stockfish = None
        if STOCKFISH_AVAILABLE:
            try:
                self.stockfish = stockfish.Stockfish()
            except:
                pass
        
        # State
        self.board = chess.Board()
        self.history = []
        self.game_state = {"moves": [], "board": self.board.fen()}
        
    def get_board_svg(self) -> str:
        """Get SVG representation of the current board."""
        return chess.svg.board(board=self.board, size=400)
    
    def get_fen(self) -> str:
        """Get FEN of current position."""
        return self.board.fen()
    
    def get_pgn(self) -> str:
        """Get PGN of current game."""
        if not self.history:
            return ""
        game = chess.Game()
        game.setup(self.board)
        return str(game.game())
    
    def get_move_history(self) -> str:
        """Get formatted move history."""
        if not self.history:
            return "No moves yet"
        
        moves = []
        for i, move in enumerate(self.history):
            if i % 2 == 0:
                moves.append(f"{i//2 + 1}. {move}")
            else:
                moves[-1] += f" {move}"
        return "\n".join(moves)
    
    def play_move(self, uci: str) -> Tuple[str, str, str, dict]:
        """Play a move on the board.
        
        Args:
            uci: UCI move string (e.g., 'e2e4')
            
        Returns:
            Tuple of (board_svg, fen, move_history, game_state)
        """
        try:
            self.board.push_uci(uci)
            self.history.append(uci)
            self.game_state = {
                "moves": self.history.copy(),
                "board": self.board.fen(),
                "ply": self.board.ply(),
                "turn": "white" if self.board.turn == chess.WHITE else "black"
            }
            return (
                self.get_board_svg(),
                self.get_fen(),
                self.get_move_history(),
                self.game_state
            )
        except ValueError as e:
            return (
                self.get_board_svg(),
                self.get_fen(),
                f"Error: {e}",
                self.game_state
            )
    
    def undo_move(self) -> Tuple[str, str, str, dict]:
        """Undo the last move.
        
        Returns:
            Tuple of (board_svg, fen, move_history, game_state)
        """
        if self.board.move_stack:
            self.board.pop()
            if self.history:
                self.history.pop()
        
        self.game_state = {
            "moves": self.history.copy(),
            "board": self.board.fen(),
            "ply": self.board.ply(),
            "turn": "white" if self.board.turn == chess.WHITE else "black"
        }
        
        return (
            self.get_board_svg(),
            self.get_fen(),
            self.get_move_history(),
            self.game_state
        )
    
    def reset_board(self) -> Tuple[str, str, str, dict]:
        """Reset to starting position.
        
        Returns:
            Tuple of (board_svg, fen, move_history, game_state)
        """
        self.board = chess.Board()
        self.history = []
        self.game_state = {
            "moves": [],
            "board": self.board.fen(),
            "ply": 0,
            "turn": "white"
        }
        
        return (
            self.get_board_svg(),
            self.get_fen(),
            "",
            self.game_state
        )
    
    def get_model_prediction(self) -> Tuple[dict, str]:
        """Get model's move predictions with probabilities.
        
        Returns:
            Tuple of (predictions_dict, summary_string)
        """
        if not self.board.move_stack:
            history = []
        else:
            history = [m.uci() for m in self.board.move_stack]
        
        # Get top-10 predictions
        tokens, probs = self.probe_player.get_topk(history, k=10)
        
        # Get legal moves
        legal_moves = {m.uci(): m for m in self.board.legal_moves}
        
        # Build predictions dict
        predictions = []
        for token, prob in zip(tokens, probs):
            is_legal = token in legal_moves
            predictions.append({
                "move": token,
                "probability": f"{prob*100:.2f}%",
                "legal": is_legal,
                "rank": len(predictions) + 1
            })
        
        # Summary
        legal_count = sum(1 for p in predictions if p["legal"])
        summary = f"Model predictions (top 10)\nTotal legal in top-10: {legal_count}/10\nTotal legal moves available: {len(legal_moves)}"
        
        return predictions, summary
    
    def get_model_move(self) -> Tuple[str, str, str, dict, dict, str]:
        """Get the model's chosen move with full analysis.
        
        Returns:
            Tuple of (board_svg, fen, move_history, game_state, predictions, summary)
        """
        # Get predictions
        predictions, _ = self.get_model_prediction()
        
        # Get model's actual choice
        choice = self.model_player.choose(self.board)
        
        if choice.move:
            # Play the move
            uci = choice.move.uci()
            self.board.push(choice.move)
            self.history.append(uci)
            
            summary = f"Model chose: {uci} (attempt {choice.attempt})"
        else:
            # Fallback to random
            legal = list(self.board.legal_moves)
            if legal:
                move = random.choice(legal)
                uci = move.uci()
                self.board.push(move)
                self.history.append(uci)
                summary = f"All top-3 illegal! Random move: {uci}"
            else:
                summary = "No legal moves available"
        
        self.game_state = {
            "moves": self.history.copy(),
            "board": self.board.fen(),
            "ply": self.board.ply(),
            "turn": "white" if self.board.turn == chess.WHITE else "black"
        }
        
        return (
            self.get_board_svg(),
            self.get_fen(),
            self.get_move_history(),
            self.game_state,
            predictions,
            summary
        )
    
    def get_engine_evaluation(self, depth: int = 3) -> Tuple[str, str, float]:
        """Get evaluation from the built-in engine.
        
        Args:
            depth: Search depth (1-4)
            
        Returns:
            Tuple of (best_move, evaluation_string, score)
        """
        engine = Engine(level=depth, seed=42)
        best_move = engine.best_move(self.board)
        
        if best_move:
            # Make the move to get evaluation
            self.board.push(best_move)
            score = engine.evaluate(self.board)
            self.board.pop()
            
            # Score is from white's perspective
            if self.board.turn == chess.BLACK:
                score = -score
            
            # Convert to pawns
            score_pawns = score / 100.0
            
            eval_str = f"{score_pawns:+.2f}"
            return best_move.uci(), eval_str, score_pawns
        else:
            return "", "0.00", 0.0
    
    def get_stockfish_evaluation(self, depth: int = 18) -> Tuple[str, str, float]:
        """Get evaluation from Stockfish if available.
        
        Args:
            depth: Search depth
            
        Returns:
            Tuple of (best_move, evaluation_string, score)
        """
        if not self.stockfish:
            return "Stockfish not available", "N/A", 0.0
        
        self.stockfish.set_fen_position(self.board.fen())
        self.stockfish.set_depth(depth)
        
        info = self.stockfish.get_evaluation()
        
        if info["type"] == "mate":
            mate_in = info["value"]
            eval_str = f"#{-mate_in}" if mate_in < 0 else f"#{mate_in}"
            score = float('inf') if mate_in > 0 else float('-inf')
        else:
            score = info["value"] / 100.0  # Convert to pawns
            eval_str = f"{score:+.2f}"
        
        best_move = self.stockfish.get_best_move()
        
        return best_move, eval_str, score
    
    def auto_play_to_ply(self, target_ply: int, engine_depth: int = 2) -> Tuple[str, str, str, dict]:
        """Auto-play moves using the engine until reaching target ply.
        
        Args:
            target_ply: Target ply number
            engine_depth: Engine search depth
            
        Returns:
            Tuple of (board_svg, fen, move_history, game_state)
        """
        current_ply = self.board.ply()
        
        if current_ply >= target_ply:
            return (
                self.get_board_svg(),
                self.get_fen(),
                self.get_move_history(),
                self.game_state
            )
        
        engine = Engine(level=engine_depth, seed=42)
        
        for _ in range(target_ply - current_ply):
            best_move = engine.best_move(self.board)
            if best_move:
                self.board.push(best_move)
                self.history.append(best_move.uci())
            else:
                break
        
        self.game_state = {
            "moves": self.history.copy(),
            "board": self.board.fen(),
            "ply": self.board.ply(),
            "turn": "white" if self.board.turn == chess.WHITE else "black"
        }
        
        return (
            self.get_board_svg(),
            self.get_fen(),
            self.get_move_history(),
            self.game_state
        )
    
    def model_self_play(self, max_moves: int = 10) -> Tuple[str, str, str, dict, str]:
        """Have the model play against itself.
        
        Args:
            max_moves: Maximum number of moves to play
            
        Returns:
            Tuple of (board_svg, fen, move_history, game_state, result)
        """
        result = ""
        
        for _ in range(max_moves):
            if self.board.is_game_over():
                outcome = self.board.outcome(claim_draw=True)
                if outcome:
                    result = outcome.result()
                break
            
            choice = self.model_player.choose(self.board)
            
            if choice.move:
                self.board.push(choice.move)
                self.history.append(choice.move.uci())
            else:
                # Fallback
                legal = list(self.board.legal_moves)
                if legal:
                    move = random.choice(legal)
                    self.board.push(move)
                    self.history.append(move.uci())
        
        self.game_state = {
            "moves": self.history.copy(),
            "board": self.board.fen(),
            "ply": self.board.ply(),
            "turn": "white" if self.board.turn == chess.WHITE else "black"
        }
        
        return (
            self.get_board_svg(),
            self.get_fen(),
            self.get_move_history(),
            self.game_state,
            result
        )
    
    def get_legal_moves(self) -> list:
        """Get list of legal moves in the current position."""
        return [m.uci() for m in self.board.legal_moves]
    
    def get_game_info(self) -> dict:
        """Get comprehensive game information."""
        return {
            "fen": self.board.fen(),
            "ply": self.board.ply(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "is_check": self.board.is_check(),
            "is_checkmate": self.board.is_checkmate(),
            "is_stalemate": self.board.is_stalemate(),
            "is_insufficient_material": self.board.is_insufficient_material(),
            "can_claim_draw": self.board.can_claim_draw(),
            "halfmove_clock": self.board.halfmove_clock,
            "fullmove_number": self.board.fullmove_number,
            "legal_moves_count": len(list(self.board.legal_moves)),
            "material_white": material_value(self.board, chess.WHITE),
            "material_black": material_value(self.board, chess.BLACK),
            "material_diff": material_diff(self.board, chess.WHITE),
            "phase": phase(self.board.ply())
        }


def create_interface(model_path: str, device: str = "cpu"):
    """Create the Gradio interface.
    
    Args:
        model_path: Path to the model
        device: Device to run on
        
    Returns:
        Gradio Blocks interface
    """
    gui = ChessGUI(model_path, device)
    
    with gr.Blocks(title="ChessGPT Tester") as demo:
        gr.Markdown("# ChessGPT Model Tester")
        gr.Markdown("Test and visualize the ChessGPT model's predictions and gameplay.")
        
        with gr.Tabs():
            # Tab 1: Manual Play
            with gr.Tab("Manual Play"):
                gr.Markdown("### Manual Board Manipulation")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        board_display = gr.HTML(gui.get_board_svg(), label="Chess Board")
                    with gr.Column(scale=1):
                        fen_display = gr.Textbox(gui.get_fen(), label="FEN")
                        move_history = gr.Textbox(gui.get_move_history(), label="Move History")
                        game_info = gr.JSON(gui.get_game_info(), label="Game Info")
                
                with gr.Row():
                    uci_input = gr.Textbox(label="Enter UCI Move (e.g., e2e4)", placeholder="e2e4")
                    play_btn = gr.Button("Play Move", variant="primary")
                    undo_btn = gr.Button("Undo", variant="secondary")
                    reset_btn = gr.Button("Reset", variant="stop")
                
                play_btn.click(
                    gui.play_move,
                    inputs=[uci_input],
                    outputs=[board_display, fen_display, move_history, game_info]
                )
                undo_btn.click(
                    gui.undo_move,
                    outputs=[board_display, fen_display, move_history, game_info]
                )
                reset_btn.click(
                    gui.reset_board,
                    outputs=[board_display, fen_display, move_history, game_info]
                )
                
                # Model prediction
                gr.Markdown("### Model Predictions")
                with gr.Row():
                    predict_btn = gr.Button("Get Model Predictions", variant="primary")
                
                predictions_table = gr.Dataframe(
                    headers=["Rank", "Move", "Probability", "Legal"],
                    label="Top 10 Predictions"
                )
                prediction_summary = gr.Textbox(label="Summary")
                
                def get_preds():
                    preds, summary = gui.get_model_prediction()
                    # Convert to dataframe format
                    rows = [[p["rank"], p["move"], p["probability"], p["legal"]] for p in preds]
                    return rows, summary
                
                predict_btn.click(
                    get_preds,
                    outputs=[predictions_table, prediction_summary]
                )
                
                # Model move
                gr.Markdown("### Model's Move")
                with gr.Row():
                    model_move_btn = gr.Button("Play Model's Move", variant="primary")
                
                model_summary = gr.Textbox(label="Model Move Summary")
                
                def play_model_move():
                    board_svg, fen, hist, state, preds, summary = gui.get_model_move()
                    rows = [[p["rank"], p["move"], p["probability"], p["legal"]] for p in preds]
                    return board_svg, fen, hist, state, rows, summary
                
                model_move_btn.click(
                    play_model_move,
                    outputs=[board_display, fen_display, move_history, game_info, predictions_table, model_summary]
                )
            
            # Tab 2: Evaluation
            with gr.Tab("Evaluation"):
                gr.Markdown("### Position Evaluation")
                
                with gr.Row():
                    eval_engine_btn = gr.Button("Evaluate (Built-in Engine)")
                    if STOCKFISH_AVAILABLE:
                        eval_sf_btn = gr.Button("Evaluate (Stockfish)")
                
                with gr.Row():
                    eval_move = gr.Textbox(label="Best Move")
                    eval_score = gr.Textbox(label="Evaluation")
                    eval_bar = gr.Slider(label="Score (pawns)", minimum=-10, maximum=10, value=0)
                
                def do_engine_eval(depth=3):
                    move, eval_str, score = gui.get_engine_evaluation(depth)
                    return move, eval_str, score
                
                def do_sf_eval(depth=18):
                    move, eval_str, score = gui.get_stockfish_evaluation(depth)
                    return move, eval_str, score
                
                eval_engine_btn.click(
                    do_engine_eval,
                    outputs=[eval_move, eval_score, eval_bar]
                )
                
                if STOCKFISH_AVAILABLE:
                    eval_sf_btn.click(
                        do_sf_eval,
                        outputs=[eval_move, eval_score, eval_bar]
                    )
                else:
                    gr.Markdown("*Stockfish not available - install with `pip install stockfish`*")
            
            # Tab 3: Auto-Play
            with gr.Tab("Auto-Play"):
                gr.Markdown("### Auto-Play to Position")
                
                with gr.Row():
                    target_ply = gr.Slider(label="Target Ply", minimum=0, maximum=50, value=10, step=1)
                    engine_depth = gr.Slider(label="Engine Depth", minimum=1, maximum=4, value=2, step=1)
                    auto_play_btn = gr.Button("Auto-Play", variant="primary")
                
                auto_play_btn.click(
                    gui.auto_play_to_ply,
                    inputs=[target_ply, engine_depth],
                    outputs=[board_display, fen_display, move_history, game_info]
                )
                
                # Self-play
                gr.Markdown("### Model Self-Play")
                
                with gr.Row():
                    max_moves = gr.Slider(label="Max Moves", minimum=1, maximum=50, value=10, step=1)
                    self_play_btn = gr.Button("Model vs Model", variant="primary")
                
                self_play_result = gr.Textbox(label="Result")
                
                def do_self_play(max_moves):
                    board_svg, fen, hist, state, result = gui.model_self_play(max_moves)
                    return board_svg, fen, hist, state, result
                
                self_play_btn.click(
                    do_self_play,
                    inputs=[max_moves],
                    outputs=[board_display, fen_display, move_history, game_info, self_play_result]
                )
            
            # Tab 4: Analysis
            with gr.Tab("Analysis"):
                gr.Markdown("### Position Analysis")
                
                with gr.Row():
                    analyze_btn = gr.Button("Analyze Position", variant="primary")
                
                analysis_output = gr.JSON(label="Full Analysis")
                
                analyze_btn.click(
                    gui.get_game_info,
                    outputs=[analysis_output]
                )
                
                gr.Markdown("### Legal Moves")
                legal_moves_display = gr.Textbox(label="Legal Moves")
                
                def get_legal():
                    return ", ".join(gui.get_legal_moves())
                
                analyze_btn.click(
                    get_legal,
                    outputs=[legal_moves_display]
                )
        
        # Initial state
        demo.load(
            lambda: (
                gui.get_board_svg(),
                gui.get_fen(),
                gui.get_move_history(),
                gui.get_game_info()
            ),
            outputs=[board_display, fen_display, move_history, game_info]
        )
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="ChessGPT Testing GUI")
    parser.add_argument("--model", type=str, default="base_model", help="Path to model directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("--share", action="store_true", help="Share the Gradio app publicly")
    args = parser.parse_args()
    
    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        # Try relative to chess-post-training
        base_dir = Path(__file__).resolve().parent.parent
        model_path = base_dir / args.model
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {args.model}")
    
    demo = create_interface(str(model_path), args.device)
    demo.launch(share=args.share, server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
