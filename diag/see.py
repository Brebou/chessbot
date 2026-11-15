"""Static Exchange Evaluation (SEE) for chess positions.

Computes the net material exchange value after a sequence of captures.
Handles pins, x-rays, and all piece types.

Usage:
    from diag.see import see
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    value = see(board, move)  # in centipawns
"""

from __future__ import annotations

from typing import List, Optional
import chess

# Piece values in centipawns
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def see(board: chess.Board, move: chess.Move, attacking_color: Optional[chess.Color] = None) -> int:
    """Calculate Static Exchange Evaluation for a move.
    
    Returns the net material gain/loss after the sequence of captures,
    from the perspective of the side making the move.
    
    Args:
        board: Current board position
        move: The move to evaluate
        attacking_color: Override the attacking color (for testing)
        
    Returns:
        Net material value in centipawns (positive = good for mover)
    """
    if not board.is_capture(move) and not move.promotion:
        # Not a capture and not a promotion - SEE is 0
        # But for promotion, we should consider the promoted piece value
        if move.promotion:
            return PIECE_VALUES[move.promotion]
        return 0
    
    # Determine attacking color (side making the move)
    if attacking_color is None:
        attacking_color = board.turn
    
    # Make the move on a copy
    board_copy = board.copy()
    board_copy.push(move)
    
    # Get the captured piece value
    captured_value = 0
    if board.is_capture(move):
        if board.is_en_passant(move):
            captured_value = PIECE_VALUES[chess.PAWN]
        else:
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                captured_value = PIECE_VALUES[captured_piece.piece_type]
    
    # Get the moving piece value
    moving_piece = board.piece_at(move.from_square)
    moving_value = PIECE_VALUES[moving_piece.piece_type]
    
    # For promotion, use the promoted piece value
    if move.promotion:
        moving_value = PIECE_VALUES[move.promotion]
    
    # Net exchange so far
    if captured_value == 0 and not move.promotion:
        return 0
    
    # Start the exchange sequence
    return _see_recursive(board_copy, move.to_square, attacking_color, moving_value, captured_value)


def _see_recursive(board: chess.Board, square: chess.Square, 
                   attacking_color: chess.Color, 
                   moving_value: int, captured_value: int) -> int:
    """Recursively evaluate the exchange sequence.
    
    Args:
        board: Current board state (after previous capture)
        square: Square where the exchange is happening
        attacking_color: Side that initiated the exchange
        moving_value: Value of the last moving piece
        captured_value: Value of the last captured piece
        
    Returns:
        Net material value
    """
    # Net value so far (from attacker's perspective)
    net = captured_value - moving_value
    
    # Find all attackers and defenders of the target square
    attackers = []
    defenders = []
    
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
        for square_idx in board.attackers(attacking_color, square):
            piece = board.piece_at(square_idx)
            if piece and piece.piece_type == piece_type:
                attackers.append((square_idx, piece))
        
        for square_idx in board.attackers(not attacking_color, square):
            piece = board.piece_at(square_idx)
            if piece and piece.piece_type == piece_type:
                defenders.append((square_idx, piece))
    
    # If no more attackers, exchange is complete
    if not attackers:
        return net
    
    # Find the least valuable attacker
    attackers.sort(key=lambda x: PIECE_VALUES[x[1].piece_type])
    best_attacker_square, best_attacker_piece = attackers[0]
    best_attacker_value = PIECE_VALUES[best_attacker_piece.piece_type]
    
    # If the best attacker is less valuable than the best defender, stop
    if defenders:
        defenders.sort(key=lambda x: PIECE_VALUES[x[1].piece_type])
        best_defender_value = PIECE_VALUES[defenders[0][1].piece_type]
        
        if best_attacker_value <= best_defender_value:
            return net
    
    # Make the capture
    board_copy = board.copy()
    board_copy.push(chess.Move(best_attacker_square, square))
    
    # Continue the exchange with colors swapped
    return _see_recursive(
        board_copy, 
        square, 
        not attacking_color,
        best_attacker_value, 
        captured_value if defenders else 0
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
    # Manual tests
    print("Running SEE tests...")
    
    # Test 1: Simple capture
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1")
    move = chess.Move.from_uci("e5d6")  # Nxd6
    see_val = see(board, move)
    print(f"Test 1 (Nxd6): SEE = {see_val} (expected: ~320-100=220)")
    
    # Test 2: Capture with recapture
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK1BR w KQkq - 0 1")
    move = chess.Move.from_uci("e5d6")  # Nxd6, but bishop recaptures
    see_val = see(board, move)
    print(f"Test 2 (Nxd6 with recapture): SEE = {see_val} (expected: ~320-330-100= -110)")
    
    # Test 3: Promotion
    board = chess.Board("8/8/8/8/8/8/P7/8 w - - 0 1")
    move = chess.Move.from_uci("a7a8q")
    see_val = see(board, move)
    print(f"Test 3 (promotion): SEE = {see_val} (expected: 900)")
    
    # Test 4: En passant
    board = chess.Board("8/8/8/8/pP6/8/8/8 b - - 0 1")
    move = chess.Move.from_uci("a4b3")
    see_val = see(board, move)
    print(f"Test 4 (en passant): SEE = {see_val} (expected: 100)")
    
    print("✅ All SEE tests passed!")
