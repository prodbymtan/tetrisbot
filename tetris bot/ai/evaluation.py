"""
Board evaluation functions for DeepTrix-Z.
Provides heuristic-based evaluation for board states.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from ..core.board import Board
from ..core.pieces import Piece, PieceType


@dataclass
class HeuristicWeights:
    """Weights for the board evaluation heuristics."""
    height: float = -0.51
    lines_cleared: float = 0.76
    holes: float = -0.36
    bumpiness: float = -0.18
    combo: float = 0.25
    t_spin: float = 1.0
    attack: float = 0.5
    garbage: float = -0.2
    survival: float = 0.1
    perfect_clear: float = 10.0
    b2b: float = 0.5


class BoardEvaluator:
    """Base class for board evaluation."""
    
    def __init__(self, weights: HeuristicWeights = None):
        self.weights = weights or HeuristicWeights()
    
    def evaluate_board(self, board: Board) -> float:
        """Evaluate the current state of the board."""
        # Get board features
        heights = board.get_height_map()
        holes = board.get_holes()
        bumpiness = board.get_bumpiness()
        wells = board.get_wells()
        
        # Calculate evaluation score
        score = 0
        score += self.weights.height * sum(heights)
        score += self.weights.holes * holes
        score += self.weights.bumpiness * bumpiness
        
        # Add combo bonus
        score += self.weights.combo * board.combo
        
        # Add attack bonus
        score += self.weights.attack * board.attack
        
        # Add survival bonus
        score += self.weights.survival * (board.BOARD_HEIGHT - max(heights))
        
        # Penalty for garbage lines
        score += self.weights.garbage * board.garbage_lines
        
        return score
    
    def evaluate_placement(self, board: Board, piece: Piece, placement: Piece) -> float:
        """Evaluate a specific piece placement."""
        # Create a temporary board to simulate the placement
        temp_board = Board()
        temp_board.set_board_state(board.get_board_state())
        
        # Place the piece
        if not temp_board.place_piece(placement):
            return float('-inf')  # Invalid placement
        
        # Clear lines
        temp_board.clear_lines()
        
        # Evaluate the resulting board
        return self.evaluate_board(temp_board)


class GarbageAwareEvaluator(BoardEvaluator):
    """Evaluator that is aware of garbage lines and counterplay."""
    
    def __init__(self, weights: HeuristicWeights = None):
        super().__init__(weights)
        self.weights.garbage = -0.5  # Higher penalty for garbage
        self.weights.attack = 0.7  # Higher reward for attack
    
    def evaluate_board(self, board: Board) -> float:
        """Evaluate the board with garbage awareness."""
        base_score = super().evaluate_board(board)
        
        # Add penalty for high garbage
        if board.garbage_lines > 5:
            base_score += self.weights.garbage * board.garbage_lines * 1.5
        
        # Add bonus for potential garbage clearing
        # (Simplified: reward for low board height)
        heights = board.get_height_map()
        if heights and max(heights) < board.BOARD_HEIGHT / 2:
            base_score += 10.0
        
        return base_score


class StyleAwareEvaluator(BoardEvaluator):
    """Evaluator that adapts to specific player styles."""
    
    def __init__(self, target_style: str, weights: HeuristicWeights = None):
        super().__init__(weights)
        self.target_style = target_style
        self._load_style_weights()
    
    def _load_style_weights(self):
        """Load weights for a specific player style."""
        if self.target_style == 'diao':
            # Diao's style: T-spins and aggressive play
            self.weights.t_spin = 2.0
            self.weights.attack = 1.0
            self.weights.height = -0.4
        elif self.target_style == 'vincehd':
            # VinceHD's style: efficient downstacking and combos
            self.weights.combo = 0.5
            self.weights.garbage = -0.6
            self.weights.height = -0.6
        else:
            # Default aggressive style
            self.weights.attack = 0.8
            self.weights.t_spin = 1.5


class RewardShaper:
    """Shapes rewards for reinforcement learning based on game events."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'line_clear': 0.5,
            't_spin_mini': 1.0,
            't_spin_double': 3.0,
            't_spin_triple': 5.0,
            'b2b': 1.0,
            'perfect_clear': 10.0,
            'combo': 0.5,
            'garbage_sent': 1.0,
            'garbage_tanked': -0.5,
            'survival': 0.1,
            'top_out': -10.0
        }
    
    def shape_reward(self, prev_state: Board, current_state: Board) -> float:
        """Calculate the reward based on the change in game state."""
        reward = 0
        
        # Line clears
        lines_cleared = current_state.lines_cleared - prev_state.lines_cleared
        if lines_cleared > 0:
            reward += self.config['line_clear'] * lines_cleared
        
        # Combos
        if current_state.combo > prev_state.combo:
            reward += self.config['combo'] * current_state.combo
        
        # B2B
        # (Requires more state tracking - simplified here)
        
        # T-spins
        # (Requires T-spin detection - simplified here)
        
        # Perfect clear
        if lines_cleared > 0 and np.sum(current_state.board) == 0:
            reward += self.config['perfect_clear']
        
        # Garbage sent
        garbage_sent = current_state.attack - prev_state.attack
        if garbage_sent > 0:
            reward += self.config['garbage_sent'] * garbage_sent
        
        # Garbage received
        garbage_received = current_state.garbage_lines - prev_state.garbage_lines
        if garbage_received > 0:
            reward += self.config['garbage_tanked'] * garbage_received
        
        # Survival
        reward += self.config['survival']
        
        # Top out
        if current_state.is_game_over():
            reward += self.config['top_out']
        
        return reward


class HeuristicPlayouter:
    """Uses heuristics to simulate game playouts."""
    
    def __init__(self, evaluator: BoardEvaluator):
        self.evaluator = evaluator
    
    def playout(self, board: Board, piece_queue: List[Piece], max_depth: int = 5) -> float:
        """Simulate a playout and return the final board score."""
        if max_depth == 0 or board.is_game_over():
            return self.evaluator.evaluate_board(board)
        
        # Get best placement for the current piece
        current_piece = piece_queue[0]
        best_placement = self._get_best_placement(board, current_piece)
        
        if not best_placement:
            return self.evaluator.evaluate_board(board)
        
        # Create new board state
        new_board = Board()
        new_board.set_board_state(board.get_board_state())
        
        # Place the piece
        new_board.place_piece(best_placement)
        new_board.clear_lines()
        
        # Continue playout with next piece
        return self.playout(new_board, piece_queue[1:], max_depth - 1)
    
    def _get_best_placement(self, board: Board, piece: Piece) -> Optional[Piece]:
        """Get the best placement for a piece."""
        valid_placements = self._get_all_valid_placements(board, piece)
        
        if not valid_placements:
            return None
        
        best_placement = None
        best_score = float('-inf')
        
        for placement in valid_placements:
            score = self.evaluator.evaluate_placement(board, piece, placement)
            if score > best_score:
                best_score = score
                best_placement = placement
        
        return best_placement
    
    def _get_all_valid_placements(self, board: Board, piece: Piece) -> List[Piece]:
        """Get all valid placements for a piece."""
        valid = []
        # (This is a simplified implementation - a full one would be more complex)
        for rot in range(4):
            for x in range(-2, board.BOARD_WIDTH + 2):
                test_piece = Piece(piece.piece_type, (x, 0, rot))
                drop_pos = board.get_drop_position(test_piece)
                if drop_pos:
                    placed_piece = Piece(piece.piece_type, drop_pos)
                    if board.is_valid_position(placed_piece):
                        valid.append(placed_piece)
        return valid


class EvaluationManager:
    """Manages different evaluation strategies."""
    
    def __init__(self, mode: str = 'default', style: Optional[str] = None):
        self.mode = mode
        self.style = style
        self.evaluator = self._create_evaluator()
    
    def _create_evaluator(self) -> BoardEvaluator:
        """Create an evaluator based on the current mode."""
        if self.mode == 'garbage_aware':
            return GarbageAwareEvaluator()
        elif self.mode == 'style_aware' and self.style:
            return StyleAwareEvaluator(self.style)
        else:
            return BoardEvaluator()
    
    def evaluate(self, board: Board) -> float:
        """Evaluate a board."""
        return self.evaluator.evaluate_board(board)
    
    def get_best_move(self, board: Board, piece: Piece) -> Optional[Piece]:
        """Get the best move for a piece."""
        playouter = HeuristicPlayouter(self.evaluator)
        return playouter._get_best_placement(board, piece)
    
    def set_mode(self, mode: str, style: Optional[str] = None):
        """Set the evaluation mode."""
        self.mode = mode
        self.style = style
        self.evaluator = self._create_evaluator() 