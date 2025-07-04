"""
Finesse engine for DeepTrix-Z.
Provides frame-perfect input optimization and optimal piece placement paths.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
from dataclasses import dataclass
from .pieces import Piece, PieceType, Position
from .board import Board


@dataclass
class Input:
    """Represents a single input action."""
    action: str  # 'left', 'right', 'rotate_cw', 'rotate_ccw', 'soft_drop', 'hard_drop', 'hold'
    frame: int   # Frame when this input should be executed
    duration: int = 1  # How long to hold the input (in frames)


@dataclass
class FinessePath:
    """Represents a complete finesse path for placing a piece."""
    inputs: List[Input]
    final_position: Position
    total_frames: int
    is_optimal: bool


class FinesseEngine:
    """Engine for calculating optimal finesse paths for piece placement."""
    
    # Optimal finesse paths for each piece type and target position
    # This is a simplified version - a full implementation would have complete finesse tables
    FIENNSE_TABLES = {
        PieceType.I: {
            # Simplified I-piece finesse - full implementation would be much more complex
            (0, 0): ['rotate_cw', 'left', 'left', 'left', 'hard_drop'],
            (1, 0): ['rotate_cw', 'left', 'left', 'hard_drop'],
            (2, 0): ['rotate_cw', 'left', 'hard_drop'],
            (3, 0): ['rotate_cw', 'hard_drop'],
            (4, 0): ['hard_drop'],
            (5, 0): ['right', 'hard_drop'],
            (6, 0): ['right', 'right', 'hard_drop'],
            (7, 0): ['right', 'right', 'right', 'hard_drop'],
            (8, 0): ['right', 'right', 'right', 'right', 'hard_drop'],
            (9, 0): ['right', 'right', 'right', 'right', 'right', 'hard_drop'],
        },
        PieceType.O: {
            # O-piece finesse (simplified)
            (0, 0): ['left', 'left', 'left', 'left', 'hard_drop'],
            (1, 0): ['left', 'left', 'left', 'hard_drop'],
            (2, 0): ['left', 'left', 'hard_drop'],
            (3, 0): ['left', 'hard_drop'],
            (4, 0): ['hard_drop'],
            (5, 0): ['right', 'hard_drop'],
            (6, 0): ['right', 'right', 'hard_drop'],
            (7, 0): ['right', 'right', 'right', 'hard_drop'],
            (8, 0): ['right', 'right', 'right', 'right', 'hard_drop'],
            (9, 0): ['right', 'right', 'right', 'right', 'right', 'hard_drop'],
        }
    }
    
    # DAS (Delayed Auto Shift) and ARR (Auto Repeat Rate) settings
    DAS_DELAY = 8  # Frames before auto-repeat starts
    ARR_RATE = 1   # Frames between auto-repeat
    
    def __init__(self):
        self.current_frame = 0
        self.last_input_frame = 0
    
    def get_finesse_path(self, board: Board, piece: Piece, target_position: Position) -> Optional[FinessePath]:
        """Get the optimal finesse path to place a piece at the target position."""
        if not board.is_valid_position(Piece(piece.piece_type, target_position)):
            return None
        
        # Get the optimal input sequence
        input_sequence = self._get_optimal_inputs(piece, target_position)
        if not input_sequence:
            return None
        
        # Convert to Input objects with proper timing
        inputs = self._convert_to_timed_inputs(input_sequence)
        
        return FinessePath(
            inputs=inputs,
            final_position=target_position,
            total_frames=len(inputs),
            is_optimal=True
        )
    
    def _get_optimal_inputs(self, piece: Piece, target_position: Position) -> List[str]:
        """Get the optimal input sequence for a piece placement."""
        # This is a simplified implementation
        # A full implementation would use comprehensive finesse tables
        
        current_x = piece.position.x
        target_x = target_position.x
        current_rotation = piece.position.rotation
        target_rotation = target_position.rotation
        
        inputs = []
        
        # Handle rotation first
        rotation_diff = (target_rotation - current_rotation) % 4
        if rotation_diff == 1:
            inputs.append('rotate_cw')
        elif rotation_diff == 3:
            inputs.append('rotate_ccw')
        elif rotation_diff == 2:
            # Choose the shorter rotation direction
            inputs.append('rotate_cw')
            inputs.append('rotate_cw')
        
        # Handle horizontal movement
        x_diff = target_x - current_x
        if x_diff < 0:
            # Move left
            for _ in range(abs(x_diff)):
                inputs.append('left')
        elif x_diff > 0:
            # Move right
            for _ in range(x_diff):
                inputs.append('right')
        
        # Always end with hard drop
        inputs.append('hard_drop')
        
        return inputs
    
    def _convert_to_timed_inputs(self, input_sequence: List[str]) -> List[Input]:
        """Convert input sequence to timed Input objects."""
        inputs = []
        current_frame = 0
        
        for action in input_sequence:
            if action in ['left', 'right']:
                # Handle DAS/ARR for horizontal movement
                if current_frame == 0:
                    # First press
                    inputs.append(Input(action, current_frame, self.DAS_DELAY))
                    current_frame += self.DAS_DELAY
                else:
                    # Auto-repeat
                    inputs.append(Input(action, current_frame, self.ARR_RATE))
                    current_frame += self.ARR_RATE
            else:
                # Other actions (rotation, drop) are instant
                inputs.append(Input(action, current_frame, 1))
                current_frame += 1
        
        return inputs
    
    def get_all_valid_placements(self, board: Board, piece: Piece) -> List[Position]:
        """Get all valid placement positions for a piece."""
        valid_placements = []
        
        # Try all rotations
        for rotation in range(4):
            rotated_piece = Piece(piece.piece_type, Position(
                piece.position.x, piece.position.y, rotation
            ))
            
            # Try all horizontal positions
            for x in range(-2, board.BOARD_WIDTH + 2):
                test_piece = Piece(piece.piece_type, Position(x, 0, rotation))
                
                # Get drop position
                drop_pos = board.get_drop_position(test_piece)
                if drop_pos and board.is_valid_position(Piece(piece.piece_type, drop_pos)):
                    valid_placements.append(drop_pos)
        
        return valid_placements
    
    def get_best_placement(self, board: Board, piece: Piece, evaluator) -> Optional[Position]:
        """Get the best placement position for a piece using an evaluator function."""
        valid_placements = self.get_all_valid_placements(board, piece)
        
        if not valid_placements:
            return None
        
        best_position = None
        best_score = float('-inf')
        
        for position in valid_placements:
            # Create a temporary board to evaluate the placement
            temp_board = Board()
            temp_board.set_board_state(board.get_board_state())
            
            # Place the piece
            placed_piece = Piece(piece.piece_type, position)
            if temp_board.place_piece(placed_piece):
                # Clear lines
                temp_board.clear_lines()
                
                # Evaluate the resulting board
                score = evaluator(temp_board)
                
                if score > best_score:
                    best_score = score
                    best_position = position
        
        return best_position
    
    def execute_finesse_path(self, board: Board, path: FinessePath) -> bool:
        """Execute a finesse path on the board."""
        current_piece = board.current_piece
        if not current_piece:
            return False
        
        # Apply the final position directly
        final_piece = Piece(current_piece.piece_type, path.final_position)
        
        # Place the piece
        if board.place_piece(final_piece):
            # Clear lines
            board.clear_lines()
            return True
        
        return False
    
    def get_finesse_error(self, board: Board, piece: Piece, target_position: Position) -> Optional[str]:
        """Check if a piece placement would be a finesse error."""
        optimal_path = self.get_finesse_path(board, piece, target_position)
        if not optimal_path:
            return "Invalid placement"
        
        # Count the number of inputs needed
        input_count = len(optimal_path.inputs)
        
        # Check if this is more inputs than necessary
        # This is a simplified check - a full implementation would be more sophisticated
        if input_count > 6:  # Arbitrary threshold
            return f"Too many inputs ({input_count})"
        
        return None
    
    def optimize_inputs(self, inputs: List[Input]) -> List[Input]:
        """Optimize a sequence of inputs to minimize frame count."""
        optimized = []
        current_frame = 0
        
        for input_action in inputs:
            # Adjust frame timing for optimal execution
            optimized_input = Input(
                action=input_action.action,
                frame=current_frame,
                duration=input_action.duration
            )
            optimized.append(optimized_input)
            current_frame += input_action.duration
        
        return optimized
    
    def get_input_delay(self) -> Dict[str, int]:
        """Get the current input delay settings."""
        return {
            'DAS_DELAY': self.DAS_DELAY,
            'ARR_RATE': self.ARR_RATE
        }
    
    def set_input_delay(self, das_delay: int, arr_rate: int):
        """Set the input delay settings."""
        self.DAS_DELAY = das_delay
        self.ARR_RATE = arr_rate
    
    def reset_frame_counter(self):
        """Reset the frame counter."""
        self.current_frame = 0
        self.last_input_frame = 0
    
    def advance_frame(self):
        """Advance the frame counter."""
        self.current_frame += 1


class FinesseTrainer:
    """Trainer for improving finesse accuracy."""
    
    def __init__(self):
        self.error_counts = {}
        self.total_attempts = 0
    
    def record_attempt(self, piece_type: PieceType, target_position: Position, success: bool):
        """Record a finesse attempt."""
        key = (piece_type, target_position.x, target_position.y, target_position.rotation)
        
        if key not in self.error_counts:
            self.error_counts[key] = {'success': 0, 'failure': 0}
        
        if success:
            self.error_counts[key]['success'] += 1
        else:
            self.error_counts[key]['failure'] += 1
        
        self.total_attempts += 1
    
    def get_success_rate(self, piece_type: PieceType, target_position: Position) -> float:
        """Get the success rate for a specific piece placement."""
        key = (piece_type, target_position.x, target_position.y, target_position.rotation)
        
        if key not in self.error_counts:
            return 1.0  # Assume success if no data
        
        data = self.error_counts[key]
        total = data['success'] + data['failure']
        
        if total == 0:
            return 1.0
        
        return data['success'] / total
    
    def get_overall_success_rate(self) -> float:
        """Get the overall finesse success rate."""
        if self.total_attempts == 0:
            return 1.0
        
        total_success = sum(data['success'] for data in self.error_counts.values())
        return total_success / self.total_attempts
    
    def get_problematic_placements(self, threshold: float = 0.8) -> List[Tuple]:
        """Get placements with success rate below threshold."""
        problematic = []
        
        for key, data in self.error_counts.items():
            total = data['success'] + data['failure']
            if total >= 10:  # Only consider placements with sufficient data
                success_rate = data['success'] / total
                if success_rate < threshold:
                    problematic.append((key, success_rate))
        
        return sorted(problematic, key=lambda x: x[1])  # Sort by success rate 