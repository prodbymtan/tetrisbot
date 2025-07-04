"""
Main Tetris engine for DeepTrix-Z.
Coordinates all game components and manages the game loop.
"""

from typing import List, Optional, Callable, Dict, Any
import numpy as np
import time
from dataclasses import dataclass
from .board import Board, BoardState
from .pieces import Piece, PieceType, Position, get_piece_bag
from .finesse import FinesseEngine, FinessePath


@dataclass
class GameConfig:
    """Configuration for the Tetris game."""
    gravity_level: int = 1  # Lines per level
    soft_drop_speed: int = 1  # Lines per frame when soft dropping
    hard_drop_speed: int = 20  # Lines per frame when hard dropping
    lock_delay: int = 15  # Frames before piece locks
    line_clear_delay: int = 6  # Frames to wait after line clear
    garbage_enabled: bool = True
    hold_enabled: bool = True
    next_pieces_count: int = 5
    max_level: int = 20


@dataclass
class GameState:
    """Current state of the Tetris game."""
    board: Board
    current_piece: Optional[Piece]
    hold_piece: Optional[Piece]
    next_pieces: List[PieceType]
    piece_bag: List[PieceType]
    bag_index: int
    level: int
    lines_cleared: int
    score: int
    attack: int
    combo: int
    garbage_lines: int
    frame_count: int
    game_over: bool
    can_hold: bool


class TetrisEngine:
    """Main Tetris game engine."""
    
    def __init__(self, config: Optional[GameConfig] = None):
        self.config = config or GameConfig()
        self.board = Board()
        self.finesse_engine = FinesseEngine()
        
        # Game state
        self.current_piece = None
        self.hold_piece = None
        self.next_pieces = []
        self.piece_bag = []
        self.bag_index = 0
        
        # Game stats
        self.level = 1
        self.lines_cleared = 0
        self.score = 0
        self.attack = 0
        self.combo = 0
        self.garbage_lines = 0
        self.frame_count = 0
        
        # Game flags
        self.game_over = False
        self.can_hold = True
        self.piece_locked = False
        self.lock_delay_frames = 0
        
        # Callbacks
        self.on_piece_placed: Optional[Callable] = None
        self.on_line_cleared: Optional[Callable] = None
        self.on_game_over: Optional[Callable] = None
        self.on_level_up: Optional[Callable] = None
        
        # Initialize the game
        self._initialize_game()
    
    def _initialize_game(self):
        """Initialize the game state."""
        self.board.reset()
        self.piece_bag = get_piece_bag()
        self.bag_index = 0
        self.next_pieces = []
        
        # Fill next pieces queue
        for _ in range(self.config.next_pieces_count):
            self._add_next_piece()
        
        # Spawn first piece
        self._spawn_piece()
    
    def _add_next_piece(self):
        """Add a piece to the next pieces queue."""
        if self.bag_index >= len(self.piece_bag):
            # Create new bag
            self.piece_bag = get_piece_bag()
            self.bag_index = 0
        
        piece_type = self.piece_bag[self.bag_index]
        self.next_pieces.append(piece_type)
        self.bag_index += 1
    
    def _spawn_piece(self):
        """Spawn a new piece."""
        if not self.next_pieces:
            return
        
        piece_type = self.next_pieces.pop(0)
        spawn_pos = Position(3, 0, 0)  # Default spawn position
        
        # Adjust spawn position for specific pieces
        if piece_type == PieceType.I:
            spawn_pos = Position(3, 0, 0)
        elif piece_type == PieceType.O:
            spawn_pos = Position(4, 0, 0)
        
        self.current_piece = Piece(piece_type, spawn_pos)
        self.board.current_piece = self.current_piece
        
        # Check for game over
        if not self.board.is_valid_position(self.current_piece):
            self.game_over = True
            if self.on_game_over:
                self.on_game_over()
            return
        
        # Add new piece to queue
        self._add_next_piece()
        
        # Reset hold ability
        self.can_hold = True
        
        # Reset lock delay
        self.piece_locked = False
        self.lock_delay_frames = 0
    
    def _get_gravity_delay(self) -> int:
        """Get the current gravity delay based on level."""
        # Standard Tetris gravity formula
        if self.level <= 8:
            return max(1, 48 - (self.level * 5))
        elif self.level <= 12:
            return max(1, 8 - ((self.level - 8) * 2))
        elif self.level <= 15:
            return max(1, 2 - (self.level - 12))
        else:
            return 1
    
    def update(self, inputs: Optional[List[str]] = None):
        """Update the game state for one frame."""
        if self.game_over:
            return
        
        self.frame_count += 1
        
        # Handle inputs
        if inputs:
            self._handle_inputs(inputs)
        
        # Apply gravity
        if self.current_piece and not self.piece_locked:
            self._apply_gravity()
        
        # Handle lock delay
        if self.piece_locked:
            self._handle_lock_delay()
    
    def _handle_inputs(self, inputs: List[str]):
        """Handle input actions."""
        if not self.current_piece:
            return
        
        for action in inputs:
            if action == 'left':
                self._move_piece(-1, 0)
            elif action == 'right':
                self._move_piece(1, 0)
            elif action == 'rotate_cw':
                self._rotate_piece(1)
            elif action == 'rotate_ccw':
                self._rotate_piece(-1)
            elif action == 'soft_drop':
                self._soft_drop()
            elif action == 'hard_drop':
                self._hard_drop()
            elif action == 'hold' and self.config.hold_enabled:
                self._hold_piece()
    
    def _move_piece(self, dx: int, dy: int) -> bool:
        """Move the current piece by the given offset."""
        if not self.current_piece:
            return False
        
        new_pos = Position(
            self.current_piece.position.x + dx,
            self.current_piece.position.y + dy,
            self.current_piece.position.rotation
        )
        
        new_piece = Piece(self.current_piece.piece_type, new_pos)
        
        if self.board.is_valid_position(new_piece):
            self.current_piece = new_piece
            self.board.current_piece = new_piece
            return True
        
        return False
    
    def _rotate_piece(self, direction: int) -> bool:
        """Rotate the current piece."""
        if not self.current_piece:
            return False
        
        # Try basic rotation first
        rotated_piece = self.current_piece.rotate(direction)
        
        if self.board.is_valid_position(rotated_piece):
            self.current_piece = rotated_piece
            self.board.current_piece = rotated_piece
            return True
        
        # Try wall kicks
        wall_kicks = self.current_piece.get_wall_kicks(
            self.current_piece.position.rotation,
            rotated_piece.position.rotation
        )
        
        for dx, dy in wall_kicks:
            kicked_piece = Piece(
                self.current_piece.piece_type,
                Position(
                    rotated_piece.position.x + dx,
                    rotated_piece.position.y + dy,
                    rotated_piece.position.rotation
                )
            )
            
            if self.board.is_valid_position(kicked_piece):
                self.current_piece = kicked_piece
                self.board.current_piece = kicked_piece
                return True
        
        return False
    
    def _soft_drop(self):
        """Apply soft drop to the current piece."""
        if self.current_piece:
            self._move_piece(0, self.config.soft_drop_speed)
    
    def _hard_drop(self):
        """Hard drop the current piece."""
        if not self.current_piece:
            return
        
        # Find drop position
        drop_pos = self.board.get_drop_position(self.current_piece)
        if drop_pos:
            # Place piece at drop position
            dropped_piece = Piece(self.current_piece.piece_type, drop_pos)
            self._place_piece(dropped_piece)
    
    def _hold_piece(self):
        """Hold the current piece."""
        if not self.can_hold or not self.current_piece:
            return
        
        # Swap current piece with hold piece
        if self.hold_piece:
            # Place hold piece as current piece
            spawn_pos = self.hold_piece.get_spawn_position()
            self.current_piece = Piece(self.hold_piece.piece_type, spawn_pos)
            self.board.current_piece = self.current_piece
            
            # Set current piece as hold piece
            self.hold_piece = Piece(self.current_piece.piece_type, self.current_piece.position)
        else:
            # First hold - just store current piece and spawn next
            self.hold_piece = Piece(self.current_piece.piece_type, self.current_piece.position)
            self._spawn_piece()
        
        self.can_hold = False
    
    def _apply_gravity(self):
        """Apply gravity to the current piece."""
        if not self.current_piece:
            return
        
        # Check if gravity should be applied this frame
        gravity_delay = self._get_gravity_delay()
        if self.frame_count % gravity_delay != 0:
            return
        
        # Try to move piece down
        if not self._move_piece(0, 1):
            # Piece can't move down - start lock delay
            self.piece_locked = True
            self.lock_delay_frames = 0
    
    def _handle_lock_delay(self):
        """Handle piece lock delay."""
        self.lock_delay_frames += 1
        
        if self.lock_delay_frames >= self.config.lock_delay:
            # Lock the piece
            self._place_piece(self.current_piece)
    
    def _place_piece(self, piece: Piece):
        """Place a piece on the board."""
        if not self.board.place_piece(piece):
            return False
        
        # Clear lines
        lines_cleared, line_indices = self.board.clear_lines()
        
        # Update stats
        self.lines_cleared = self.board.lines_cleared
        self.score = self.board.score
        self.attack = self.board.attack
        self.combo = self.board.combo
        self.garbage_lines = self.board.garbage_lines
        
        # Check for level up
        new_level = (self.lines_cleared // 10) + 1
        if new_level > self.level:
            self.level = new_level
            if self.on_level_up:
                self.on_level_up(self.level)
        
        # Call callbacks
        if self.on_piece_placed:
            self.on_piece_placed(piece, lines_cleared)
        
        if lines_cleared > 0 and self.on_line_cleared:
            self.on_line_cleared(lines_cleared, line_indices)
        
        # Spawn next piece
        self._spawn_piece()
        
        return True
    
    def get_game_state(self) -> GameState:
        """Get the current game state."""
        return GameState(
            board=self.board,
            current_piece=self.current_piece,
            hold_piece=self.hold_piece,
            next_pieces=self.next_pieces.copy(),
            piece_bag=self.piece_bag.copy(),
            bag_index=self.bag_index,
            level=self.level,
            lines_cleared=self.lines_cleared,
            score=self.score,
            attack=self.attack,
            combo=self.combo,
            garbage_lines=self.garbage_lines,
            frame_count=self.frame_count,
            game_over=self.game_over,
            can_hold=self.can_hold
        )
    
    def set_game_state(self, state: GameState):
        """Set the game to a specific state."""
        self.board = state.board
        self.current_piece = state.current_piece
        self.hold_piece = state.hold_piece
        self.next_pieces = state.next_pieces.copy()
        self.piece_bag = state.piece_bag.copy()
        self.bag_index = state.bag_index
        self.level = state.level
        self.lines_cleared = state.lines_cleared
        self.score = state.score
        self.attack = state.attack
        self.combo = state.combo
        self.garbage_lines = state.garbage_lines
        self.frame_count = state.frame_count
        self.game_over = state.game_over
        self.can_hold = state.can_hold
    
    def get_finesse_path(self, target_position: Position) -> Optional[FinessePath]:
        """Get the optimal finesse path to a target position."""
        if not self.current_piece:
            return None
        
        return self.finesse_engine.get_finesse_path(
            self.board, self.current_piece, target_position
        )
    
    def execute_finesse_path(self, path: FinessePath) -> bool:
        """Execute a finesse path."""
        return self.finesse_engine.execute_finesse_path(self.board, path)
    
    def get_all_valid_placements(self) -> List[Position]:
        """Get all valid placement positions for the current piece."""
        if not self.current_piece:
            return []
        
        return self.finesse_engine.get_all_valid_placements(
            self.board, self.current_piece
        )
    
    def get_best_placement(self, evaluator: Callable) -> Optional[Position]:
        """Get the best placement position using an evaluator function."""
        if not self.current_piece:
            return None
        
        return self.finesse_engine.get_best_placement(
            self.board, self.current_piece, evaluator
        )
    
    def add_garbage_lines(self, lines: int):
        """Add garbage lines to the board."""
        self.board.add_garbage_lines(lines)
        self.garbage_lines = self.board.garbage_lines
    
    def reset(self):
        """Reset the game to initial state."""
        self._initialize_game()
        self.level = 1
        self.lines_cleared = 0
        self.score = 0
        self.attack = 0
        self.combo = 0
        self.garbage_lines = 0
        self.frame_count = 0
        self.game_over = False
        self.can_hold = True
        self.piece_locked = False
        self.lock_delay_frames = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current game statistics."""
        return {
            'level': self.level,
            'lines_cleared': self.lines_cleared,
            'score': self.score,
            'attack': self.attack,
            'combo': self.combo,
            'garbage_lines': self.garbage_lines,
            'frame_count': self.frame_count,
            'game_over': self.game_over,
            'board_height': max(self.board.get_height_map()) if self.board.get_height_map() else 0,
            'holes': self.board.get_holes(),
            'bumpiness': self.board.get_bumpiness(),
            't_spin_potential': self.board.get_t_spin_potential(),
            'combo_potential': self.board.get_combo_potential()
        }
    
    def __str__(self):
        """String representation of the game state."""
        result = []
        result.append(f"Level: {self.level}")
        result.append(f"Lines: {self.lines_cleared}")
        result.append(f"Score: {self.score}")
        result.append(f"Attack: {self.attack}")
        result.append(f"Combo: {self.combo}")
        result.append(f"Garbage: {self.garbage_lines}")
        result.append(f"Frame: {self.frame_count}")
        result.append("")
        result.append(str(self.board))
        return "\n".join(result) 