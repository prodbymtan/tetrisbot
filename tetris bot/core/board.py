"""
Board state management for DeepTrix-Z.
Handles board representation, line clearing, garbage, and board evaluation.
"""

from typing import List, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass
from .pieces import Piece, Position


@dataclass
class BoardState:
    """Represents the complete state of a Tetris board."""
    board: np.ndarray  # 20x10 board (0 = empty, 1 = filled)
    current_piece: Optional[Piece]
    hold_piece: Optional[Piece]
    next_pieces: List[Piece]  # Next 5 pieces
    garbage_lines: int
    combo: int
    lines_cleared: int
    level: int
    score: int
    attack: int  # Lines sent to opponent
    
    def __post_init__(self):
        if self.board is None:
            self.board = np.zeros((20, 10), dtype=np.int8)
        if self.next_pieces is None:
            self.next_pieces = []


class Board:
    """Main board class for Tetris game state management."""
    
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20
    VISIBLE_HEIGHT = 20  # Height visible to player
    
    def __init__(self):
        self.board = np.zeros((self.BOARD_HEIGHT, self.BOARD_WIDTH), dtype=np.int8)
        self.current_piece = None
        self.hold_piece = None
        self.next_pieces = []
        self.garbage_lines = 0
        self.combo = 0
        self.lines_cleared = 0
        self.level = 1
        self.score = 0
        self.attack = 0
        self._line_clear_history = []
    
    def reset(self):
        """Reset the board to initial state."""
        self.board = np.zeros((self.BOARD_HEIGHT, self.BOARD_WIDTH), dtype=np.int8)
        self.current_piece = None
        self.hold_piece = None
        self.next_pieces = []
        self.garbage_lines = 0
        self.combo = 0
        self.lines_cleared = 0
        self.level = 1
        self.score = 0
        self.attack = 0
        self._line_clear_history = []
    
    def is_valid_position(self, piece: Piece) -> bool:
        """Check if a piece position is valid (within bounds and not colliding)."""
        cells = piece.get_occupied_cells()
        
        for x, y in cells:
            # Check bounds
            if x < 0 or x >= self.BOARD_WIDTH or y < 0 or y >= self.BOARD_HEIGHT:
                return False
            
            # Check collision with existing pieces
            if self.board[y][x]:
                return False
        
        return True
    
    def place_piece(self, piece: Piece) -> bool:
        """Place a piece on the board. Returns True if successful."""
        if not self.is_valid_position(piece):
            return False
        
        cells = piece.get_occupied_cells()
        for x, y in cells:
            self.board[y][x] = 1
        
        return True
    
    def remove_piece(self, piece: Piece):
        """Remove a piece from the board."""
        cells = piece.get_occupied_cells()
        for x, y in cells:
            if 0 <= x < self.BOARD_WIDTH and 0 <= y < self.BOARD_HEIGHT:
                self.board[y][x] = 0
    
    def get_drop_position(self, piece: Piece) -> Optional[Position]:
        """Get the position where a piece would land if dropped."""
        current_pos = piece.position
        drop_pos = Position(current_pos.x, current_pos.y, current_pos.rotation)
        
        # Drop the piece until it collides
        while True:
            test_pos = Position(drop_pos.x, drop_pos.y + 1, drop_pos.rotation)
            test_piece = Piece(piece.piece_type, test_pos)
            
            if not self.is_valid_position(test_piece):
                break
            
            drop_pos = test_pos
        
        return drop_pos
    
    def clear_lines(self) -> Tuple[int, List[int]]:
        """Clear completed lines and return (lines_cleared, line_indices)."""
        lines_to_clear = []
        
        # Find lines to clear
        for y in range(self.BOARD_HEIGHT):
            if np.all(self.board[y]):
                lines_to_clear.append(y)
        
        if not lines_to_clear:
            return 0, []
        
        # Clear the lines
        for y in sorted(lines_to_clear, reverse=True):
            # Remove the line
            self.board = np.delete(self.board, y, axis=0)
            # Add empty line at top
            self.board = np.vstack([np.zeros((1, self.BOARD_WIDTH), dtype=np.int8), self.board])
        
        # Update stats
        lines_cleared = len(lines_to_clear)
        self.lines_cleared += lines_cleared
        
        # Update combo
        if lines_cleared > 0:
            self.combo += 1
        else:
            self.combo = 0
        
        # Calculate attack (lines sent to opponent)
        attack = self._calculate_attack(lines_cleared, self.combo)
        self.attack += attack
        
        # Update score
        self.score += self._calculate_score(lines_cleared, self.combo)
        
        # Update level
        self.level = (self.lines_cleared // 10) + 1
        
        return lines_cleared, lines_to_clear
    
    def _calculate_attack(self, lines_cleared: int, combo: int) -> int:
        """Calculate attack (lines sent to opponent) based on lines cleared and combo."""
        if lines_cleared == 0:
            return 0
        
        # Base attack values
        attack_values = {
            1: 0,  # Single line
            2: 1,  # Double
            3: 2,  # Triple
            4: 4   # Tetris
        }
        
        base_attack = attack_values.get(lines_cleared, 0)
        
        # Combo bonus
        combo_bonus = max(0, combo - 1)
        
        return base_attack + combo_bonus
    
    def _calculate_score(self, lines_cleared: int, combo: int) -> int:
        """Calculate score based on lines cleared and combo."""
        if lines_cleared == 0:
            return 0
        
        # Base score values
        score_values = {
            1: 100,   # Single
            2: 300,   # Double
            3: 500,   # Triple
            4: 800    # Tetris
        }
        
        base_score = score_values.get(lines_cleared, 0)
        
        # Level multiplier
        level_multiplier = self.level
        
        # Combo bonus
        combo_bonus = combo * 50
        
        return (base_score + combo_bonus) * level_multiplier
    
    def add_garbage_lines(self, lines: int):
        """Add garbage lines to the bottom of the board."""
        if lines <= 0:
            return
        
        # Remove lines from bottom
        self.board = self.board[lines:]
        
        # Add garbage lines at top
        garbage_lines = []
        for _ in range(lines):
            # Create garbage line with one hole
            garbage_line = np.ones(self.BOARD_WIDTH, dtype=np.int8)
            hole_pos = np.random.randint(0, self.BOARD_WIDTH)
            garbage_line[hole_pos] = 0
            garbage_lines.append(garbage_line)
        
        self.board = np.vstack([np.array(garbage_lines), self.board])
        self.garbage_lines += lines
    
    def get_height_map(self) -> List[int]:
        """Get the height of each column."""
        heights = []
        for x in range(self.BOARD_WIDTH):
            height = 0
            for y in range(self.BOARD_HEIGHT):
                if self.board[y][x]:
                    height = self.BOARD_HEIGHT - y
                    break
            heights.append(height)
        return heights
    
    def get_bumpiness(self) -> int:
        """Calculate board bumpiness (sum of height differences between adjacent columns)."""
        heights = self.get_height_map()
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness
    
    def get_holes(self) -> int:
        """Count the number of holes in the board."""
        holes = 0
        for x in range(self.BOARD_WIDTH):
            found_block = False
            for y in range(self.BOARD_HEIGHT):
                if self.board[y][x]:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes
    
    def get_wells(self) -> List[int]:
        """Get the depth of each well (empty column)."""
        wells = []
        for x in range(self.BOARD_WIDTH):
            well_depth = 0
            for y in range(self.BOARD_HEIGHT):
                if self.board[y][x] == 0:
                    well_depth += 1
                else:
                    break
            wells.append(well_depth)
        return wells
    
    def get_t_spin_potential(self) -> int:
        """Calculate T-spin setup potential."""
        # This is a simplified version - a full implementation would be more complex
        t_spin_potential = 0
        
        # Look for T-spin setups (3 corners filled with T piece)
        for y in range(self.BOARD_HEIGHT - 2):
            for x in range(self.BOARD_WIDTH - 2):
                # Check for T-spin setup pattern
                if (self.board[y][x] and self.board[y][x+2] and 
                    self.board[y+2][x] and not self.board[y+1][x+1]):
                    t_spin_potential += 1
        
        return t_spin_potential
    
    def get_combo_potential(self) -> int:
        """Calculate combo potential."""
        # Count the number of lines that are almost complete
        combo_potential = 0
        for y in range(self.BOARD_HEIGHT):
            filled_cells = np.sum(self.board[y])
            if filled_cells >= self.BOARD_WIDTH - 2:  # 8+ cells filled
                combo_potential += 1
        
        return combo_potential
    
    def is_game_over(self) -> bool:
        """Check if the game is over (board is full at spawn area)."""
        # Check if spawn area is blocked
        spawn_area = self.board[0:4, 3:7]  # 4x4 spawn area
        return np.any(spawn_area)
    
    def get_board_state(self) -> BoardState:
        """Get the current board state."""
        return BoardState(
            board=self.board.copy(),
            current_piece=self.current_piece,
            hold_piece=self.hold_piece,
            next_pieces=self.next_pieces.copy(),
            garbage_lines=self.garbage_lines,
            combo=self.combo,
            lines_cleared=self.lines_cleared,
            level=self.level,
            score=self.score,
            attack=self.attack
        )
    
    def set_board_state(self, state: BoardState):
        """Set the board to a specific state."""
        self.board = state.board.copy()
        self.current_piece = state.current_piece
        self.hold_piece = state.hold_piece
        self.next_pieces = state.next_pieces.copy()
        self.garbage_lines = state.garbage_lines
        self.combo = state.combo
        self.lines_cleared = state.lines_cleared
        self.level = state.level
        self.score = state.score
        self.attack = state.attack
    
    def __str__(self):
        """String representation of the board."""
        result = []
        for y in range(self.BOARD_HEIGHT):
            row = ""
            for x in range(self.BOARD_WIDTH):
                if self.board[y][x]:
                    row += "█"
                else:
                    row += "·"
            result.append(row)
        
        # Add current piece if it exists
        if self.current_piece:
            cells = self.current_piece.get_occupied_cells()
            for x, y in cells:
                if 0 <= x < self.BOARD_WIDTH and 0 <= y < self.BOARD_HEIGHT:
                    result[y] = result[y][:x] + "○" + result[y][x+1:]
        
        return "\n".join(result)
    
    def __repr__(self):
        return f"Board(lines_cleared={self.lines_cleared}, level={self.level}, score={self.score})" 