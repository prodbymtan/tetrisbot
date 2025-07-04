"""
Tetromino piece definitions and operations for DeepTrix-Z.
Includes all 7 Tetris pieces with their rotations, wall kicks, and finesse data.
"""

from enum import Enum
from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass


class PieceType(Enum):
    """The 7 standard Tetris pieces."""
    I = 0
    O = 1
    T = 2
    S = 3
    Z = 4
    J = 5
    L = 6


@dataclass
class Position:
    """Represents a piece position on the board."""
    x: int
    y: int
    rotation: int  # 0, 1, 2, 3 for 0°, 90°, 180°, 270°


class Piece:
    """Represents a Tetris piece with all its properties and operations."""
    
    # Piece definitions: [rotation][y][x] where 1 = filled, 0 = empty
    PIECE_DEFINITIONS = {
        PieceType.I: [
            [[0, 0, 0, 0],
             [1, 1, 1, 1],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],
            [[0, 0, 1, 0],
             [0, 0, 1, 0],
             [0, 0, 1, 0],
             [0, 0, 1, 0]],
            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [1, 1, 1, 1],
             [0, 0, 0, 0]],
            [[0, 1, 0, 0],
             [0, 1, 0, 0],
             [0, 1, 0, 0],
             [0, 1, 0, 0]]
        ],
        PieceType.O: [
            [[0, 1, 1, 0],
             [0, 1, 1, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]] * 4  # O piece has only one rotation
        ],
        PieceType.T: [
            [[0, 1, 0, 0],
             [1, 1, 1, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],
            [[0, 1, 0, 0],
             [0, 1, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 0]],
            [[0, 0, 0, 0],
             [1, 1, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 0]],
            [[0, 1, 0, 0],
             [1, 1, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 0]]
        ],
        PieceType.S: [
            [[0, 1, 1, 0],
             [1, 1, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],
            [[0, 1, 0, 0],
             [0, 1, 1, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 0]],
            [[0, 0, 0, 0],
             [0, 1, 1, 0],
             [1, 1, 0, 0],
             [0, 0, 0, 0]],
            [[1, 0, 0, 0],
             [1, 1, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 0]]
        ],
        PieceType.Z: [
            [[1, 1, 0, 0],
             [0, 1, 1, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],
            [[0, 0, 1, 0],
             [0, 1, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 0]],
            [[0, 0, 0, 0],
             [1, 1, 0, 0],
             [0, 1, 1, 0],
             [0, 0, 0, 0]],
            [[0, 1, 0, 0],
             [1, 1, 0, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 0]]
        ],
        PieceType.J: [
            [[1, 0, 0, 0],
             [1, 1, 1, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],
            [[0, 1, 1, 0],
             [0, 1, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 0]],
            [[0, 0, 0, 0],
             [1, 1, 1, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 0]],
            [[0, 1, 0, 0],
             [0, 1, 0, 0],
             [1, 1, 0, 0],
             [0, 0, 0, 0]]
        ],
        PieceType.L: [
            [[0, 0, 1, 0],
             [1, 1, 1, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],
            [[0, 1, 0, 0],
             [0, 1, 0, 0],
             [0, 1, 1, 0],
             [0, 0, 0, 0]],
            [[0, 0, 0, 0],
             [1, 1, 1, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 0]],
            [[1, 1, 0, 0],
             [0, 1, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 0]]
        ]
    }
    
    # SRS (Super Rotation System) wall kick data
    # Format: [from_rotation][to_rotation][test_number] = (x_offset, y_offset)
    SRS_WALL_KICKS = {
        PieceType.I: {
            (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
            (1, 0): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
            (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
            (2, 1): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
            (2, 3): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
            (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
            (3, 0): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
            (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)]
        },
        # Other pieces use standard SRS
        PieceType.T: {
            (0, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
            (1, 0): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
            (1, 2): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
            (2, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
            (2, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
            (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
            (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
            (0, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)]
        }
    }
    
    # Standard SRS for S, Z, J, L pieces
    STANDARD_SRS = {
        (0, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
        (1, 0): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
        (1, 2): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
        (2, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
        (2, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
        (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
        (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
        (0, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)]
    }
    
    def __init__(self, piece_type: PieceType, position: Position):
        self.piece_type = piece_type
        self.position = position
        self._shape = None
    
    @property
    def shape(self) -> np.ndarray:
        """Get the current shape of the piece based on its rotation."""
        if self._shape is None:
            # Fix: O piece only has one rotation (index 0)
            rotation = 0 if self.piece_type == PieceType.O else self.position.rotation
            self._shape = np.array(self.PIECE_DEFINITIONS[self.piece_type][rotation])
        return self._shape
    
    def get_occupied_cells(self) -> List[Tuple[int, int]]:
        """Get the board coordinates occupied by this piece."""
        cells = []
        shape = self.shape
        for y in range(4):
            for x in range(4):
                if shape[y][x]:
                    board_x = self.position.x + x
                    board_y = self.position.y + y
                    cells.append((board_x, board_y))
        return cells
    
    def rotate(self, direction: int) -> 'Piece':
        """Rotate the piece (1 for clockwise, -1 for counterclockwise)."""
        new_rotation = (self.position.rotation + direction) % 4
        return Piece(self.piece_type, Position(
            self.position.x, self.position.y, new_rotation
        ))
    
    def translate(self, dx: int, dy: int) -> 'Piece':
        """Move the piece by the given offsets."""
        return Piece(self.piece_type, Position(
            self.position.x + dx, self.position.y + dy, self.position.rotation
        ))
    
    def get_wall_kicks(self, from_rotation: int, to_rotation: int) -> List[Tuple[int, int]]:
        """Get the wall kick offsets for a rotation."""
        if self.piece_type == PieceType.O:
            return [(0, 0)]  # O piece doesn't need wall kicks
        
        key = (from_rotation, to_rotation)
        if self.piece_type == PieceType.I:
            return self.SRS_WALL_KICKS[PieceType.I].get(key, [(0, 0)])
        elif self.piece_type == PieceType.T:
            return self.SRS_WALL_KICKS[PieceType.T].get(key, [(0, 0)])
        else:
            return self.STANDARD_SRS.get(key, [(0, 0)])
    
    def get_spawn_position(self) -> Position:
        """Get the spawn position for this piece type."""
        # Standard spawn positions (centered horizontally, at the top)
        spawn_x = 3  # Center of 10-width board
        spawn_y = 0  # Top of the board
        
        # Adjust for piece-specific spawn positions
        if self.piece_type == PieceType.I:
            spawn_x = 3
        elif self.piece_type == PieceType.O:
            spawn_x = 4
        
        return Position(spawn_x, spawn_y, 0)
    
    def __eq__(self, other):
        if not isinstance(other, Piece):
            return False
        return (self.piece_type == other.piece_type and 
                self.position.x == other.position.x and
                self.position.y == other.position.y and
                self.position.rotation == other.position.rotation)
    
    def __hash__(self):
        return hash((self.piece_type, self.position.x, self.position.y, self.position.rotation))
    
    def __repr__(self):
        return f"Piece({self.piece_type.name}, x={self.position.x}, y={self.position.y}, r={self.position.rotation})"


def get_all_piece_types() -> List[PieceType]:
    """Get all piece types."""
    return list(PieceType)


def get_random_piece() -> PieceType:
    """Get a random piece type."""
    return np.random.choice(list(PieceType))


def get_piece_bag() -> List[PieceType]:
    """Get a 7-bag of pieces (one of each piece in random order)."""
    pieces = list(PieceType)
    np.random.shuffle(pieces)
    return pieces 