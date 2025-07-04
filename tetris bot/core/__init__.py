"""
Core module for DeepTrix-Z Tetris bot.
Contains the fundamental game engine, board management, and piece operations.
"""

from .tetris_engine import TetrisEngine
from .board import Board
from .pieces import Piece, PieceType, Position
from .finesse import FinesseEngine

__all__ = ['TetrisEngine', 'Board', 'Piece', 'PieceType', 'Position', 'FinesseEngine'] 