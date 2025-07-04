"""
Basic tests for DeepTrix-Z core functionality.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tetris_engine import TetrisEngine
from core.board import Board
from core.pieces import Piece, PieceType, Position
from core.finesse import FinesseEngine
from ai.evaluation import BoardEvaluator


class TestTetrisEngine(unittest.TestCase):
    """Test the main Tetris engine."""
    
    def setUp(self):
        self.engine = TetrisEngine()
    
    def test_engine_initialization(self):
        """Test that the engine initializes correctly."""
        self.assertIsNotNone(self.engine)
        self.assertFalse(self.engine.game_over)
        self.assertEqual(self.engine.level, 1)
        self.assertEqual(self.engine.lines_cleared, 0)
        self.assertEqual(self.engine.score, 0)
    
    def test_piece_spawning(self):
        """Test that pieces spawn correctly."""
        self.assertIsNotNone(self.engine.current_piece)
        self.assertIsInstance(self.engine.current_piece.piece_type, PieceType)
        self.assertIsInstance(self.engine.current_piece.position, Position)
    
    def test_piece_movement(self):
        """Test piece movement and placement."""
        initial_piece = self.engine.current_piece
        initial_pos = initial_piece.position
        
        # Test movement
        new_pos = Position(initial_pos.x + 1, initial_pos.y, initial_pos.rotation)
        moved_piece = Piece(initial_piece.piece_type, new_pos)
        
        self.assertNotEqual(initial_piece.position.x, moved_piece.position.x)
    
    def test_line_clearing(self):
        """Test line clearing functionality."""
        # Create a board with a full line
        self.engine.board.board[19] = np.ones(10)  # Fill bottom row
        
        # Place a piece to trigger line clearing
        piece = Piece(PieceType.I, Position(3, 18, 0))
        self.engine.board.place_piece(piece)
        
        lines_cleared, _ = self.engine.board.clear_lines()
        self.assertEqual(lines_cleared, 1)


class TestBoard(unittest.TestCase):
    """Test the board functionality."""
    
    def setUp(self):
        self.board = Board()
    
    def test_board_initialization(self):
        """Test board initialization."""
        self.assertEqual(self.board.board.shape, (20, 10))
        self.assertTrue(np.all(self.board.board == 0))
    
    def test_piece_placement(self):
        """Test piece placement on board."""
        piece = Piece(PieceType.I, Position(3, 0, 0))
        
        # Test valid placement
        self.assertTrue(self.board.is_valid_position(piece))
        self.assertTrue(self.board.place_piece(piece))
        
        # Test invalid placement (out of bounds)
        invalid_piece = Piece(PieceType.I, Position(-1, 0, 0))
        self.assertFalse(self.board.is_valid_position(invalid_piece))
    
    def test_height_map(self):
        """Test height map calculation."""
        # Place a piece
        piece = Piece(PieceType.I, Position(3, 18, 0))
        self.board.place_piece(piece)
        
        height_map = self.board.get_height_map()
        self.assertEqual(len(height_map), 10)
        self.assertEqual(height_map[3], 2)  # Height should be 2 for column 3
    
    def test_holes_detection(self):
        """Test hole detection."""
        # Create a board with holes
        self.board.board[19, 3] = 1  # Bottom piece
        self.board.board[18, 3] = 0  # Hole above
        self.board.board[17, 3] = 1  # Piece above hole
        
        holes = self.board.get_holes()
        self.assertEqual(holes, 1)


class TestPieces(unittest.TestCase):
    """Test piece functionality."""
    
    def test_piece_creation(self):
        """Test piece creation."""
        piece = Piece(PieceType.T, Position(3, 0, 0))
        self.assertEqual(piece.piece_type, PieceType.T)
        self.assertEqual(piece.position.x, 3)
        self.assertEqual(piece.position.y, 0)
        self.assertEqual(piece.position.rotation, 0)
    
    def test_piece_rotation(self):
        """Test piece rotation."""
        piece = Piece(PieceType.T, Position(3, 0, 0))
        rotated = piece.rotate(1)  # Clockwise
        
        self.assertEqual(rotated.position.rotation, 1)
        self.assertEqual(rotated.position.x, 3)
        self.assertEqual(rotated.position.y, 0)
    
    def test_piece_translation(self):
        """Test piece translation."""
        piece = Piece(PieceType.T, Position(3, 0, 0))
        translated = piece.translate(2, 1)
        
        self.assertEqual(translated.position.x, 5)
        self.assertEqual(translated.position.y, 1)
    
    def test_piece_shape(self):
        """Test piece shape generation."""
        piece = Piece(PieceType.I, Position(3, 0, 0))
        shape = piece.shape
        
        self.assertEqual(shape.shape, (4, 4))
        self.assertTrue(np.any(shape))  # Should have some filled cells
    
    def test_occupied_cells(self):
        """Test occupied cells calculation."""
        piece = Piece(PieceType.I, Position(3, 0, 0))
        cells = piece.get_occupied_cells()
        
        self.assertEqual(len(cells), 4)  # I piece has 4 cells
        for x, y in cells:
            self.assertTrue(0 <= x < 10)  # Within board bounds
            self.assertTrue(0 <= y < 20)


class TestFinesseEngine(unittest.TestCase):
    """Test finesse engine functionality."""
    
    def setUp(self):
        self.finesse_engine = FinesseEngine()
        self.board = Board()
    
    def test_finesse_engine_initialization(self):
        """Test finesse engine initialization."""
        self.assertIsNotNone(self.finesse_engine)
        self.assertEqual(self.finesse_engine.current_frame, 0)
    
    def test_valid_placements(self):
        """Test valid placement calculation."""
        piece = Piece(PieceType.I, Position(3, 0, 0))
        placements = self.finesse_engine.get_all_valid_placements(self.board, piece)
        
        self.assertIsInstance(placements, list)
        self.assertGreater(len(placements), 0)
        
        for placement in placements:
            self.assertIsInstance(placement, Position)
            self.assertTrue(0 <= placement.x < 10)
            self.assertTrue(0 <= placement.y < 20)
            self.assertTrue(0 <= placement.rotation < 4)
    
    def test_finesse_path(self):
        """Test finesse path calculation."""
        piece = Piece(PieceType.I, Position(3, 0, 0))
        target = Position(3, 18, 0)
        
        path = self.finesse_engine.get_finesse_path(self.board, piece, target)
        
        if path:  # Path might not exist for all positions
            self.assertIsNotNone(path.inputs)
            self.assertEqual(path.final_position, target)
            self.assertTrue(path.is_optimal)


class TestBoardEvaluator(unittest.TestCase):
    """Test board evaluation functionality."""
    
    def setUp(self):
        self.evaluator = BoardEvaluator()
        self.board = Board()
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        self.assertIsNotNone(self.evaluator)
        self.assertIsNotNone(self.evaluator.weights)
    
    def test_board_evaluation(self):
        """Test basic board evaluation."""
        score = self.evaluator.evaluate_board(self.board)
        self.assertIsInstance(score, float)
        self.assertGreater(score, float('-inf'))
    
    def test_empty_board_evaluation(self):
        """Test evaluation of empty board."""
        score = self.evaluator.evaluate_board(self.board)
        # Empty board should have a reasonable score
        self.assertGreater(score, -1000)
    
    def test_detailed_evaluation(self):
        """Test detailed evaluation metrics."""
        metrics = self.evaluator.get_detailed_evaluation(self.board)
        
        expected_keys = [
            'max_height', 'avg_height', 'holes', 'bumpiness',
            'max_well_depth', 'row_transitions', 'col_transitions',
            'edge_touches', 't_spin_potential', 'combo_potential',
            'perfect_clear_potential', 'garbage_lines', 'lines_cleared',
            'combo', 'overall_score'
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (int, float))


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_complete_game_cycle(self):
        """Test a complete game cycle."""
        engine = TetrisEngine()
        
        # Run a few moves
        for _ in range(10):
            if engine.game_over:
                break
            
            if engine.current_piece:
                # Get valid placements
                placements = engine.get_all_valid_placements()
                if placements:
                    # Choose first valid placement
                    placement = placements[0]
                    piece = Piece(engine.current_piece.piece_type, placement)
                    engine.place_piece(piece)
            
            engine.update()
        
        # Game should still be running or ended naturally
        self.assertTrue(engine.game_over or engine.current_piece is not None)
    
    def test_mcts_integration(self):
        """Test MCTS integration (basic test)."""
        try:
            from ai.neural_net import NeuralNetworkManager
            from ai.mcts import MCTS
            
            network_manager = NeuralNetworkManager()
            mcts = MCTS(network_manager, num_simulations=10)  # Small number for test
            
            engine = TetrisEngine()
            game_state = engine.get_game_state()
            
            # This should not crash
            action = mcts.search(game_state)
            self.assertIsInstance(action, tuple)
            self.assertEqual(len(action), 3)
            
        except ImportError:
            self.skipTest("Neural network dependencies not available")


if __name__ == '__main__':
    unittest.main() 