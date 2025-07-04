import unittest
from deeptrix_z_project.deeptrix_z.pieces import Piece, PieceFactory, TETROMINOES

class TestPieces(unittest.TestCase):

    def test_piece_creation(self):
        for name, data in TETROMINOES.items():
            piece = Piece(name, data['color'], data['rotations'])
            self.assertEqual(piece.name, name)
            self.assertEqual(piece.color, data['color'])
            self.assertEqual(len(piece.rotations), 4) # All pieces should have 4 rotation states defined
            self.assertEqual(piece.rotation_state, 0)
            self.assertEqual(piece.shape, data['rotations'][0])

    def test_piece_rotation(self):
        t_piece_data = TETROMINOES['T']
        piece = Piece('T', t_piece_data['color'], t_piece_data['rotations'])

        # Initial state
        self.assertEqual(piece.rotation_state, 0)
        self.assertEqual(piece.shape, t_piece_data['rotations'][0])

        # Rotate clockwise
        piece.rotate(1) # CW
        self.assertEqual(piece.rotation_state, 1)
        self.assertEqual(piece.shape, t_piece_data['rotations'][1])

        # Rotate clockwise again
        piece.rotate(1) # CW
        self.assertEqual(piece.rotation_state, 2)
        self.assertEqual(piece.shape, t_piece_data['rotations'][2])

        # Rotate counter-clockwise
        piece.rotate(-1) # CCW
        self.assertEqual(piece.rotation_state, 1)
        self.assertEqual(piece.shape, t_piece_data['rotations'][1])

        # Rotate to wrap around (CW from state 3 to 0)
        piece.rotation_state = 3
        piece.shape = piece.rotations[3]
        piece.rotate(1) # CW
        self.assertEqual(piece.rotation_state, 0)
        self.assertEqual(piece.shape, t_piece_data['rotations'][0])

        # Rotate to wrap around (CCW from state 0 to 3)
        piece.rotate(-1) # CCW
        self.assertEqual(piece.rotation_state, 3)
        self.assertEqual(piece.shape, t_piece_data['rotations'][3])

    def test_get_block_positions(self):
        o_piece_data = TETROMINOES['O']
        # O piece shape: [[1,1], [1,1]]
        piece = Piece('O', o_piece_data['color'], o_piece_data['rotations'])
        piece.x = 3
        piece.y = 5
        # Expected positions: (y,x), (y,x+1), (y+1,x), (y+1,x+1)
        # (5,3), (5,4), (6,3), (6,4)
        expected_pos = sorted([(5,3), (5,4), (6,3), (6,4)])
        actual_pos = sorted(piece.get_block_positions())
        self.assertEqual(actual_pos, expected_pos)

        # Test T piece initial rotation
        # T shape: [[0,1,0], [1,1,1], [0,0,0]]
        t_piece_data = TETROMINOES['T']
        piece_t = Piece('T', t_piece_data['color'], t_piece_data['rotations'])
        piece_t.x = 4
        piece_t.y = 2
        # Expected: (2, 4+1), (3, 4), (3, 4+1), (3, 4+2)
        # (2,5), (3,4), (3,5), (3,6)
        expected_pos_t = sorted([(2,5), (3,4), (3,5), (3,6)])
        actual_pos_t = sorted(piece_t.get_block_positions())
        self.assertEqual(actual_pos_t, expected_pos_t)


    def test_piece_factory_7_bag(self):
        factory = PieceFactory()
        piece_names_in_bag = list(TETROMINOES.keys())

        generated_pieces_set1 = set()
        for _ in range(len(piece_names_in_bag)):
            piece = factory.next_piece()
            self.assertIn(piece.name, piece_names_in_bag)
            self.assertNotIn(piece.name, generated_pieces_set1) # Check for uniqueness in first bag
            generated_pieces_set1.add(piece.name)
        self.assertEqual(len(generated_pieces_set1), len(piece_names_in_bag)) # Ensure all pieces were generated

        generated_pieces_set2 = set()
        for _ in range(len(piece_names_in_bag)):
            piece = factory.next_piece()
            self.assertIn(piece.name, piece_names_in_bag)
            self.assertNotIn(piece.name, generated_pieces_set2)
            generated_pieces_set2.add(piece.name)
        self.assertEqual(len(generated_pieces_set2), len(piece_names_in_bag))

        # Check that the two bags are not identical in order (highly probable with shuffle)
        # This is not a strict test of shuffle, but a basic check.
        # To be more robust, one might check counts over many generations.
        # For now, just ensure it produces pieces.
        self.assertTrue(len(factory.bag) < len(piece_names_in_bag)) # Bag should be partially empty now


if __name__ == '__main__':
    unittest.main()
