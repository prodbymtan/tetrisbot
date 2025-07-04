import unittest
import numpy as np

from deeptrix_z_project.deeptrix_z.board import Board
from deeptrix_z_project.deeptrix_z.pieces import Piece, PieceFactory, TETROMINOES
from deeptrix_z_project.deeptrix_z.exceptions import GameOverException

class TestBoard(unittest.TestCase):

    def setUp(self):
        # Default board for many tests: 10x20 visible, 4 hidden for easier math
        self.board = Board(width=10, height=20, hidden_rows=4)
        self.piece_factory = PieceFactory()

    def _get_piece_by_name(self, name: str) -> Piece:
        data = TETROMINOES[name]
        return Piece(name, data['color'], data['rotations'])

    def test_board_initialization(self):
        self.assertEqual(self.board.width, 10)
        self.assertEqual(self.board.height, 20)
        self.assertEqual(self.board.total_height, 24) # 20 visible + 4 hidden
        self.assertEqual(self.board.grid.shape, (24, 10))
        self.assertTrue(np.all(self.board.grid == 0)) # Board should be empty

    def test_spawn_piece_normal(self):
        piece_t = self._get_piece_by_name('T')
        self.assertTrue(self.board.spawn_piece(piece_t))
        # T piece: [[0,1,0], [1,1,1], [0,0,0]], width 3. Spawns centered.
        # Board width 10. Spawn x = 10 // 2 - (3 // 2) = 5 - 1 = 4.
        # Spawn y = hidden_rows - 2 = 4 - 2 = 2 (for T, L, J, S, Z)
        # For I: x = 10//2 - 2 = 3. For O: x = 10//2 - 1 = 4.
        self.assertEqual(piece_t.x, 4) # (10 // 2) - (3 // 2) -> 5 - 1 = 4
        self.assertEqual(piece_t.y, self.board.spawn_pos_y) # Default spawn_pos_y = hidden_rows - 2
                                                          # or specific adjustment in spawn_piece

        piece_i = self._get_piece_by_name('I')
        self.assertTrue(self.board.spawn_piece(piece_i))
        # I piece: [[0,0,0,0], [1,1,1,1], [0,0,0,0], [0,0,0,0]], width 4
        # Spawn x = 10 // 2 - 2 = 3
        self.assertEqual(piece_i.x, 3)
        self.assertEqual(piece_i.y, self.board.spawn_pos_y)

    def test_spawn_piece_game_over(self):
        # Fill the spawn area to cause game over
        # Spawn area for T is around y = hidden_rows - 2 = 2, x = 4
        # T piece shape: [[0,1,0], [1,1,1], [0,0,0]]
        # Occupies (y+0, x+1), (y+1,x), (y+1,x+1), (y+1,x+2)
        # If y=2, x=4: (2,5), (3,4), (3,5), (3,6)
        self.board.grid[2, 5] = 1 # Block one cell in T's spawn path

        piece_t = self._get_piece_by_name('T')
        with self.assertRaises(GameOverException):
            self.board.spawn_piece(piece_t)

    def test_is_valid_position(self):
        piece_o = self._get_piece_by_name('O') # Shape [[1,1],[1,1]]
        self.board.spawn_piece(piece_o) # Spawns at e.g. x=4, y=2 (hidden_rows=4)

        self.assertTrue(self.board.is_valid_position(piece_o)) # Current position

        # Out of bounds left
        piece_o.x = -1
        self.assertFalse(self.board.is_valid_position(piece_o))
        piece_o.x = 0 # Reset

        # Out of bounds right
        piece_o.x = self.board.width - 1 # O is 2 wide, x=9 means it occupies 9, 10. Max index is 9.
                                         # x=9, shape[0][1] is at col 10, invalid.
        self.assertFalse(self.board.is_valid_position(piece_o))
        piece_o.x = self.board.width - 2 # x=8, occupies 8,9. Valid.
        self.assertTrue(self.board.is_valid_position(piece_o))


        # Out of bounds bottom (total_height)
        piece_o.y = self.board.total_height - 1 # O is 2 high. y = 23, occupies 23, 24. Invalid.
        self.assertFalse(self.board.is_valid_position(piece_o))
        piece_o.y = self.board.total_height - 2 # y=22, occupies 22,23. Valid.
        self.assertTrue(self.board.is_valid_position(piece_o))

        # Collision with existing block
        self.board.spawn_piece(piece_o) # Reset spawn pos x=4, y=2
        self.board.grid[piece_o.y + 1, piece_o.x + 1] = 1 # Place a block where O would be
        self.assertFalse(self.board.is_valid_position(piece_o))
        self.board.grid[piece_o.y + 1, piece_o.x + 1] = 0 # Clear block

    def test_move_piece(self):
        piece_l = self._get_piece_by_name('L')
        self.board.spawn_piece(piece_l)
        initial_x, initial_y = piece_l.x, piece_l.y

        # Move right
        self.assertTrue(self.board.move_piece(piece_l, 1, 0))
        self.assertEqual(piece_l.x, initial_x + 1)
        # Move down
        self.assertTrue(self.board.move_piece(piece_l, 0, 1))
        self.assertEqual(piece_l.y, initial_y + 1)

        # Move into wall (blocked)
        piece_l.x = 0
        self.assertFalse(self.board.move_piece(piece_l, -1, 0)) # Try move left into wall
        self.assertEqual(piece_l.x, 0) # Should not have moved

        # Move into existing piece (blocked)
        self.board.grid[initial_y + 2, initial_x] = 1 # Place a block below
        piece_l.y = initial_y
        piece_l.x = initial_x
        self.assertFalse(self.board.move_piece(piece_l, 0, 1)) # Try move down
        self.assertEqual(piece_l.y, initial_y) # Should not have moved
        self.board.grid[initial_y + 2, initial_x] = 0 # Clear

    def test_lock_piece_and_clear_lines(self):
        # Create a situation for line clear
        # Fill entire row at bottom of visible area
        # Visible area starts at self.board.hidden_rows (4)
        # Bottom visible row is self.board.hidden_rows + self.board.height - 1 = 4 + 20 - 1 = 23
        bottom_row_idx = self.board.total_height - 1 # Index of the absolute bottom row

        # Fill 9 cells of the bottom row
        for c in range(self.board.width -1):
            self.board.grid[bottom_row_idx, c] = 1

        # Create an I piece to complete the line
        # I piece: [[0,0,0,0], [1,1,1,1], [0,0,0,0], [0,0,0,0]] (rot 0)
        # We want its [1,1,1,1] part to land on the bottom row.
        # For test simplicity, manually place and lock a small piece (1x1)
        # to complete the line.

        # Simpler: use a 1x1 piece (e.g. a modified 'O' or manual piece)
        # For now, let's use an 'I' piece and try to place it.
        # This requires rotation and movement logic to be robust, which is what we are testing.

        # Let's test _clear_lines directly first for simplicity
        self.board.grid[bottom_row_idx, :] = 1 # Fill entire bottom row
        self.board.grid[bottom_row_idx -1, :5] = 2 # Fill half of row above

        lines_cleared = self.board._clear_lines()
        self.assertEqual(lines_cleared, 1)
        self.assertTrue(np.all(self.board.grid[bottom_row_idx, :] == 0)) # Bottom row should be cleared
        # Row that was at bottom_row_idx - 1 should have shifted down to bottom_row_idx
        self.assertTrue(np.all(self.board.grid[bottom_row_idx, :5] == 2))
        self.assertTrue(np.all(self.board.grid[bottom_row_idx, 5:] == 0))
        # Topmost rows should be empty
        self.assertTrue(np.all(self.board.grid[0, :] == 0))

        # Test locking a piece that causes a clear
        self.setUp() # Reset board
        # Fill 9 cells of row y=23 (total_height-1)
        for c in range(self.board.width - 1): # cols 0-8
            self.board.grid[self.board.total_height - 1, c] = self.board.name_to_id('L')

        # Create a 1x1 piece (e.g. from O, take one block) to drop into col 9, row 23
        # Piece 'O' is [[1,1],[1,1]]. Let's use a single block from it.
        # Manually create a piece that is just one block for simplicity here.
        # This is hard without a piece like that. Let's place an I piece horizontally.
        i_piece = self._get_piece_by_name('I') # rot 0 is horizontal: [1,1,1,1] on its 2nd row
        self.board.spawn_piece(i_piece) # x=3, y=2 (hidden_rows=4) for I

        # Manually move I piece to fill the last cell of bottom line
        i_piece.x = self.board.width - 4 # To place its 4 blocks at columns 6,7,8,9
        i_piece.y = self.board.total_height - 1 - 1 # I piece's blocks are at y+1 for rot0.
                                                    # So piece.y should be total_height - 2 for its blocks to be on total_height -1

        # Check if this placement is valid before lock (it might not be due to other parts of I)
        # This is tricky. Let's simplify the setup for line clear test with lock_piece.
        # Fill all but one cell of the last line
        self.setUp() # reset
        for c in range(self.board.width):
            if c != self.board.width -1 : # leave last cell empty
                 self.board.grid[self.board.total_height -1, c] = self.board.name_to_id('L')

        # Create a piece that will fill just that one cell (e.g. vertical I piece, part of it)
        s_piece = self._get_piece_by_name('S') # S: [[0,1,1],[1,1,0]]
        self.board.spawn_piece(s_piece) # x=4, y=2
        s_piece.x = self.board.width - 2 # S is 3 wide. x=8. Blocks at (y, x+1), (y,x+2), (y+1,x), (y+1,x+1)
                                        # (y,9), (y,10-invalid), (y+1,8), (y+1,9)
                                        # Need to use a piece that can land a single block at (total_h-1, width-1)
        # Let's use a piece that is 1 wide, like I piece rotated.
        i_piece_rot1 = self._get_piece_by_name('I')
        i_piece_rot1.rotate(1) # Rotated I: [[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]] (shape[0][2] is a block)
        self.board.spawn_piece(i_piece_rot1)
        i_piece_rot1.x = self.board.width - 1 - 2 # x=7. Block at (y, x+2) -> (y,9)
        i_piece_rot1.y = self.board.total_height - 4 # Land its 4 blocks from y=(total_h-4) to (total_h-1)
                                                     # Blocks at (y+0,x+2), (y+1,x+2), (y+2,x+2), (y+3,x+2)
                                                     # So at (total_h-4, 9), (total_h-3, 9), (total_h-2, 9), (total_h-1, 9)

        # Check if this desired position is valid
        self.assertTrue(self.board.is_valid_position(i_piece_rot1))

        lines, tspin, pc = self.board.lock_piece(i_piece_rot1)
        self.assertEqual(lines, 1)
        self.assertFalse(tspin)
        self.assertFalse(pc)
        # Check that the line was cleared and piece parts are on board
        self.assertTrue(np.all(self.board.grid[self.board.total_height -1, :] == 0)) # Cleared line
        # Check that other parts of I piece are there
        self.assertEqual(self.board.grid[self.board.total_height - 2, self.board.width -1], self.board.name_to_id('I'))


    def test_perfect_clear(self):
        self.assertTrue(self.board._check_perfect_clear()) # Initially empty board
        # Add one block to make it not PC
        self.board.grid[self.board.total_height -1, 0] = 1
        self.assertFalse(self.board._check_perfect_clear())

        # Test PC after a line clear that empties the board
        self.board.grid[:,:] = 0 # Clear board
        # Place a full line with an I piece
        i_piece = self._get_piece_by_name('I')
        self.board.spawn_piece(i_piece)
        i_piece.x = 0 # Place at left edge
        i_piece.y = self.board.total_height - 1 - 1 # So its [1,1,1,1] part is on the last row

        # Fill the rest of the line so I-piece makes a tetris and PC
        # I piece shape is 4x4. For rot0, blocks are piece.shape[1][0] to piece.shape[1][3]
        # So piece.x = 0 means blocks at cols 0,1,2,3 on row piece.y+1
        # We need to fill cols 4,5,6,7,8,9 on row piece.y+1
        for c in range(i_piece.get_bounding_box_size()[1], self.board.width):
            self.board.grid[i_piece.y + 1, c] = self.board.name_to_id('L') # Some other piece type

        lines, tspin, pc = self.board.lock_piece(i_piece)
        self.assertEqual(lines, 1) # Only one line made by the I piece and others
        self.assertTrue(pc) # Board should be empty after this clear


    def test_srs_wall_kicks_t_piece(self):
        # Test T-piece rotation 0 -> 1 (clockwise)
        # T piece: [[0,1,0], [1,1,1], [0,0,0]]
        # Board: 10W x 24H (4 hidden)
        # Place T piece at x=0, y=22 (near bottom, against left wall)
        # Its actual blocks: (22,1), (23,0), (23,1), (23,2)
        t_piece = self._get_piece_by_name('T')
        self.board.spawn_piece(t_piece) # spawns high, e.g. x=4, y=2
        t_piece.x = 0
        t_piece.y = self.board.total_height - 2 # y=22. T is 2 rows high effectively.

        # Initial state: Rotation 0. Blocks: (22,1), (23,0), (23,1), (23,2)
        # Grid should be empty here.
        self.assertTrue(self.board.is_valid_position(t_piece))

        # Try to rotate clockwise (0 -> 1). This should fail without kicks.
        # Rotated T (state 1): [[0,1,0],[0,1,1],[0,1,0]]. At (0,22), this would be:
        # (22,1), (23,1), (23,2), (24,1)-out of bounds.
        # Kick data for T (0->1): (0,0), (-1,0), (-1,+1), (0,-2), (-1,-2)
        # Wiki kicks: (0,0), (+1,0), (+1,-1), (0,+2), (+1,+2) - if our table is (dx, -dy_wiki)
        # Our table JLSTZ_WALL_KICKS (0,1): [(0,0), (-1,0), (-1,+1), (0,-2), (-1,-2)]
        # These are (dx, dy_world_positive_up)
        # So board.rotate_piece uses (dx, final_kick_dy = -kick_dy_table)
        # Test 1 (0,0): piece at (0,22). Fails (part of T goes to y=24).
        # Test 2 (-1,0) -> (dx, -dy_wiki) -> (+1,0) in wiki -> dx=+1, dy=0.
        #   Our table: (-1,0) -> dx=-1. This seems to be direct (dx, dy_game_world_positive_up)
        #   The code uses `final_kick_dy = -kick_dy` for `is_valid_position(piece, offset_x=kick_dx, offset_y=final_kick_dy)`
        #   So if table is (-1, +1), it tests dx=-1, dy=-1 (left 1, up 1).

        # Let's place a wall to force a specific kick.
        # T at x=0, y=21. Blocks: (21,1), (22,0), (22,1), (22,2)
        t_piece.x = 0
        t_piece.y = 21
        self.board.grid[21,0] = 1 # Wall at (21,0) to block (0,0) kick's new (22,0) if T moves.
                                   # T-rot1: (21,1),(22,1),(22,2),(23,1)
                                   # (0,0) kick: x=0, y=21. Shape is rot1. Blocks: (21,1), (22,1), (22,2), (23,1). Valid.

        # If T is at x=0, y=21. Rotate 0->1 (CW)
        # Kick (0,0): new shape at (0,21). Valid. Piece moves to (0,21), rot1.
        self.assertTrue(self.board.rotate_piece(t_piece, 1))
        self.assertEqual(t_piece.rotation_state, 1)
        self.assertEqual(t_piece.x, 0) # Based on (0,0) kick
        self.assertEqual(t_piece.y, 21)

        # Force a different kick:
        self.setUp() # Reset board
        t_piece = self._get_piece_by_name('T')
        self.board.spawn_piece(t_piece)
        t_piece.x = 0
        t_piece.y = 21
        # Block the (0,0) kick's position for rot1
        # Rot1 shape: [[0,1,0],[0,1,1],[0,1,0]]. Blocks at (y,x+1), (y+1,x+1), (y+1,x+2), (y+2,x+1)
        # If (0,0) kick: (21,1), (22,1), (22,2), (23,1). Let's block (21,1)
        self.board.grid[21,1] = 1

        # Kicks (0,1) for JLSTZ: (0,0), (-1,0), (-1,+1), (0,-2), (-1,-2)
        # Test 1 (0,0): dx=0, dy=0. (piece.x=0, piece.y=21). Invalid due to grid[21,1].
        # Test 2 (-1,0): dx=-1, dy_table=0 -> final_dy=0. (piece.x=-1, piece.y=21). Invalid (out of bounds left).
        # (This depends on how piece.x is updated before is_valid_position is called for kicks)
        # The code is: piece.x += kick_dx; piece.y += final_kick_dy; then check.
        # It should be: is_valid_position(piece_orig_pos, offset_x=kick_dx, offset_y=final_kick_dy)
        # The current code in board.py:
        #   `if self.is_valid_position(piece, offset_x=kick_dx, offset_y=final_kick_dy):`
        #   `    piece.x += kick_dx`
        #   `    piece.y += final_kick_dy`
        # This is correct. `piece` still holds its original x,y before rotation attempt.

        # Test 2 (-1,0) from table: dx=-1, dy_table=0. final_dy=0.
        #   is_valid_position(piece_at_0,21, offset_x=-1, offset_y=0). Rotated shape.
        #   Piece would be at x=-1. Invalid.
        # Test 3 (-1,+1) from table: dx=-1, dy_table=+1. final_dy=-1.
        #   is_valid_position(piece_at_0,21, offset_x=-1, offset_y=-1).
        #   Piece effective pos: x=-1, y=20. Invalid.
        # Test 4 (0,-2) from table: dx=0, dy_table=-2. final_dy=+2.
        #   is_valid_position(piece_at_0,21, offset_x=0, offset_y=+2).
        #   Piece effective pos: x=0, y=23. Rot1 shape.
        #   Blocks: (23,1), (24,1)-invalid, (24,2)-invalid, (25,1)-invalid.
        # This implies my y calculations or understanding of kick effects is off.
        # The kick (0,-2) means piece moves "down 2" by table's y-axis (which is "up" for game world).
        # So final_dy = -(-2) = +2. Piece moves DOWN by 2 on board.
        # Original y=21. New y = 21+2 = 23.
        # Rot1 blocks: (23,1), (23+1,1)=(24,1), (23+1,2)=(24,2), (23+2,1)=(25,1). All out of bounds (max y_idx=23).

        # Let's re-verify kick data application:
        # JLSTZ_WALL_KICKS[(0,1)] = [(0,0), (-1,0), (-1,+1), (0,-2), (-1,-2)] (dx, dy_world_up)
        # piece.rotate_piece does: final_kick_dy = -kick_dy_table
        # So, for (-1,+1): dx=-1, final_kick_dy = -1. (move left 1, move up 1 on board)
        # For (0,-2): dx=0, final_kick_dy = +2. (no horizontal, move down 2 on board)

        # Back to test: T at x=0, y=21. grid[21,1] is blocked.
        # Kick (0,0) -> dx=0, final_dy=0. Pos (0,21). Fails due to grid[21,1].
        # Kick (-1,0) -> dx=-1, final_dy=0. Pos (-1,21). Fails (out of bounds).
        # Kick (-1,+1) -> dx=-1, final_dy=-1. Pos (-1,20). Fails (out of bounds).
        # Kick (0,-2) -> dx=0, final_dy=+2. Pos (0,23). Rot1 shape.
        #   Blocks: (23,1), (23+1,1)=(24,1), (23+1,2)=(24,2), (23+2,1)=(25,1). Fails (OOB y).
        # Kick (-1,-2) -> dx=-1, final_dy=+2. Pos (-1,23). Fails (OOB x).

        # This implies that for T at (0,21) with grid[21,1] blocked, rotation 0->1 is impossible.
        self.assertFalse(self.board.rotate_piece(t_piece, 1))
        self.assertEqual(t_piece.rotation_state, 0) # Should revert
        self.assertEqual(t_piece.x, 0)
        self.assertEqual(t_piece.y, 21)

        # Try a kick that should work: T-spin setup
        #  XX
        # XTX  <- T piece, X are blocks, . is empty
        #  X.
        # Board state:
        # ..... (y=19)
        # .BBB. (y=20) (B=block)
        # .B.B. (y=21)
        # .BBB. (y=22)
        # Place T at (x=2, y=20), rot0: [[0,T,0],[T,T,T],[0,0,0]]
        # Blocks: (20,3), (21,2), (21,3), (21,4)
        # Target: rotate to rot3 (CCW): [[0,T,0],[.TT,0],[0,T,0]] (y-axis mirrored from wiki)
        # My rot3: [[0,1,0],[1,1,0],[0,1,0]] (from pieces.py)
        # If T is at (x=2,y=21) and we rotate 0->3 (CCW)
        # Kicks (0,3) for JLSTZ: (0,0), (+1,0), (+1,+1), (0,-2), (+1,-2)
        # Setup:
        self.setUp()
        self.board.grid[20,1]=1; self.board.grid[20,2]=1; self.board.grid[20,3]=1; # Top overhang
        self.board.grid[21,1]=1; self.board.grid[21,3]=1; # Side supports
        # T piece at x=2, y=20 (rot0)
        t_piece = self._get_piece_by_name('T')
        self.board.spawn_piece(t_piece)
        t_piece.x = 2; t_piece.y = 20; t_piece.rotation_state=0; t_piece.shape = t_piece.rotations[0]
        # Check it's valid: (20,3), (21,2), (21,3), (21,4). (21,3) collides.
        # So this initial position is bad. T needs to be at y=21 for this setup.
        # T at x=2, y=21 (rot0): Blocks (21,3), (22,2), (22,3), (22,4)
        t_piece.y = 21
        self.assertTrue(self.board.is_valid_position(t_piece))

        # Now rotate CCW (0->3). Expect kick (+1,0) from table. (dx=+1, final_dy=0)
        # Kick (0,0): Pos(2,21) rot3. Blocks: (21,3), (22,2), (22,3), (23,3). (21,3) collides. Fails.
        # Kick (+1,0): Pos(3,21) rot3. Blocks: (21,4), (22,3), (22,4), (23,4). Valid.
        self.assertTrue(self.board.rotate_piece(t_piece, -1)) # CCW
        self.assertEqual(t_piece.rotation_state, 3)
        self.assertEqual(t_piece.x, 3) # Kicked right by 1
        self.assertEqual(t_piece.y, 21)

        # Check T-Spin conditions were met (simplified check)
        self.assertTrue(self.board.last_move_was_rotate)
        # Center of T at (3,21) rot3 [[0,1,0],[1,1,0],[0,1,0]] is (y+1, x+0) or (y+1,x+1)?
        # Bounding box of T is 3x3. Center is (piece.y+1, piece.x+1)
        # T is at (3,22). Corners: (3-1,22-1)=(2,21) (3-1,22+1)=(2,23) (3+1,22-1)=(4,21) (3+1,22+1)=(4,23)
        # Original setup: grid[20,1], grid[20,2], grid[20,3] are blocks.
        # grid[21,1], grid[21,3] are blocks.
        # Piece is now at (x=3, y=21), rot3. Center (22,4).
        # Corners: (21,3)-is block, (21,5)-empty, (23,3)-empty, (23,5)-empty
        # This is not 3 corners. The _check_t_spin_conditions needs careful validation.
        # My corner check is based on the piece's (x,y) and its fixed center [1,1] in its matrix.
        # T is at (x=3,y=21). Center on board is (y+1, x+1) = (22,4).
        # Corners: (21,3)-is block, (21,5)-empty, (23,3)-empty, (23,5)-empty. Only 1 corner.
        # The example setup was probably for a different rotation or expectation.
        # The t-spin corner check:
        # center_r, center_c = t_piece.y + 1, t_piece.x + 1
        # corners = [(center_r-1,center_c-1), (center_r-1,center_c+1), etc.]
        # If t_piece is at (3,21), center is (22,4).
        # A=(21,3) -> grid[21,3]=1. Occupied.
        # B=(21,5) -> grid[21,5]=0. Empty.
        # C=(23,3) -> grid[23,3]=0. Empty. (Assuming board extends here)
        # D=(23,5) -> grid[23,5]=0. Empty.
        # So, self.board.t_spin_corners_occupied would be 1. This is not a T-spin by 3-corner rule.

        # The example from Tetris Wiki for TST kick usually involves piece ending up pointing down.
        # This test primarily checks if kicks are attempted and piece moves. Detailed T-spin logic is separate.

if __name__ == '__main__':
    unittest.main()
