# DeepTrix-Z - A Competitive Tetris RL Bot
# board.py - Manages the Tetris board state, piece locking, line clearing, and collisions.

import numpy as np
from .pieces import Piece, TETROMINOES, I_WALL_KICKS, JLSTZ_WALL_KICKS
from .exceptions import CollisionException, GameOverException

class Board:
    """
    Represents the Tetris game board.
    The board is represented as a 2D numpy array where 0 is empty
    and other numbers might represent colors or piece types of locked blocks.
    The standard Tetris board is 10 columns wide and 20 rows tall (visible area).
    An additional buffer area (e.g., 20 rows) is often used above the visible area
    to allow pieces to spawn and rotate before entering the visible playfield.
    So, a common internal representation is 10x40.
    """
    def __init__(self, width: int = 10, height: int = 20, hidden_rows: int = 20):
        self.width = width
        self.height = height  # Visible height
        self.total_height = height + hidden_rows  # Total height including buffer for spawning

        # Grid: 0 for empty, 1-7 for piece colors (or other representation)
        # Using a simple 1 for filled, 0 for empty for now.
        # Later, this can store color info or piece IDs.
        self.grid = np.zeros((self.total_height, self.width), dtype=int)

        self.spawn_pos_x = self.width // 2 - 2 # Centered, accounting for 4x4 piece bounding box
        self.spawn_pos_y = hidden_rows - 2      # Spawn in the hidden area

        # T-Spin related state
        self.last_move_was_rotate = False
        self.t_spin_corners_occupied = 0 # Number of corners occupied for T-piece before lock

    def is_valid_position(self, piece: Piece, offset_x: int = 0, offset_y: int = 0) -> bool:
        """
        Checks if the piece at its current position + offset is within bounds and not colliding.
        """
        current_x = piece.x + offset_x
        current_y = piece.y + offset_y

        for r_idx, row in enumerate(piece.shape):
            for c_idx, cell in enumerate(row):
                if cell == 1:  # If it's a block of the piece
                    board_r, board_c = current_y + r_idx, current_x + c_idx

                    # Check bounds
                    if not (0 <= board_c < self.width and 0 <= board_r < self.total_height):
                        return False
                    # Check collision with existing blocks on the board
                    if self.grid[board_r, board_c] != 0:
                        return False
        return True

    def _try_spawn_piece(self, piece: Piece) -> bool:
        """
        Tries to place the piece at the standard spawn location.
        Adjusts spawn x for O and I pieces if needed based on common Tetris guidelines.
        """
        piece.x = self.width // 2 - (piece.get_bounding_box_size()[1] // 2)
        piece.y = self.spawn_pos_y

        # Adjust for I piece which typically spawns straddling center line if possible
        if piece.name == 'I':
             piece.x = self.width // 2 - 2 # I is 4 wide
        elif piece.name == 'O':
             piece.x = self.width // 2 - 1 # O is 2 wide

        # If spawn location is obstructed, it's a game over (lock out / block out)
        if not self.is_valid_position(piece):
            # Try to nudge up one or two rows if initial spawn fails (for pieces like I)
            for dy in [-1, -2]:
                if self.is_valid_position(piece, offset_y=dy):
                    piece.y += dy
                    return True
            return False
        return True

    def spawn_piece(self, piece: Piece) -> bool:
        """
        Sets the piece's initial position on the board.
        Returns False if piece cannot be spawned (Game Over).
        """
        piece.rotation_state = 0 # Reset rotation
        piece.shape = piece.rotations[piece.rotation_state]

        if not self._try_spawn_piece(piece):
             # Check if any part of the piece is above the visible area after failing to spawn.
             # If so, it's a "block out" game over.
            for r_idx, row_val in enumerate(piece.shape):
                for c_idx, cell_val in enumerate(row_val):
                    if cell_val == 1:
                        # If y + r_idx is less than hidden_rows, it means part of piece is in visible area
                        if piece.y + r_idx < self.total_height - self.height: # self.hidden_rows
                             pass # This is fine, means piece is in hidden area
                        # If piece is in visible area and obstructed, it's game over
                        # This check is effectively done by _try_spawn_piece's is_valid_position

            # If _try_spawn_piece fails, it means the spawn area is blocked.
            # A more precise game over check: if any cell in the spawn area for the piece
            # (specifically cells that would be occupied by the piece) is already filled
            # AND is within the visible part of the board or just above it, it's game over.
            # For simplicity here, if _try_spawn_piece fails, we declare game over.
            raise GameOverException("Cannot spawn piece in designated area.")

        self.last_move_was_rotate = False
        return True


    def move_piece(self, piece: Piece, dx: int, dy: int) -> bool:
        """
        Moves the piece by dx, dy if the new position is valid.
        Returns True if successful, False otherwise (collision).
        """
        if self.is_valid_position(piece, offset_x=dx, offset_y=dy):
            piece.x += dx
            piece.y += dy
            self.last_move_was_rotate = False
            return True
        return False

    def rotate_piece(self, piece: Piece, direction: int) -> bool:
        """
        Rotates the piece using SRS wall kicks.
        :param direction: 1 for clockwise, -1 for counter-clockwise.
        Returns True if successful, False otherwise.
        """
        original_rotation_state = piece.rotation_state
        original_x, original_y = piece.x, piece.y

        piece.rotate(direction) # Tentatively rotate

        kick_table = I_WALL_KICKS if piece.name == 'I' else JLSTZ_WALL_KICKS
        target_rotation_state = piece.rotation_state # This is new state after piece.rotate()

        kick_key = (original_rotation_state, target_rotation_state)

        if kick_key not in kick_table: # Should not happen with proper direction
            # This can happen if direction is not +/- 1 or logic is flawed.
            # Revert rotation if kick data is missing (defensive).
            piece.rotation_state = original_rotation_state
            piece.shape = piece.rotations[piece.rotation_state]
            return False

        for kick_idx, (kick_dx, kick_dy) in enumerate(kick_table[kick_key]):
            # Note: SRS wall kick data is often (x,y) but here we use (col_offset, row_offset)
            # Standard SRS y-axis is inverted compared to typical array y-axis.
            # So, a kick_dy of +1 in SRS table (up) means -1 in our array y.
            # Let's assume kick_data is (delta_col, delta_row_board_coords)
            # If kick data is from Tetris Wiki: (x,y) where y positive is "up"
            # then we need to invert y for our array: test_y = original_y - kick_dy
            # However, common implementations store kicks as (dx, -dy_wiki)
            # Let's assume the provided kick data is already adjusted for array coordinates,
            # or that positive dy means "down" in the kick table.
            # The provided tables seem to use +y as "up" (e.g. (-1,+1) for 0->1 JLSTZ)
            # So we need to flip the y-offset from the table.

            # Let's test with kick_dy as is. If it's wrong, we'll flip it.
            # The piece coordinates (x,y) are top-left.
            # A kick like (-1, +1) for JLSTZ (0->1), Test 2: (-1,0), Test 3: (-1,1)
            # Test 3: (-1, +1) means move left 1, up 1. In our array, y increases downwards.
            # So if table says (+1 y), it means piece.y should decrease.
            # So, test_x = original_x + kick_dx, test_y = original_y - kick_dy (if y is "up" in table)
            # The tables are (col_offset, row_offset_game_coords) where row_offset_game_coords is "up"

            # Let's assume kick_dy is for array coords (positive = down)
            # If using wiki data directly: (0,0), (-1,0), (-1,+1), (0,-2), (-1,-2) for JLSTZ 0->1
            # Our piece.y increases downwards. Wiki +y is up. So wiki +1y is our -1y.

            # Test 1: (0,0) -> piece.x = original_x + 0, piece.y = original_y + 0
            # Test 2: (-1,0) -> piece.x = original_x - 1, piece.y = original_y + 0
            # Test 3: (-1,+1) from table -> piece.x = original_x - 1, piece.y = original_y - 1 (move up)

            # The code currently in `pieces.py` implies kick_dy is positive for up.
            # Let's use the table offsets directly for now and verify with tests.
            # The crucial part is consistency. The (0,0) test is always first.
            # If the kick data is for (delta_x, delta_y_game_world) where positive y is UP:
            # test_piece_x = original_x + kick_dx
            # test_piece_y = original_y - kick_dy
            # If the kick data is for (delta_x, delta_y_array_coords) where positive y is DOWN:
            # test_piece_x = original_x + kick_dx
            # test_piece_y = original_y + kick_dy

            # Assuming kick_dy in tables is for "game world up" (decrease in array y)
            final_kick_dy = -kick_dy

            if self.is_valid_position(piece, offset_x=kick_dx, offset_y=final_kick_dy):
                piece.x += kick_dx
                piece.y += final_kick_dy

                # Check for T-Spin specific conditions IF the piece is a T-piece
                if piece.name == 'T':
                    self._check_t_spin_conditions(piece, original_x, original_y, kick_idx +1) # kick_idx is 0-4 for 5 tests

                self.last_move_was_rotate = True
                return True

        # If no kick works, revert rotation and position
        piece.x, piece.y = original_x, original_y
        piece.rotation_state = original_rotation_state
        piece.shape = piece.rotations[piece.rotation_state]
        return False

    def _check_t_spin_conditions(self, t_piece: Piece, prev_x: int, prev_y: int, kick_test_used: int):
        """
        Checks conditions for a T-Spin. This is a simplified check.
        A common definition:
        1. Last move was a rotation.
        2. The T-piece is locked.
        3. Three of the four corners around the T's center are occupied by blocks or walls.
           The center of a T-piece (3x3 bounding box) is at (x+1, y+1) for its shape matrix.
           When T is [[0,1,0], [1,1,1], [0,0,0]], center is shape[1][1].
           Board coordinates: (t_piece.y + 1, t_piece.x + 1)
        4. For "mini" T-spin vs "full" T-spin, one of the two front corners must be occupied.
           Also, if the rotation used specific wall kicks (like T-Spin Triple kicks), it can influence this.

        This method is called *after* a successful rotation with kicks.
        It should store information that will be used when the piece is locked.
        """
        if not self.last_move_was_rotate: # Should be true if called from rotate_piece
            self.t_spin_corners_occupied = 0
            return

        # T-piece center relative to its own bounding box [1,1]
        # T-piece center on board: (piece.y + 1, piece.x + 1)
        center_r, center_c = t_piece.y + 1, t_piece.x + 1

        # Corners relative to T-piece center:
        # A: top-left (-1,-1) -> (center_r - 1, center_c - 1)
        # B: top-right (-1,+1) -> (center_r - 1, center_c + 1)
        # C: bottom-left (+1,-1) -> (center_r + 1, center_c - 1)
        # D: bottom-right (+1,+1) -> (center_r + 1, center_c + 1)
        corners = [
            (center_r - 1, center_c - 1), (center_r - 1, center_c + 1),
            (center_r + 1, center_c - 1), (center_r + 1, center_c + 1)
        ]

        occupied_count = 0
        for r, c in corners:
            if not (0 <= c < self.width and 0 <= r < self.total_height): # Wall is occupied
                occupied_count += 1
            elif self.grid[r, c] != 0: # Existing block is occupied
                occupied_count += 1

        self.t_spin_corners_occupied = occupied_count

        # More detailed T-spin logic (e.g. mini vs full) can be added here or at lock time.
        # For example, checking which specific corners are filled (front vs back relative to T orientation)
        # Also, "immobile" T-Spin check (cannot move after rotation without rotating back)
        # This is a complex part of modern Tetris rules. For now, 3+ corners is a good start.


    def lock_piece(self, piece: Piece) -> tuple[int, bool, bool]:
        """
        Locks the current piece onto the board.
        Updates the grid.
        Returns (lines_cleared, is_t_spin, is_pc).
        is_pc = Perfect Clear
        """
        is_t_spin_clear = False
        is_mini_t_spin = False # Placeholder

        for r_idx, row in enumerate(piece.shape):
            for c_idx, cell in enumerate(row):
                if cell == 1:
                    board_r, board_c = piece.y + r_idx, piece.x + c_idx
                    # Check for top out (locking a piece that is even partially above visible area)
                    # The check for GameOverException during spawn should handle most "block out" scenarios.
                    # A "lock out" happens if a piece locks entirely above the visible screen,
                    # or partially above in a way that makes future spawns impossible.
                    if board_r < (self.total_height - self.height): # hidden_rows
                        # This means part of the piece locked in the hidden buffer zone.
                        # If any *visible* part of the board is obstructed by this for the next spawn,
                        # it's effectively game over.
                        # A simpler check: if any part of the piece locks above row `hidden_rows - some_threshold`
                        # (e.g. hidden_rows - 2), it could be problematic.
                        # For now, GameOverException in spawn_piece is the main game over trigger.
                        pass # Allow locking in buffer, spawn handles game over

                    if not (0 <= board_r < self.total_height and 0 <= board_c < self.width):
                        # This should ideally not happen if moves are validated. Defensive.
                        # Could indicate an issue with rotation or movement logic allowing out-of-bounds lock.
                        # Consider raising an error or specific handling.
                        # For now, clip to board if this occurs, though it's a symptom of a bug.
                        # print(f"Warning: Piece locked partially out of bounds at ({board_r},{board_c}). Clipping.")
                        # board_r = max(0, min(board_r, self.total_height - 1))
                        # board_c = max(0, min(board_c, self.width - 1))
                        # This situation signifies a critical error if piece is outside width or total_height.
                        # If piece.y + r_idx is negative, it's a major issue.
                        continue # Skip if somehow a block is out of bounds

                    # Use a simple 1 for filled, or piece.color or piece_id later
                    self.grid[board_r, board_c] = piece.name_to_id(piece.name) # Or just 1, or color

        # T-Spin check after lock
        if piece.name == 'T' and self.last_move_was_rotate and self.t_spin_corners_occupied >= 3:
            # Check if it resulted in a line clear for it to be a "T-Spin Clear"
            # This is a simplified T-Spin check. Full rules are more nuanced (e.g. mini, TST)
            is_t_spin_clear = True # Assume any such lock is a T-Spin for now if it clears lines
            # More precise: check "facing" corners based on T's orientation
            # For example, for T pointing down (initial spawn), corners A and B are "front".
            # If one of these is filled, it can be a T-Spin. If both, and kick #5 was used, also T-Spin.

        lines_cleared = self._clear_lines()

        if is_t_spin_clear and lines_cleared == 0: # T-Spin, but no lines cleared (e.g. T-Spin Mini setup)
            is_t_spin_clear = False # It's a T-Spin setup, but not a "T-Spin X" clear.
                                    # Could be a "T-Spin Mini No Lines"
            # Some rules grant points for T-Spin Mini even with 0 lines.
            # For our reward, we'll differentiate T-Spin Single, Double, Triple.

        # Reset T-spin state for next piece
        self.last_move_was_rotate = False
        self.t_spin_corners_occupied = 0

        is_pc = self._check_perfect_clear()

        # The `is_t_spin_clear` here means "a T-spin action that contributed to line clears".
        # The reward function will use lines_cleared and this flag.
        return lines_cleared, is_t_spin_clear, is_pc

    def _clear_lines(self) -> int:
        """
        Checks for and clears completed lines.
        Shifts lines down. Returns number of lines cleared.
        """
        lines_cleared = 0
        rows_to_clear = []

        # Iterate from bottom to top of the visible board area
        for r in range(self.total_height - 1, self.total_height - self.height - 1, -1):
            if r < 0: continue # Should not happen with standard board setup

            is_line_full = np.all(self.grid[r, :] != 0)
            if is_line_full:
                rows_to_clear.append(r)
                lines_cleared += 1

        if lines_cleared > 0:
            # Remove lines from top of list to maintain indices for np.delete
            rows_to_clear.sort(reverse=False)
            for row_idx in rows_to_clear:
                self.grid = np.delete(self.grid, row_idx, axis=0)

            # Add new empty lines at the top (within the total_height buffer)
            new_lines = np.zeros((lines_cleared, self.width), dtype=int)
            self.grid = np.vstack((new_lines, self.grid))

        return lines_cleared

    def _check_perfect_clear(self) -> bool:
        """
        Checks if the board is empty (Perfect Clear).
        Only checks the visible part of the board.
        """
        visible_board = self.grid[self.total_height - self.height :, :]
        return np.all(visible_board == 0)

    def get_board_state(self, include_hidden_rows=False) -> np.ndarray:
        """
        Returns the current grid state.
        By default, returns only the visible part of the board.
        """
        if include_hidden_rows:
            return self.grid.copy()
        else:
            return self.grid[self.total_height - self.height :, :].copy()

    @staticmethod
    def name_to_id(name: str) -> int:
        """Converts piece name to a numerical ID for the grid."""
        # Simple mapping, can be expanded or use colors.
        # 0 is empty. So IDs start from 1.
        return list(TETROMINOES.keys()).index(name) + 1

    @staticmethod
    def id_to_name(id_val: int) -> str | None:
        """Converts numerical ID back to piece name."""
        if id_val == 0: return None # Empty
        try:
            return list(TETROMINOES.keys())[id_val - 1]
        except IndexError:
            return None # Unknown ID


if __name__ == '__main__':
    board = Board(width=10, height=20, hidden_rows=4) # Smaller hidden for easier print
    print("Initial board (visible portion):")
    print(board.get_board_state())
    print(f"Total grid shape: {board.grid.shape}")

    # Example: Manually add some blocks
    board.grid[board.total_height - 1, :] = 1 # Fill bottom line
    board.grid[board.total_height - 2, :5] = 2 # Fill half of second to bottom line
    print("\nBoard after manual additions:")
    print(board.get_board_state())

    lines = board._clear_lines()
    print(f"\nCleared {lines} lines.")
    print("Board after clearing lines:")
    print(board.get_board_state())
    print(f"Is Perfect Clear? {board._check_perfect_clear()}")

    # Test spawning
    from .pieces import PieceFactory
    factory = PieceFactory()

    print("\nAttempting to spawn pieces...")
    for i in range(2):
        current_piece = factory.next_piece()
        print(f"Spawning {current_piece.name}")
        try:
            board.spawn_piece(current_piece)
            print(f"Spawned {current_piece.name} at ({current_piece.x}, {current_piece.y})")
            # Simulate locking it immediately for test
            # board.lock_piece(current_piece)
        except GameOverException as e:
            print(f"Game Over: {e}")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    # Test T-spin conditions (simplified)
    # Setup: Place a T piece, rotate it into a T-slot
    # Requires a board setup where T-spin is possible.
    # This part is harder to test in isolation without full game loop.

    # Test perfect clear
    board.grid[:, :] = 0 # Clear board completely
    print(f"\nBoard cleared. Is Perfect Clear? {board._check_perfect_clear()}")
    board.grid[board.total_height - 1, 0] = 1 # Add one block
    print(f"One block added. Is Perfect Clear? {board._check_perfect_clear()}")

```
