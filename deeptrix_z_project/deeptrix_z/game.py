# DeepTrix-Z - A Competitive Tetris RL Bot
# game.py - The main Tetris game environment compatible with OpenAI Gym.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from .board import Board
from .pieces import PieceFactory, Piece
from .exceptions import GameOverException, CollisionException

class TetrisEnv(gym.Env):
    """
    A Tetris environment that conforms to the OpenAI Gym API.
    It simulates modern Tetris rules including 7-bag, SRS, hold,
    line clears, B2B, combos, and garbage.

    Action Space:
    The action space needs to be carefully defined. A common approach for RL in Tetris
    is to define actions as (target_x_position, target_rotation_state, use_hold_piece).
    The environment would then try to place the piece at that (x, rotation) using a
    sequence of moves (left, right, rotate, hard_drop).
    Alternatively, a more granular action space could be:
    - 0: Move Left
    - 1: Move Right
    - 2: Rotate Clockwise
    - 3: Rotate Counter-Clockwise
    - 4: Soft Drop
    - 5: Hard Drop
    - 6: Hold Piece
    - 7: Do Nothing (let piece fall by one step due to gravity)

    For a high-level bot, predicting the final placement is often more effective.
    Let's start with a discrete action space representing all possible final placements
    for the current piece. This can be complex due to SRS.
    A simpler discrete space: (10 columns * 4 rotations) + 1 for hold. Max ~41 actions.
    The agent picks one of these, and the environment executes it (places piece if valid).

    Observation Space:
    - Board matrix (e.g., 10x20 or 10x40 binary or integer representation)
    - Current active piece (ID or one-hot encoded)
    - Hold piece (ID or one-hot encoded)
    - Next N pieces (IDs or one-hot encoded)
    - B2B state (integer)
    - Combo count (integer)
    - Incoming garbage lines (integer)
    - (Optional) Opponent board representation

    For now, a simplified observation:
    - Flattened board (10x20 visible part)
    - Current piece ID
    - Hold piece ID
    - Next piece ID
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, width=10, height=20, hidden_rows=20, num_next_pieces=5, manual_action_override=False):
        super(TetrisEnv, self).__init__()

        self.board_width = width
        self.board_height = height # Visible height
        self.hidden_rows = hidden_rows
        self.total_board_height = height + hidden_rows

        self.board = Board(width, height, hidden_rows)
        self.piece_factory = PieceFactory()
        self.num_next_pieces = num_next_pieces
        self.manual_action_override = manual_action_override # If true, step() expects specific piece placement

        self.current_piece: Piece | None = None
        self.hold_piece: Piece | None = None
        self.can_hold_this_turn = True # Can only hold once per piece

        self.next_pieces_queue = []
        self._fill_next_pieces_queue()

        self.score = 0
        self.lines_cleared_total = 0
        self.level = 1 # Could affect gravity later
        self.game_over = False

        # Game state for rewards and advanced logic
        self.b2b_state = 0  # Back-to-Back chain length
        self.combo_count = 0 # Combo chain length (clearing lines in consecutive turns)
        self.incoming_garbage = 0 # Number of pending garbage lines to receive
        self.last_clear_was_difficult = False # True if last clear was Tetris or T-Spin

        # Action Space Definition:
        # For now, let's define a discrete number of "expert actions" or "placements".
        # An action would be an index representing (target_column, target_rotation, use_hold).
        # There are 10 columns, 4 rotations. So 40 base placements.
        # Add one for hold.
        # This means the agent chooses *where* a piece should end up.
        # The environment then needs to simulate the moves to get it there.
        # This is a common approach (e.g., MisaMino pathfinder).

        # Simpler action space for now: (chosen_x, chosen_rotation_idx)
        # Max 10 columns * 4 rotations = 40 actions. Add hold action.
        # This still requires a pathfinder or assumes direct placement.

        # Let's use a more direct, if large, action space:
        # For each piece, all possible (x, rotation) states it can be in when it locks.
        # This can be very large.
        # Alternative: Action = index from a list of pre-calculated valid placements.
        # This is what many high-level bots do. The "action" is selecting one such placement.

        # For stable-baselines3, a simple Discrete or MultiDiscrete space is easiest to start.
        # Let's use: For each of (10 cols * 4 rotations for current piece) + 1 for HOLD.
        # Total actions = 10 * 4 + 1 = 41.
        # The environment will try to move the piece to (col, rot) and hard drop.
        # If `manual_action_override` is True, action is (piece_x, piece_y, piece_rot_state, piece_id_to_confirm)
        if not self.manual_action_override:
            self.num_rotations = 4
            self.action_space_size = self.board_width * self.num_rotations + 1 # +1 for hold
            self.action_space = spaces.Discrete(self.action_space_size)
        else:
            # For manual testing: action is a tuple (x, y, rotation_idx, piece_name)
            # This isn't a standard Gym action space, used for direct control for now.
            pass # No standard gym space if manual override

        # Observation Space Definition:
        # For "CnnPolicy" of stable-baselines3, image-like input is good.
        # (channels, height, width)
        # Channel 1: Current board (binary: occupied/empty)
        # Channel 2: Current piece projection (where it would land with hard drop)
        # Channel 3: Hold piece (represented on a small grid or as features)
        # Channel 4: Next pieces (represented on small grids or as features)
        # Additional features: B2B, combo, garbage (can be concatenated vector)

        # Simplified observation for now:
        # Flattened visible board (10x20 = 200)
        # Current piece ID (1)
        # Hold piece ID (1, 0 if none)
        # Next piece ID (1)
        # B2B state (1)
        # Combo count (1)
        # Incoming garbage (1)
        # Total features = 200 + 1 + 1 + 1 + 1 + 1 + 1 = 206
        # This is a Box space.

        # Let's try a Dict space for better organization, and CNNs can handle parts of it.
        # For CNN, board should be HxWxC or CxHxW. StableBaselines3 CnnPolicy expects CxHxW.
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=7, shape=(1, self.total_board_height, self.board_width), dtype=np.uint8), # Max 7 piece types + empty
            "current_piece_id": spaces.Discrete(len(self.piece_factory.bag) + 1), # Piece types + 1 for none
            "hold_piece_id": spaces.Discrete(len(self.piece_factory.bag) + 1), # 0 if no hold piece
            "next_piece_ids": spaces.Box(low=0, high=len(self.piece_factory.bag), shape=(self.num_next_pieces,), dtype=np.int8),
            "b2b_state": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int16),
            "combo_count": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int16),
            "incoming_garbage": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int16),
            "ghost_piece_map": spaces.Box(low=0, high=1, shape=(1, self.total_board_height, self.board_width), dtype=np.uint8)
        })

        self._spawn_new_piece()

    def _get_ghost_piece_positions(self) -> list[tuple[int, int]]:
        """Calculates the board positions for the ghost piece."""
        if not self.current_piece:
            return []

        # Create a temporary piece to simulate the drop
        ghost = Piece(self.current_piece.name, self.current_piece.color, self.current_piece.rotations)
        ghost.x = self.current_piece.x
        ghost.y = self.current_piece.y
        ghost.rotation_state = self.current_piece.rotation_state
        ghost.shape = self.current_piece.shape # Ensure shape is consistent with rotation state

        # Drop the ghost piece
        while self.board.is_valid_position(ghost, offset_y=1):
            ghost.y += 1

        return ghost.get_block_positions()

    def _fill_next_pieces_queue(self):
        while len(self.next_pieces_queue) < self.num_next_pieces:
            self.next_pieces_queue.append(self.piece_factory.next_piece())

    def _get_piece_id(self, piece: Piece | None) -> int:
        if piece is None:
            return 0 # 0 for no piece
        return self.board.name_to_id(piece.name) # 1-7 for piece types

    def _get_observation(self):
        board_state = self.board.grid.astype(np.uint8).reshape((1, self.total_board_height, self.board_width))

        ghost_piece_map_np = np.zeros_like(self.board.grid, dtype=np.uint8)
        ghost_positions = self._get_ghost_piece_positions()
        for r, c in ghost_positions:
            if 0 <= r < self.total_board_height and 0 <= c < self.board_width:
                ghost_piece_map_np[r, c] = 1
        ghost_piece_map_reshaped = ghost_piece_map_np.reshape((1, self.total_board_height, self.board_width))

        obs = {
            "board": board_state,
            "current_piece_id": self._get_piece_id(self.current_piece),
            "hold_piece_id": self._get_piece_id(self.hold_piece),
            "next_piece_ids": np.array([self._get_piece_id(p) for p in self.next_pieces_queue[:self.num_next_pieces]], dtype=np.int8),
            "b2b_state": np.array([self.b2b_state], dtype=np.int16),
            "combo_count": np.array([self.combo_count], dtype=np.int16),
            "incoming_garbage": np.array([self.incoming_garbage], dtype=np.int16),
            "ghost_piece_map": ghost_piece_map_reshaped
        }
        return obs

    def _spawn_new_piece(self) -> bool:
        if not self.next_pieces_queue: # Should not happen with _fill_next_pieces_queue
            self._fill_next_pieces_queue()

        self.current_piece = self.next_pieces_queue.pop(0)
        self._fill_next_pieces_queue()
        self.can_hold_this_turn = True

        try:
            self.board.spawn_piece(self.current_piece)
            return True
        except GameOverException:
            self.game_over = True
            return False

    def _handle_action(self, action: int) -> bool:
        """
        Interprets the discrete action and attempts to perform it.
        Action `self.action_space_size - 1` is HOLD.
        Other actions are `rot * self.board_width + col`.
        Returns True if piece was placed/held, False if action is invalid for current piece state (rare).
        """
        if self.current_piece is None: # Should not happen if game not over
            return False

        if action == self.action_space_size - 1: # Hold action
            return self._perform_hold()
        else:
            target_rotation = action // self.board_width
            target_col = action % self.board_width

            # This is a simplified placement. A real agent would need a pathfinder.
            # For now, we'll try to set rotation, then move to column, then hard drop.

            # 1. Set rotation (try all kicks)
            current_rot = self.current_piece.rotation_state
            # Calculate rotation direction (shortest path)
            # (target_rotation - current_rot + num_rotations) % num_rotations gives num clockwise steps
            # (current_rot - target_rotation + num_rotations) % num_rotations gives num counter-clockwise steps

            # Simpler: try to set rotation directly if board allows
            original_shape = self.current_piece.shape
            original_rot_state = self.current_piece.rotation_state

            # Try to apply target_rotation. This needs more than just setting state.
            # It needs to find a sequence of rotations (CW/CCW) to reach target_rotation.
            # For now, let's assume we can directly attempt the target_rotation.
            # This means the piece's internal rotation_state must match target_rotation.
            # This is a placeholder for a proper pathfinding/move execution.

            # Let's try to rotate to the target_rotation
            # We need to find a sequence of single rotations
            # For now, just apply the rotation if it's different
            if self.current_piece.rotation_state != target_rotation:
                # This is naive, doesn't account for multiple steps or kick preferences
                # Try to rotate towards target_rotation one step at a time or directly set
                # This is a complex part: how to map agent's choice (target_rot) to game actions.
                # For now, let's assume we can test the final state:

                temp_piece = Piece(self.current_piece.name, self.current_piece.color, self.current_piece.rotations)
                temp_piece.x = self.current_piece.x # Keep current x for now
                temp_piece.y = self.current_piece.y
                temp_piece.rotation_state = target_rotation # Force it
                temp_piece.shape = temp_piece.rotations[target_rotation]

                # Check if this rotated piece can be placed at target_col
                # This is also complex. We need to find a valid y for this (x, rot)
                # by dropping it from the top.

                # The action space (target_col, target_rot) implies the piece *ends up* there.
                # So, we need to simulate placing a piece with `target_rotation` at `target_col`
                # and let it fall.

                # Create a test piece with the desired rotation
                test_piece = Piece(self.current_piece.name, self.current_piece.color, self.current_piece.rotations)
                test_piece.rotation_state = target_rotation
                test_piece.shape = test_piece.rotations[target_rotation]

                # Try to place it at target_col, as high as possible, then drop.
                test_piece.x = target_col
                # Adjust x if piece is too wide for the column (e.g. I piece)
                piece_width = test_piece.get_bounding_box_size()[1]
                if test_piece.x + piece_width > self.board_width:
                    test_piece.x = self.board_width - piece_width

                test_piece.y = self.board.spawn_pos_y # Start high

                if not self.board.is_valid_position(test_piece):
                    # If even starting high is invalid (e.g. target_col makes it go out of bounds immediately)
                    # This action might be impossible.
                    # A robust system would prune such actions from agent's choice.
                    return False # Action leads to invalid state

                # Now, hard drop this test_piece configuration
                while self.board.is_valid_position(test_piece, offset_y=1):
                    test_piece.y += 1

                # test_piece.y is now the lock position for this (target_col, target_rot)
                # We need to apply this to self.current_piece
                # This involves:
                # 1. Rotating self.current_piece to target_rotation (with kicks)
                # 2. Moving self.current_piece to test_piece.x (horizontally)
                # 3. Soft/Hard dropping self.current_piece to test_piece.y

                # This is where a full move sequence generator (A* or similar) would be used.
                # For a gym env, we might have to simplify this.
                # If the action is "place piece with this (rot, col) and hard drop",
                # we assume the low-level controller can do it if it's possible.

                # Simplified: If we can find a valid final spot by dropping from top
                # with target_rotation at target_col, then we "teleport" the piece there.
                # This is common in AI that choose final placements.

                self.current_piece.rotation_state = test_piece.rotation_state
                self.current_piece.shape = test_piece.shape
                self.current_piece.x = test_piece.x
                self.current_piece.y = test_piece.y # This is the final y after dropping

                if not self.board.is_valid_position(self.current_piece):
                    # This chosen (col, rot) is impossible to achieve or results in collision.
                    # This implies the action chosen by agent is bad.
                    # We might penalize this, or the environment should only allow valid final states.
                    # For now, if the chosen high-level action is bad, it's like a null move.
                    # Or, the game could end if agent makes too many invalid choices.
                    # Let's assume for now the agent only picks from valid (col, rot) that lead to a lock.
                    # This means action space should be dynamic or pre-filtered.
                    # For simplicity with stable-baselines, fixed action space is easier.
                    # So, if an action is "bad", we can just not move the piece and it will auto-fall.
                    # Or, let's try to make the action "best effort"
                    # This simplified _handle_action is problematic.
                    # A better approach for fixed action space:
                    # Action X -> try to move current piece to column X, keep current rotation, hard drop.
                    # Action Y -> try to rotate current piece, keep current col, hard drop.
                    # Action Z -> hold.
                    # This is more like direct control.

                    # Let's revert to a more standard granular action space for step-by-step control
                    # if the high-level placement is too complex to implement robustly quickly.
                    # For now, we assume the (target_col, target_rot) implies a hard drop.
                    # The agent is choosing the final resting place.
                    self._perform_hard_drop(piece_to_drop=self.current_piece) # current_piece is now at chosen x, rot, final y
                    return True # Piece was "placed"

            # If no rotation change, just move to target_col and hard_drop
            self.current_piece.x = target_col
            piece_width = self.current_piece.get_bounding_box_size()[1]
            if self.current_piece.x + piece_width > self.board_width:
                self.current_piece.x = self.board_width - piece_width

            # Ensure x is not negative
            if self.current_piece.x < 0: self.current_piece.x = 0

            # Drop from current y or spawn_y if it was just rotated to a new position
            # This logic needs to be cleaner. If rotation happened, y might be high.
            # The core idea: action = (final_x, final_rotation), then hard_drop.

            if not self.board.is_valid_position(self.current_piece):
                # If moving to target_col itself is invalid (e.g. blocked high up)
                # This action is also problematic.
                # A truly robust system would have the agent choose from a list of
                # *all possible valid final (x, y, rot) states*.
                # For fixed action spaces, this is a known challenge.
                # Let's assume for now that if a placement is bad, we just do nothing and let gravity act.
                # This means we need a "gravity" step if no valid action taken.
                return False # Cannot achieve this state.

            self._perform_hard_drop(piece_to_drop=self.current_piece)
            return True


    def _perform_hold(self) -> bool:
        if not self.can_hold_this_turn:
            return False # Already held this turn or piece just came from hold

        if self.hold_piece is None:
            self.hold_piece = self.current_piece
            self.hold_piece.rotation_state = 0 # Reset rotation of piece going to hold
            self.hold_piece.shape = self.hold_piece.rotations[0]
            self._spawn_new_piece()
        else:
            # Swap current piece with hold piece
            temp_piece = self.current_piece
            self.current_piece = self.hold_piece
            self.hold_piece = temp_piece

            self.hold_piece.rotation_state = 0 # Reset rotation of piece going to hold
            self.hold_piece.shape = self.hold_piece.rotations[0]

            # Try to spawn the (previously held) new current_piece
            try:
                self.board.spawn_piece(self.current_piece)
            except GameOverException:
                self.game_over = True
                # If swapping from hold causes game over, original piece (now in hold) might be relevant
                # For now, just game over.

        self.can_hold_this_turn = False # Can't hold again until a piece is locked
        return True

    def _perform_soft_drop(self):
        if self.current_piece and not self.board.move_piece(self.current_piece, 0, 1):
            # Cannot soft drop further, so lock it
            self._lock_current_piece()
        # Add score for soft drop if desired

    def _perform_hard_drop(self, piece_to_drop: Piece | None = None):
        target_piece = piece_to_drop if piece_to_drop else self.current_piece
        if target_piece:
            moved = 0
            while self.board.move_piece(target_piece, 0, 1):
                moved += 1
            # Add score for hard drop based on `moved` distance if desired
            self._lock_current_piece() # Piece is now at its lowest valid position

    def _apply_gravity(self):
        """Applies one step of gravity. If piece cannot move down, it locks."""
        if self.current_piece:
            if not self.board.move_piece(self.current_piece, 0, 1):
                self._lock_current_piece()

    def _lock_current_piece(self):
        if not self.current_piece: return

        lines_cleared, is_t_spin, is_pc = self.board.lock_piece(self.current_piece)

        # Calculate reward based on lines_cleared, t_spin, pc, etc.
        # This will be done in the step() method. Store these results for reward calculation.
        self.last_lock_info = {
            "lines_cleared": lines_cleared,
            "is_t_spin": is_t_spin, # This means a T-piece was locked with 3+ corners + last move rotate
            "is_pc": is_pc,
            "piece_locked": self.current_piece.name
        }

        self.current_piece = None # Piece is now part of the board

        # Handle line clear scoring and game state updates (B2B, Combo)
        if lines_cleared > 0:
            self.lines_cleared_total += lines_cleared
            self.combo_count += 1

            # Determine if this clear was "difficult" (Tetris or T-Spin)
            is_difficult_clear = (lines_cleared == 4) or \
                                 (is_t_spin and lines_cleared > 0) # T-Spin Single/Double/Triple

            if is_difficult_clear:
                if self.last_clear_was_difficult: # Check b2b_state before resetting
                    self.b2b_state += 1
                else:
                    self.b2b_state = 1 # Start B2B chain
                self.last_clear_was_difficult = True
            else: # Easy clear (single, double, triple non-Tspin)
                self.b2b_state = 0
                self.last_clear_was_difficult = False
        else: # No lines cleared
            self.combo_count = 0 # Reset combo
            # B2B state persists if no lines cleared, only resets on non-difficult clear or game end

        # Spawn next piece if game not over
        if not self.game_over: # Game over might be set by board.lock_piece if it tops out
            if not self._spawn_new_piece(): # This sets self.game_over if spawn fails
                pass # Game is over

        # If there's incoming garbage, and lines were cleared, cancel garbage
        # Then, if lines cleared > garbage cancelled, send counter attack
        # This part needs opponent interaction or simulated garbage queue.
        # For now, just focus on self-play mechanics.

    def get_possible_placements(self, piece: Piece) -> list[tuple[int, int, int]]:
        """
        Calculates all possible (x, y, rotation_state) where the piece can be locked.
        This is a complex but crucial function for many Tetris AIs.
        Returns a list of (final_x, final_y, final_rotation_state).
        This involves simulating rotations and movements.
        A full implementation is non-trivial (e.g., similar to MisaMino's pathfinder).
        Placeholder for now.
        """
        # For each rotation state of the piece:
        #   For each column it could be in:
        #     Drop it and see where it lands.
        #     Store (x, y_landed, rotation_state) if valid.
        # This needs to handle SRS kicks for rotations properly during the search.
        return [] # Placeholder

    def step(self, action: int | tuple): # Action can be int (discrete) or tuple (manual)
        """
        Executes one time step within the environment.
        If using the predefined Discrete action space (0-40):
            The action implies a target (column, rotation) for the current piece,
            followed by a hard drop, or a HOLD action.
            The environment attempts to execute this. If the target placement is
            impossible (e.g., blocked), the piece might not move, or a penalty occurs.
            After the action, if the piece is not locked, gravity applies.

        If `manual_action_override` is True:
            `action` is a tuple (target_x, target_y, target_rotation_idx, piece_name_to_confirm)
            This directly places the piece if piece_name matches current_piece.
            This is for debugging or specific scripted play.
        """
        if self.game_over:
            # Should return current state and 0 reward if already game over
            return self._get_observation(), 0, True, False, {"reason": "Game already over"}

        reward = 0.0
        terminated = False # Game ended naturally (top out)
        truncated = False  # Episode ended due to time limit or other external factor (not used here yet)

        self.last_lock_info = {} # Reset info for this step

        if self.manual_action_override:
            # Manual control mode for debugging / specific scenarios
            if not isinstance(action, tuple) or len(action) != 4:
                raise ValueError("Manual action must be (x, y, rotation_idx, piece_name)")

            target_x, target_y, target_rot, piece_name = action
            if self.current_piece and self.current_piece.name == piece_name:
                self.current_piece.x = target_x
                self.current_piece.y = target_y
                if self.current_piece.rotation_state != target_rot:
                    self.current_piece.rotation_state = target_rot
                    self.current_piece.shape = self.current_piece.rotations[target_rot]

                # Check if this manual placement is valid before locking
                if not self.board.is_valid_position(self.current_piece):
                    # This manual placement is invalid, could be game over or just ignore
                    # For testing, let's assume manual placements are "forced" if possible
                    # or it implies an error in the manual command if it's truly invalid.
                    # For now, we'll attempt to lock it. If it causes issues, needs rethink.
                    # print(f"Warning: Manual placement of {piece_name} at ({target_x},{target_y}) rot {target_rot} is invalid on board.")
                    # For simplicity, if manual is bad, let's assume it ends the turn for that piece somehow
                    # This mode is not for RL training, so rules can be looser.
                    # Let's try to lock it. If it's out of bounds, lock_piece might handle it or error.
                     pass # It will be locked by _lock_current_piece below

                self._lock_current_piece() # Lock the manually placed piece
            else:
                # Piece mismatch or no current piece, effectively a null action
                # Gravity would apply if there was a piece.
                # This path needs clarification for manual mode if no piece matches.
                pass # No action taken if piece doesn't match or no piece

        else: # Standard RL action processing
            if self.current_piece:
                action_taken_successfully = self._handle_action(action) # This might lock the piece

                # If _handle_action did not result in a lock (e.g. hold, or invalid placement choice)
                # and there's still a current piece, apply gravity.
                if self.current_piece:
                    self._apply_gravity() # This might also lock the piece
            else:
                # No current piece, but game not over? This implies a piece was just locked
                # and next piece should have spawned. If spawn failed, game_over would be true.
                # This state should ideally not be reached if logic is correct.
                # If we are here, it means a piece was locked, and _spawn_new_piece was called.
                # If that spawn failed, self.game_over is true.
                # If it succeeded, self.current_piece is now populated.
                # So, this 'else' branch for 'if self.current_piece:' should be rare.
                 pass


        # Calculate reward based on self.last_lock_info and other game events
        # This is the core of the reward shaping.
        if self.last_lock_info: # A piece was locked in this step
            lines = self.last_lock_info.get("lines_cleared", 0)
            is_tspin = self.last_lock_info.get("is_t_spin", False)
            is_pc = self.last_lock_info.get("is_pc", False)

            base_clear_reward = 0
            t_spin_reward = 0
            pc_reward = 0
            b2b_bonus = 0
            combo_bonus = 0

            # T-Spin Check and Reward (Overrides base line clear rewards if it's a T-Spin clear)
            if is_tspin and lines > 0: # A T-Spin that cleared lines
                if lines == 1: # T-Spin Single
                    # Differentiate Mini vs Full T-Spin Single if possible. Assume full for now.
                    # Prompt: T-spin mini +1. If this is a TSM 1-liner, it's +1.
                    # For now, any T-Spin Single gets a higher base.
                    # Let's assume `is_tspin` implies a "full" T-Spin for now if lines > 0
                    # and T-Spin Mini is for 0 lines or specific "mini" conditions.
                    # For simplicity, use prompt's TSD/TST values and infer TSM.
                    # A "T-spin mini" for 1 line is typically +1. A "T-spin single" (not mini) is more.
                    # Let's use a single category for T-Spin Single for now.
                    t_spin_reward = 2.0 # Placeholder: A "standard" T-Spin Single value
                                        # Prompt: T-Spin Mini +1, TSD +3, TST +5.
                                        # This implies TSM (1-line) = +1.
                                        # Let's try to implement this:
                                        # Requires better is_mini flag from board.py.
                                        # For now: if is_tspin and lines==1, assume it could be mini or full.
                                        # Let's say any T-Spin 1-line is +1 for now (treat as TSM 1-line)
                    t_spin_reward = 1.0 # Based on "T-spin mini +1" possibly referring to 1-line mini.
                                        # This needs refinement with better T-spin classification.
                                        # If we assume any T-Spin 1 line is "Mini", then +1.
                                        # If it can be "Full T-Spin Single", reward should be higher.
                                        # Let's use prompt's values: TSD +3, TST +5.
                                        # If lines == 1 and is_tspin, it's likely a TSM -> +1
                    t_spin_reward = 1.0 # Simplified: T-Spin 1-line = +1
                elif lines == 2: # T-Spin Double
                    t_spin_reward = 3.0 # As per prompt
                elif lines == 3: # T-Spin Triple
                    t_spin_reward = 5.0 # As per prompt
            elif is_tspin and lines == 0: # T-Spin but no lines (e.g., T-Spin Mini setup)
                # Prompt doesn't explicitly state reward for 0-line T-Spin Mini, but implies "T-spin mini +1".
                # This "+1" could be for the setup itself or a 1-line clear.
                # Let's give a small reward for a 0-line T-spin maneuver.
                t_spin_reward = 0.5 # Small incentive for the setup

            # Base Line Clear Rewards (only if not a T-Spin line clear)
            if t_spin_reward == 0 and lines > 0:
                if lines == 1: base_clear_reward = 0.5
                elif lines == 2: base_clear_reward = 1.5
                elif lines == 3: base_clear_reward = 3.0
                elif lines == 4: base_clear_reward = 8.0 # Tetris

            # B2B bonus
            if self.b2b_state > 1 and (base_clear_reward > 0 or t_spin_reward > 0): # B2B requires a line clear
                # b2b_state counts number of consecutive difficult clears.
                # b2b_state = 1 is the first difficult clear.
                # b2b_state = 2 is the second, which should get the +1 bonus.
                b2b_bonus = (self.b2b_state - 1) * 1.0

            # Combo bonus
            if self.combo_count > 1 and (base_clear_reward > 0 or t_spin_reward > 0): # Combo requires a line clear
                # combo_count = 1 is the first clear in a sequence.
                # combo_count = 2 is the second, which should start bonus.
                # Prompt: "+combo chain length". If combo_count is the length,
                # then for combo_count=2, bonus is +2. For combo_count=3, bonus is +3.
                # This means bonus = self.combo_count if self.combo_count > 1.
                # Or, if it means "additional points equal to chain length beyond the first":
                # bonus = (self.combo_count -1) if self.combo_count > 1
                # Let's go with a scaled version of "length":
                combo_bonus = (self.combo_count - 1) * 1.0 # +1 for 2nd, +2 for 3rd etc.

            # Perfect Clear bonus
            if is_pc:
                pc_reward = 10.0

            reward = base_clear_reward + t_spin_reward + b2b_bonus + combo_bonus + pc_reward

        # Survival reward (small positive reward for not dying in this step)
        if not self.game_over:
            reward += 0.01 # Small reward per step survived

        # Penalty for topping out
        if self.game_over:
            terminated = True
            reward = -10.0 # Penalty for game over

        # (Future: Garbage sent / tanked rewards/penalties)
        # Garbage Sent: +1 per line (needs opponent model or simulation)
        # Garbage Tanked: -0.5 per line (when garbage actually lands on board)

        # Ensure a new piece is ready if previous one locked and game not over
        if not self.current_piece and not self.game_over:
            if not self._spawn_new_piece(): # This will set self.game_over if it fails
                terminated = True
                reward = -10.0 # Game over due to failed spawn after a lock

        # Info dict
        info = {}
        if self.last_lock_info:
            info.update(self.last_lock_info)
        info["score"] = self.score # self.score is not updated yet, use reward as proxy for now
        info["b2b_state"] = self.b2b_state
        info["combo_count"] = self.combo_count
        if self.game_over:
            info["reason"] = "Topped out or spawn failed"

        self.score += reward # Accumulate reward into score (optional, some envs don't track total score this way)

        return self._get_observation(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for reproducibility via seeding

        self.board.grid = np.zeros((self.total_board_height, self.board_width), dtype=int)
        self.piece_factory = PieceFactory() # Resets bag

        self.current_piece = None
        self.hold_piece = None
        self.can_hold_this_turn = True

        self.next_pieces_queue = []
        self._fill_next_pieces_queue()

        self.score = 0
        self.lines_cleared_total = 0
        self.level = 1
        self.game_over = False

        self.b2b_state = 0
        self.combo_count = 0
        self.incoming_garbage = 0
        self.last_clear_was_difficult = False
        self.last_lock_info = {}

        if not self._spawn_new_piece():
            # This should ideally not happen on a fresh reset unless board is too small.
            # If it does, the environment is misconfigured or there's a bug.
            # Handle this as an immediate game over.
            self.game_over = True
            # print("Warning: Game over immediately on reset due to spawn failure.")
            # Return a dummy observation or raise error.
            # For Gym compliance, must return obs and info.
            return self._get_observation(), {"error": "Immediate game over on reset"}

        return self._get_observation(), {"message": "Environment reset"}

    def render(self, mode='human'):
        if mode == 'rgb_array':
            # Return an RGB array representation of the game state
            # For example, convert the board grid to an image
            # This needs a mapping from piece IDs/colors to RGB values
            # For now, a simple representation:
            # Visible board: self.board_height x self.board_width
            # Each cell can be a color.

            # Create a base image (e.g., black background)
            # Visible part of board
            vis_board_height = self.board_height
            img = np.zeros((vis_board_height, self.board_width, 3), dtype=np.uint8)

            # Colors for pieces (similar to TETROMINOES but as dict for quick lookup)
            # Piece ID 0 is empty (black). IDs 1-7 are pieces.
            piece_colors = {
                0: (20, 20, 20), # Background for empty cells on board
                self.board.name_to_id('I'): (0, 255, 255),   # Cyan
                self.board.name_to_id('O'): (255, 255, 0),   # Yellow
                self.board.name_to_id('T'): (128, 0, 128),   # Purple
                self.board.name_to_id('S'): (0, 255, 0),     # Green
                self.board.name_to_id('Z'): (255, 0, 0),     # Red
                self.board.name_to_id('J'): (0, 0, 255),     # Blue
                self.board.name_to_id('L'): (255, 165, 0),   # Orange
                8: (100,100,100) # Ghost piece color (example)
            }

            # Draw locked pieces on the visible board
            board_state = self.board.get_board_state(include_hidden_rows=False) # Only visible part
            for r in range(vis_board_height):
                for c in range(self.board_width):
                    cell_val = board_state[r, c]
                    img[r, c] = piece_colors.get(cell_val, (50,50,50)) # Default to gray if unknown id

            # Draw current piece
            if self.current_piece:
                current_piece_y_offset = self.total_board_height - self.board_height # hidden_rows
                for r_idx, row_val in enumerate(self.current_piece.shape):
                    for c_idx, cell_val in enumerate(row_val):
                        if cell_val == 1:
                            board_r = self.current_piece.y + r_idx - current_piece_y_offset
                            board_c = self.current_piece.x + c_idx
                            if 0 <= board_r < vis_board_height and 0 <= board_c < self.board_width:
                                img[board_r, board_c] = self.current_piece.color
            return img

        elif mode == 'human':
            # Print to console (simple text representation)
            # Get visible board state
            display_grid = self.board.get_board_state(include_hidden_rows=False).copy().astype(str)
            display_grid[display_grid == '0'] = '.' # Empty cells as dots

            # Draw current piece onto this display grid representation
            if self.current_piece:
                current_piece_y_offset = self.total_board_height - self.board_height # hidden_rows
                for r_idx, row_val in enumerate(self.current_piece.shape):
                    for c_idx, cell_val in enumerate(row_val):
                        if cell_val == 1:
                            # Position relative to visible board's top-left
                            vis_r = self.current_piece.y + r_idx - current_piece_y_offset
                            vis_c = self.current_piece.x + c_idx
                            if 0 <= vis_r < self.board_height and 0 <= vis_c < self.board_width:
                                display_grid[vis_r, vis_c] = self.current_piece.name[0] # Show first letter of piece

            print("\n" + "=" * (self.board_width * 2 + 3))
            for r in range(self.board_height):
                print(f"| {' '.join(display_grid[r, :])} |")
            print("=" * (self.board_width * 2 + 3))

            # Print game info
            hold_p_name = self.hold_piece.name if self.hold_piece else "None"
            next_p_names = ", ".join([p.name for p in self.next_pieces_queue[:3]])
            print(f"Hold: {hold_p_name}  Next: {next_p_names}")
            print(f"Score: {self.score:.2f}  Lines: {self.lines_cleared_total}")
            print(f"B2B: {self.b2b_state}  Combo: {self.combo_count}")
            if self.game_over: print("--- GAME OVER ---")
            return None # For 'human' mode, typically render to screen and return None

    def close(self):
        # Clean up any resources if needed (e.g., Pygame window)
        pass

if __name__ == '__main__':
    # Example Usage:
    # env = TetrisEnv(width=10, height=20, hidden_rows=4) # Smaller hidden for console print
    env = TetrisEnv(width=10, height=20, hidden_rows=20)
    obs, info = env.reset()

    print("Initial Observation:")
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"{key}: {value}")
    print(info)
    env.render(mode='human')

    # Simulate a few random actions (using the simplified action space)
    # Note: The current _handle_action is a placeholder for high-level placements.
    # For proper testing, either implement that or switch to granular actions.
    # For now, let's test with some manual "gravity" steps and holds.

    if env.manual_action_override: # This is false by default
        # Example for manual action: (x, y, rot_idx, piece_name)
        # This requires knowing the current piece name.
        # Let current_piece_name = env.current_piece.name
        # action = (env.board_width // 2 -1, env.board_height -1, 0, current_piece_name)
        # obs, reward, terminated, truncated, info = env.step(action)
        pass
    else:
        # Test with some random high-level "placement" actions
        # Action: 0-39 for (col,rot), 40 for Hold
        for i in range(200): # Max 200 steps
            if env.game_over: break

            # Simplistic agent: try to hold if I piece, else try random placement
            action = env.action_space.sample() # Random action
            if env.current_piece and env.current_piece.name == 'I' and env.can_hold_this_turn:
                 if env.hold_piece is None or env.hold_piece.name != 'I':
                    action = env.action_space_size -1 # Hold action index

            # Or, try a specific placement for testing if current piece is known
            # e.g., if current piece is 'T', try action corresponding to col 4, rot 0
            # action_T_col4_rot0 = 0 * env.board_width + 4
            # action = action_T_col4_rot0

            print(f"\nStep {i+1}, Action: {action}")
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"Reward: {reward:.2f}, Terminated: {terminated}")
            print(f"Info: {info}")
            env.render(mode='human')

            if terminated:
                print(f"Game Over after {i+1} steps.")
                break

            # input("Press Enter to continue...") # For step-by-step viewing

    env.close()
