import random
import time
from core import TetrisEngine

# List of possible actions for the bot
ACTIONS = ['left', 'right', 'rotate_cw', 'rotate_ccw', 'soft_drop', 'hard_drop']

def main():
    engine = TetrisEngine()
    print("=== Simple MVP Tetris Bot ===")
    print(engine)
    time.sleep(1)

    while not engine.game_over:
        # Pick a random action (except hard_drop, which we do every few moves)
        if random.random() < 0.2:
            actions = ['hard_drop']
        else:
            actions = [random.choice(ACTIONS[:-1])]  # Exclude hard_drop for most moves

        # Clamp target position to valid board range
        chosen_x = max(0, min(BOARD_WIDTH - 1, chosen_pos[0]))
        chosen_rotation = chosen_pos[1] % 4

        engine.update(actions)
        print(engine)
        time.sleep(0.1)

    print("\nGame Over!")
    print(f"Final Score: {engine.score}")
    print(f"Lines Cleared: {engine.lines_cleared}")

if __name__ == "__main__":
    main() 