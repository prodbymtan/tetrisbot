import tkinter as tk
import random
import time
import os
import numpy as np
from core import TetrisEngine

# --- DQN imports ---
import torch
import torch.nn as nn

CELL_SIZE = 30
BOARD_WIDTH = 10
BOARD_HEIGHT = 20

COLORS = {
    0: "#222222",  # Empty
    1: "#888888",  # Filled
    'current': "#00ffff",  # Current piece
}

ACTIONS = ['left', 'right', 'rotate_cw', 'rotate_ccw', 'soft_drop', 'hard_drop']
STATE_SIZE = BOARD_WIDTH * BOARD_HEIGHT + 1

# --- DQN Model (must match rl_train.py) ---
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    def forward(self, x):
        return self.net(x)

def get_state(engine):
    board = engine.board.board.flatten()
    piece_type = int(engine.current_piece.piece_type.value) if engine.current_piece else 0
    state = np.concatenate([board, [piece_type]])
    return state.astype(np.float32)

class TetrisBotUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Tetris Bot MVP")
        self.engine = TetrisEngine()
        self.running = False
        self.use_dqn = False
        self.dqn_loaded = False
        self.policy_net = None
        self.load_dqn()

        self.canvas = tk.Canvas(root, width=BOARD_WIDTH*CELL_SIZE, height=BOARD_HEIGHT*CELL_SIZE, bg="#111111")
        self.canvas.pack()

        self.info_label = tk.Label(root, text="Score: 0 | Lines: 0", font=("Arial", 14))
        self.info_label.pack()

        self.start_button = tk.Button(root, text="Start Bot", command=self.start_bot)
        self.start_button.pack(pady=5)

        self.toggle_button = tk.Button(root, text="Mode: Random", command=self.toggle_mode)
        self.toggle_button.pack(pady=5)

        self.draw_board()

    def load_dqn(self):
        if os.path.exists("dqn_tetris.pth"):
            self.policy_net = DQN(STATE_SIZE, len(ACTIONS))
            self.policy_net.load_state_dict(torch.load("dqn_tetris.pth", map_location=torch.device('cpu')))
            self.policy_net.eval()
            self.dqn_loaded = True
        else:
            self.policy_net = None
            self.dqn_loaded = False

    def draw_board(self):
        self.canvas.delete("all")
        board = self.engine.board.board
        # Draw filled cells
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                color = COLORS[1] if board[y][x] else COLORS[0]
                self.canvas.create_rectangle(
                    x*CELL_SIZE, y*CELL_SIZE, (x+1)*CELL_SIZE, (y+1)*CELL_SIZE,
                    fill=color, outline="#333"
                )
        # Draw current piece
        if self.engine.current_piece:
            for px, py in self.engine.current_piece.get_occupied_cells():
                if 0 <= px < BOARD_WIDTH and 0 <= py < BOARD_HEIGHT:
                    self.canvas.create_rectangle(
                        px*CELL_SIZE, py*CELL_SIZE, (px+1)*CELL_SIZE, (py+1)*CELL_SIZE,
                        fill=COLORS['current'], outline="#fff"
                    )
        # Update info
        self.info_label.config(text=f"Score: {self.engine.score} | Lines: {self.engine.lines_cleared}")

    def start_bot(self):
        if not self.running:
            self.engine.reset()  # Reset the engine to start a new game
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.run_bot()

    def toggle_mode(self):
        self.use_dqn = not self.use_dqn
        if self.use_dqn and not self.dqn_loaded:
            self.toggle_button.config(text="Mode: DQN (not found)")
        elif self.use_dqn:
            self.toggle_button.config(text="Mode: DQN")
        else:
            self.toggle_button.config(text="Mode: Random")

    def select_action(self, state):
        if self.use_dqn and self.dqn_loaded and self.policy_net is not None:
            with torch.no_grad():
                state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.policy_net(state_v)
                action_idx = int(torch.argmax(q_values).item())
                return [ACTIONS[action_idx]]
        else:
            # Random move, mostly not hard_drop
            if random.random() < 0.2:
                return ['hard_drop']
            else:
                return [random.choice(ACTIONS[:-1])]

    def run_bot(self):
        if self.engine.game_over:
            self.info_label.config(text=f"GAME OVER! Score: {self.engine.score} | Lines: {self.engine.lines_cleared}")
            self.start_button.config(state=tk.NORMAL)
            self.running = False
            return
        state = get_state(self.engine)
        actions = self.select_action(state)
        self.engine.update(actions)
        self.draw_board()
        self.root.after(80, self.run_bot)

if __name__ == "__main__":
    root = tk.Tk()
    app = TetrisBotUI(root)
    root.mainloop() 