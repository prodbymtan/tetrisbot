import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from core import TetrisEngine

# --- Hyperparameters ---
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
STATE_SIZE = 13  # 10 heights + holes + bumpiness + piece type
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10
EPISODES = 500

# --- DQN Model ---
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

def get_features(engine):
    board = engine.board
    heights = board.get_height_map()  # List of 10 ints
    holes = board.get_holes()
    bumpiness = board.get_bumpiness()
    piece_type = int(engine.current_piece.piece_type.value) if engine.current_piece else 0
    return np.array(heights + [holes, bumpiness, piece_type], dtype=np.float32)

def select_action(state, policy_net, epsilon, valid_actions):
    if random.random() < epsilon:
        return random.randrange(len(valid_actions))
    with torch.no_grad():
        state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = policy_net(state_v)
        valid_q = q_values[0][[i for i in range(len(valid_actions))]]
        return int(torch.argmax(valid_q).item())

def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1)[0].detach().unsqueeze(1)
    expected_q = rewards + (1 - dones) * GAMMA * next_q_values
    loss = nn.functional.mse_loss(q_values, expected_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    # For final placement, we need a fixed action space: (x, rotation) pairs
    all_positions = [(x, r) for x in range(-2, BOARD_WIDTH+2) for r in range(4)]
    action_size = len(all_positions)
    policy_net = DQN(STATE_SIZE, action_size)
    target_net = DQN(STATE_SIZE, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = EPSILON_START
    avg_rewards = []

    for episode in range(EPISODES):
        engine = TetrisEngine()
        state = get_features(engine)
        total_reward = 0
        steps = 0
        while not engine.game_over:
            # Get all valid placements for the current piece
            valid_positions = engine.get_all_valid_placements()
            if not valid_positions:
                break
            # Map valid_positions to action indices
            valid_actions = []
            for pos in valid_positions:
                try:
                    idx = all_positions.index((pos.x, pos.rotation))
                    valid_actions.append(idx)
                except ValueError:
                    continue
            if not valid_actions:
                break
            action_idx = select_action(state, policy_net, epsilon, valid_actions)
            chosen_idx = valid_actions[action_idx]
            chosen_pos = all_positions[chosen_idx]
            # Get the finesse path to the target position
            target_position = Position(chosen_pos[0], 0, chosen_pos[1])
            finesse_path = engine.get_finesse_path(target_position)
            if finesse_path is not None:
                for action in finesse_path.inputs:
                    engine.update([action.action])
            else:
                # Fallback: just hard drop
                engine.update(['hard_drop'])
            prev_lines = engine.lines_cleared
            prev_holes = engine.board.get_holes()
            next_state = get_features(engine)
            # Reward: +1 per line, -0.1 per step, -holes*0.2, -10 for game over
            reward = (engine.lines_cleared - prev_lines)
            reward -= 0.1
            reward -= engine.board.get_holes() * 0.2
            if engine.game_over:
                reward -= 10
            done = engine.game_over
            memory.append((state, chosen_idx, reward, next_state, done))
            state = next_state
            total_reward += reward
            steps += 1
            optimize_model(memory, policy_net, target_net, optimizer)
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        avg_rewards.append(total_reward)
        print(f"Episode {episode+1}/{EPISODES} | Steps: {steps} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")
    torch.save(policy_net.state_dict(), "dqn_tetris.pth")
    print("Training complete. Model saved as dqn_tetris.pth")
    plt.plot(avg_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve')
    plt.savefig('learning_curve.png')
    plt.show()

if __name__ == "__main__":
    main() 