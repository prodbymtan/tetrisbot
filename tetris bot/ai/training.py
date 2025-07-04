"""
Training module for DeepTrix-Z.
Handles reinforcement learning, self-play training, and data collection.
"""

import torch
import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import time
import json
import os
from collections import deque
import pickle

from ..core.tetris_engine import TetrisEngine, GameState
from ..core.pieces import Piece, Position, PieceType
from .neural_net import NeuralNetworkManager, TetrisNet
from .mcts import MCTS, MCTSPlayer
from .evaluation import BoardEvaluator, GarbageAwareEvaluator


@dataclass
class TrainingConfig:
    """Configuration for training."""
    num_episodes: int = 10000
    batch_size: int = 32
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update_freq: int = 1000
    save_freq: int = 100
    eval_freq: int = 50
    max_memory_size: int = 100000
    min_memory_size: int = 1000


@dataclass
class GameRecord:
    """Record of a complete game."""
    states: List[np.ndarray]
    actions: List[Tuple[int, int, int]]
    rewards: List[float]
    final_score: float
    lines_cleared: int
    game_length: int


class ExperienceBuffer:
    """Buffer for storing training experiences."""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state: np.ndarray, action: Tuple[int, int, int], 
            reward: float, next_state: np.ndarray, done: bool):
        """Add an experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample a batch of experiences."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class Trainer:
    """Main trainer for DeepTrix-Z."""
    
    def __init__(self, config: TrainingConfig, device: str = 'cpu'):
        self.config = config
        self.device = torch.device(device)
        
        # Initialize components
        self.network_manager = NeuralNetworkManager(device)
        self.network = self.network_manager.create_network()
        self.target_network = self.network_manager.create_network()
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Initialize MCTS and evaluator
        self.mcts = MCTS(self.network_manager, num_simulations=100)
        self.evaluator = GarbageAwareEvaluator()
        
        # Training state
        self.experience_buffer = ExperienceBuffer(config.max_memory_size)
        self.epsilon = config.epsilon_start
        self.episode_count = 0
        self.total_steps = 0
        
        # Statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_scores': [],
            'episode_lines': [],
            'losses': [],
            'epsilon_values': []
        }
    
    def train_episode(self) -> Dict[str, float]:
        """Train for one episode."""
        engine = TetrisEngine()
        episode_reward = 0
        episode_length = 0
        episode_data = []
        
        while not engine.game_over:
            # Get current state
            current_state = engine.get_game_state()
            board = engine.board.board
            current_piece = engine.current_piece.piece_type.value if engine.current_piece else 0
            game_stats = engine.get_stats()
            
            # Choose action
            if random.random() < self.epsilon:
                # Random action
                valid_placements = engine.get_all_valid_placements()
                if valid_placements:
                    action = random.choice(valid_placements)
                    x, y, rotation = action.x, action.y, action.rotation
                else:
                    break
            else:
                # MCTS action
                try:
                    x, y, rotation = self.mcts.search(current_state)
                except:
                    # Fallback to random
                    valid_placements = engine.get_all_valid_placements()
                    if valid_placements:
                        action = random.choice(valid_placements)
                        x, y, rotation = action.x, action.y, action.rotation
                    else:
                        break
            
            # Execute action
            placement = Position(x, y, rotation)
            piece = Piece(engine.current_piece.piece_type, placement)
            
            # Store state before action
            pre_state = engine.board.board.copy()
            
            # Execute action
            if engine.place_piece(piece):
                # Calculate reward
                reward = self._calculate_reward(engine, pre_state)
                episode_reward += reward
                
                # Store experience
                next_state = engine.board.board.copy()
                done = engine.game_over
                
                self.experience_buffer.add(
                    pre_state, (x, y, rotation), reward, next_state, done
                )
                
                episode_data.append({
                    'state': pre_state.copy(),
                    'action': (x, y, rotation),
                    'reward': reward,
                    'next_state': next_state.copy(),
                    'done': done
                })
            
            episode_length += 1
            self.total_steps += 1
        
        # Train on episode data
        if len(self.experience_buffer) >= self.config.min_memory_size:
            loss = self._train_step()
            self.training_stats['losses'].append(loss)
        
        # Update epsilon
        self.epsilon = max(self.config.epsilon_end, 
                          self.epsilon * self.config.epsilon_decay)
        
        # Update target network
        if self.episode_count % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        
        # Record statistics
        self.training_stats['episode_rewards'].append(episode_reward)
        self.training_stats['episode_lengths'].append(episode_length)
        self.training_stats['episode_scores'].append(engine.score)
        self.training_stats['episode_lines'].append(engine.lines_cleared)
        self.training_stats['epsilon_values'].append(self.epsilon)
        
        self.episode_count += 1
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'final_score': engine.score,
            'lines_cleared': engine.lines_cleared,
            'epsilon': self.epsilon
        }
    
    def _calculate_reward(self, engine: TetrisEngine, pre_state: np.ndarray) -> float:
        """Calculate reward for an action."""
        reward = 0
        
        # Reward for lines cleared
        lines_cleared = engine.lines_cleared
        if lines_cleared > 0:
            reward += lines_cleared * 100
        
        # Reward for combos
        reward += engine.combo * 10
        
        # Reward for T-spins (simplified)
        t_spin_potential = engine.board.get_t_spin_potential()
        reward += t_spin_potential * 5
        
        # Penalty for game over
        if engine.game_over:
            reward -= 1000
        
        # Penalty for high board
        height_map = engine.board.get_height_map()
        if height_map:
            max_height = max(height_map)
            reward -= max_height * 2
        
        # Penalty for holes
        holes = engine.board.get_holes()
        reward -= holes * 10
        
        return reward
    
    def _train_step(self) -> float:
        """Perform one training step."""
        # Sample batch
        batch = self.experience_buffer.sample(self.config.batch_size)
        
        # Prepare data
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for state, action, reward, next_state, done in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.BoolTensor(dones).to(self.device)
        
        # Calculate target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor)
            target_q_values = rewards_tensor + (self.config.gamma * next_q_values * ~dones_tensor)
        
        # Calculate current Q-values
        current_q_values = self.network(states_tensor)
        
        # Calculate loss
        loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)
        
        # Backward pass
        self.network_manager.optimizer.zero_grad()
        loss.backward()
        self.network_manager.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes: Optional[int] = None) -> Dict[str, List[float]]:
        """Train for multiple episodes."""
        if num_episodes is None:
            num_episodes = self.config.num_episodes
        
        print(f"Starting training for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            episode_stats = self.train_episode()
            
            if episode % 10 == 0:
                print(f"Episode {episode}: Reward={episode_stats['episode_reward']:.1f}, "
                      f"Score={episode_stats['final_score']}, "
                      f"Lines={episode_stats['lines_cleared']}, "
                      f"Epsilon={self.epsilon:.3f}")
            
            # Save model periodically
            if episode % self.config.save_freq == 0:
                self.save_model(f"model_episode_{episode}.pth")
            
            # Evaluate periodically
            if episode % self.config.eval_freq == 0:
                eval_score = self.evaluate()
                print(f"Evaluation score: {eval_score:.1f}")
        
        return self.training_stats
    
    def evaluate(self, num_games: int = 10) -> float:
        """Evaluate the current model."""
        total_score = 0
        
        for _ in range(num_games):
            engine = TetrisEngine()
            
            while not engine.game_over:
                current_state = engine.get_game_state()
                
                try:
                    x, y, rotation = self.mcts.search(current_state)
                    placement = Position(x, y, rotation)
                    piece = Piece(engine.current_piece.piece_type, placement)
                    engine.place_piece(piece)
                except:
                    break
            
            total_score += engine.score
        
        return total_score / num_games
    
    def save_model(self, path: str):
        """Save the current model."""
        self.network_manager.save_network(path)
        
        # Save training stats
        stats_path = path.replace('.pth', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f)
    
    def load_model(self, path: str):
        """Load a trained model."""
        self.network_manager.load_network(path)
        
        # Load training stats if available
        stats_path = path.replace('.pth', '_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.training_stats = json.load(f)


class SelfPlayTrainer(Trainer):
    """Trainer that uses self-play for training."""
    
    def __init__(self, config: TrainingConfig, device: str = 'cpu'):
        super().__init__(config, device)
        self.game_records = []
    
    def collect_self_play_data(self, num_games: int = 100) -> List[GameRecord]:
        """Collect self-play data."""
        game_records = []
        
        for game in range(num_games):
            engine = TetrisEngine()
            states = []
            actions = []
            rewards = []
            
            while not engine.game_over:
                # Get current state
                current_state = engine.get_game_state()
                board = engine.board.board.copy()
                
                # Get MCTS action with probabilities
                try:
                    x, y, rotation = self.mcts.search(current_state)
                    action_probs = self.mcts.get_action_probabilities()
                except:
                    break
                
                # Store state and action
                states.append(board)
                actions.append((x, y, rotation))
                
                # Execute action
                placement = Position(x, y, rotation)
                piece = Piece(engine.current_piece.piece_type, placement)
                
                if engine.place_piece(piece):
                    # Calculate reward
                    reward = self._calculate_reward(engine, board)
                    rewards.append(reward)
                else:
                    break
            
            # Create game record
            game_record = GameRecord(
                states=states,
                actions=actions,
                rewards=rewards,
                final_score=engine.score,
                lines_cleared=engine.lines_cleared,
                game_length=len(states)
            )
            
            game_records.append(game_record)
            
            if game % 10 == 0:
                print(f"Collected {game + 1}/{num_games} self-play games")
        
        return game_records
    
    def train_on_self_play_data(self, game_records: List[GameRecord]) -> float:
        """Train the network on self-play data."""
        if not game_records:
            return 0.0
        
        # Prepare training data
        all_states = []
        all_actions = []
        all_rewards = []
        
        for record in game_records:
            all_states.extend(record.states)
            all_actions.extend(record.actions)
            all_rewards.extend(record.rewards)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(all_states)).to(self.device)
        rewards_tensor = torch.FloatTensor(all_rewards).to(self.device)
        
        # Train network
        self.network.train()
        
        # Forward pass
        predictions = self.network(states_tensor)
        
        # Calculate loss (simplified - in practice you'd use proper policy/value targets)
        loss = torch.nn.functional.mse_loss(predictions, rewards_tensor.unsqueeze(1))
        
        # Backward pass
        self.network_manager.optimizer.zero_grad()
        loss.backward()
        self.network_manager.optimizer.step()
        
        return loss.item()


class HumanDataTrainer(Trainer):
    """Trainer that learns from human game data."""
    
    def __init__(self, config: TrainingConfig, device: str = 'cpu'):
        super().__init__(config, device)
        self.human_data = []
    
    def load_human_data(self, data_path: str):
        """Load human game data from file."""
        # This would load data from TETR.IO, Jstris, or other sources
        # Format depends on the data source
        pass
    
    def train_on_human_data(self, data: List[Dict]) -> float:
        """Train the network on human game data."""
        # Implementation would depend on the data format
        pass


class StyleMimicTrainer(Trainer):
    """Trainer that learns to mimic specific player styles."""
    
    def __init__(self, config: TrainingConfig, target_style: str, device: str = 'cpu'):
        super().__init__(config, device)
        self.target_style = target_style
        
        # Load style-specific data
        self.style_data = self._load_style_data(target_style)
    
    def _load_style_data(self, style: str) -> List[Dict]:
        """Load data for a specific playing style."""
        # This would load data from top players like Diao, VinceHD, etc.
        # Implementation depends on available data
        return []
    
    def train_style_mimic(self) -> float:
        """Train the network to mimic the target style."""
        # Implementation would train the network to reproduce the style
        pass 