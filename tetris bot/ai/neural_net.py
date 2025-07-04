"""
Neural networks for DeepTrix-Z.
Policy and value networks for MCTS guidance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class NetworkOutput:
    """Output from the neural network."""
    policy: torch.Tensor  # Action probabilities
    value: torch.Tensor   # Board evaluation
    features: torch.Tensor  # Intermediate features


class TetrisNet(nn.Module):
    """Main neural network for Tetris evaluation and policy."""
    
    def __init__(self, board_height: int = 20, board_width: int = 10, 
                 num_pieces: int = 7, num_actions: int = 200):
        super().__init__()
        
        self.board_height = board_height
        self.board_width = board_width
        self.num_pieces = num_pieces
        self.num_actions = num_actions
        
        # Board encoding layers
        self.board_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.board_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.board_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Piece encoding layers
        self.piece_embedding = nn.Embedding(num_pieces, 32)
        self.piece_linear = nn.Linear(32, 64)
        
        # Game state encoding
        self.state_linear = nn.Linear(10, 64)  # 10 game state features
        
        # Combined features
        self.combined_linear1 = nn.Linear(128 * board_height * board_width + 64 + 64, 512)
        self.combined_linear2 = nn.Linear(512, 256)
        
        # Policy head
        self.policy_linear1 = nn.Linear(256, 128)
        self.policy_linear2 = nn.Linear(128, num_actions)
        
        # Value head
        self.value_linear1 = nn.Linear(256, 128)
        self.value_linear2 = nn.Linear(128, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
    
    def forward(self, board: torch.Tensor, current_piece: torch.Tensor, 
                game_state: torch.Tensor) -> NetworkOutput:
        """Forward pass through the network."""
        batch_size = board.size(0)
        
        # Process board
        x_board = F.relu(self.bn1(self.board_conv1(board)))
        x_board = F.relu(self.bn2(self.board_conv2(x_board)))
        x_board = F.relu(self.bn3(self.board_conv3(x_board)))
        x_board = x_board.view(batch_size, -1)
        
        # Process current piece
        x_piece = self.piece_embedding(current_piece)
        x_piece = F.relu(self.piece_linear(x_piece))
        
        # Process game state
        x_state = F.relu(self.state_linear(game_state))
        
        # Combine features
        x_combined = torch.cat([x_board, x_piece, x_state], dim=1)
        x_combined = F.relu(self.combined_linear1(x_combined))
        x_combined = self.dropout(x_combined)
        x_combined = F.relu(self.combined_linear2(x_combined))
        x_combined = self.dropout(x_combined)
        
        # Policy head
        x_policy = F.relu(self.policy_linear1(x_combined))
        x_policy = self.policy_linear2(x_policy)
        policy = F.softmax(x_policy, dim=1)
        
        # Value head
        x_value = F.relu(self.value_linear1(x_combined))
        x_value = self.value_linear2(x_value)
        value = torch.tanh(x_value)
        
        return NetworkOutput(
            policy=policy,
            value=value,
            features=x_combined
        )


class PolicyNetwork(nn.Module):
    """Policy network for action selection."""
    
    def __init__(self, input_size: int, hidden_size: int = 256, num_actions: int = 200):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class ValueNetwork(nn.Module):
    """Value network for board evaluation."""
    
    def __init__(self, input_size: int, hidden_size: int = 256):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.tanh(x)


class BoardEncoder:
    """Encodes board state for neural network input."""
    
    def __init__(self, board_height: int = 20, board_width: int = 10):
        self.board_height = board_height
        self.board_width = board_width
    
    def encode_board(self, board: np.ndarray) -> torch.Tensor:
        """Encode board as tensor."""
        # Convert to float tensor and add batch dimension
        board_tensor = torch.FloatTensor(board).unsqueeze(0).unsqueeze(0)
        return board_tensor
    
    def encode_piece(self, piece_type: int) -> torch.Tensor:
        """Encode piece type as tensor."""
        return torch.LongTensor([piece_type])
    
    def encode_game_state(self, stats: dict) -> torch.Tensor:
        """Encode game state statistics as tensor."""
        features = [
            stats.get('level', 1) / 20.0,  # Normalized level
            stats.get('lines_cleared', 0) / 100.0,  # Normalized lines
            stats.get('combo', 0) / 10.0,  # Normalized combo
            stats.get('garbage_lines', 0) / 10.0,  # Normalized garbage
            stats.get('board_height', 0) / 20.0,  # Normalized height
            stats.get('holes', 0) / 50.0,  # Normalized holes
            stats.get('bumpiness', 0) / 50.0,  # Normalized bumpiness
            stats.get('t_spin_potential', 0) / 10.0,  # Normalized T-spin potential
            stats.get('combo_potential', 0) / 10.0,  # Normalized combo potential
            stats.get('attack', 0) / 20.0,  # Normalized attack
        ]
        return torch.FloatTensor(features).unsqueeze(0)


class NeuralNetworkManager:
    """Manages neural network training and inference."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.network = None
        self.optimizer = None
        self.encoder = BoardEncoder()
        
    def create_network(self, board_height: int = 20, board_width: int = 10, 
                      num_pieces: int = 7, num_actions: int = 200) -> TetrisNet:
        """Create a new neural network."""
        self.network = TetrisNet(board_height, board_width, num_pieces, num_actions)
        self.network.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        
        return self.network
    
    def load_network(self, path: str) -> TetrisNet:
        """Load a trained network from file."""
        self.network = TetrisNet()
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.network.to(self.device)
        self.network.eval()
        
        return self.network
    
    def save_network(self, path: str):
        """Save the current network to file."""
        if self.network:
            torch.save(self.network.state_dict(), path)
    
    def predict(self, board: np.ndarray, current_piece: int, 
                game_state: dict) -> Tuple[np.ndarray, float]:
        """Get policy and value predictions."""
        if self.network is None:
            raise ValueError("Network not initialized")
        
        self.network.eval()
        
        with torch.no_grad():
            # Encode inputs
            board_tensor = self.encoder.encode_board(board).to(self.device)
            piece_tensor = self.encoder.encode_piece(current_piece).to(self.device)
            state_tensor = self.encoder.encode_game_state(game_state).to(self.device)
            
            # Get predictions
            output = self.network(board_tensor, piece_tensor, state_tensor)
            
            # Convert to numpy
            policy = output.policy.cpu().numpy().flatten()
            value = output.value.cpu().numpy().flatten()[0]
            
            return policy, value
    
    def train_step(self, boards: List[np.ndarray], pieces: List[int], 
                   states: List[dict], target_policies: List[np.ndarray], 
                   target_values: List[float]) -> float:
        """Perform one training step."""
        if self.network is None:
            raise ValueError("Network not initialized")
        
        self.network.train()
        
        # Prepare batch
        batch_size = len(boards)
        board_tensors = torch.stack([self.encoder.encode_board(board).squeeze() 
                                   for board in boards]).to(self.device)
        piece_tensors = torch.LongTensor(pieces).to(self.device)
        state_tensors = torch.stack([self.encoder.encode_game_state(state).squeeze() 
                                   for state in states]).to(self.device)
        
        target_policy_tensors = torch.FloatTensor(target_policies).to(self.device)
        target_value_tensors = torch.FloatTensor(target_values).unsqueeze(1).to(self.device)
        
        # Forward pass
        output = self.network(board_tensors, piece_tensors, state_tensors)
        
        # Calculate loss
        policy_loss = F.cross_entropy(output.policy, target_policy_tensors)
        value_loss = F.mse_loss(output.value, target_value_tensors)
        total_loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def get_network_summary(self) -> dict:
        """Get a summary of the network architecture."""
        if self.network is None:
            return {"error": "Network not initialized"}
        
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "architecture": str(self.network)
        }


class NetworkEnsemble:
    """Ensemble of neural networks for improved predictions."""
    
    def __init__(self, num_networks: int = 3, device: str = 'cpu'):
        self.num_networks = num_networks
        self.device = torch.device(device)
        self.networks = []
        self.encoder = BoardEncoder()
        
        # Create ensemble
        for _ in range(num_networks):
            network = TetrisNet()
            network.to(self.device)
            self.networks.append(network)
    
    def predict(self, board: np.ndarray, current_piece: int, 
                game_state: dict) -> Tuple[np.ndarray, float]:
        """Get ensemble predictions."""
        policies = []
        values = []
        
        for network in self.networks:
            network.eval()
            
            with torch.no_grad():
                # Encode inputs
                board_tensor = self.encoder.encode_board(board).to(self.device)
                piece_tensor = self.encoder.encode_piece(current_piece).to(self.device)
                state_tensor = self.encoder.encode_game_state(game_state).to(self.device)
                
                # Get predictions
                output = network(board_tensor, piece_tensor, state_tensor)
                
                policies.append(output.policy.cpu().numpy().flatten())
                values.append(output.value.cpu().numpy().flatten()[0])
        
        # Average predictions
        avg_policy = np.mean(policies, axis=0)
        avg_value = np.mean(values)
        
        return avg_policy, avg_value
    
    def load_ensemble(self, paths: List[str]):
        """Load ensemble from saved networks."""
        if len(paths) != self.num_networks:
            raise ValueError(f"Expected {self.num_networks} network paths, got {len(paths)}")
        
        for i, path in enumerate(paths):
            self.networks[i].load_state_dict(torch.load(path, map_location=self.device))
    
    def save_ensemble(self, base_path: str):
        """Save ensemble to files."""
        for i, network in enumerate(self.networks):
            path = f"{base_path}_network_{i}.pth"
            torch.save(network.state_dict(), path) 