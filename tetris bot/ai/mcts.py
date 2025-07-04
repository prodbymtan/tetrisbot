"""
Monte Carlo Tree Search (MCTS) for DeepTrix-Z.
Implements AlphaZero-style MCTS with neural network guidance.
"""

import math
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import time
from copy import deepcopy

from ..core.tetris_engine import TetrisEngine, GameState
from ..core.pieces import Piece, Position
from .neural_net import NeuralNetworkManager


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    state: GameState
    parent: Optional['MCTSNode']
    action: Optional[Tuple[int, int, int]]  # (x, y, rotation)
    children: Dict[Tuple[int, int, int], 'MCTSNode']
    visit_count: int
    total_value: float
    prior_probability: float
    is_terminal: bool
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
    
    @property
    def value(self) -> float:
        """Get the average value of this node."""
        return self.total_value / max(1, self.visit_count)
    
    @property
    def ucb_score(self) -> float:
        """Calculate UCB score for this node."""
        if self.visit_count == 0:
            return float('inf')
        
        # UCB1 formula with exploration constant
        exploration_constant = 1.414  # sqrt(2)
        exploitation = self.value
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visit_count) / self.visit_count)
        
        return exploitation + exploration


class MCTS:
    """Monte Carlo Tree Search with neural network guidance."""
    
    def __init__(self, network_manager: NeuralNetworkManager, 
                 num_simulations: int = 800, exploration_constant: float = 1.414):
        self.network_manager = network_manager
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.root = None
        
    def search(self, game_state: GameState, time_limit: Optional[float] = None) -> Tuple[int, int, int]:
        """Search for the best action using MCTS."""
        # Create root node
        self.root = MCTSNode(
            state=game_state,
            parent=None,
            action=None,
            children={},
            visit_count=0,
            total_value=0.0,
            prior_probability=1.0,
            is_terminal=False
        )
        
        # Expand root node
        self._expand_node(self.root)
        
        # Run simulations
        start_time = time.time()
        simulation_count = 0
        
        while simulation_count < self.num_simulations:
            if time_limit and (time.time() - start_time) > time_limit:
                break
            
            # Selection
            node = self._select(self.root)
            
            # Expansion
            if not node.is_terminal and node.visit_count > 0:
                node = self._expand_node(node)
            
            # Simulation
            value = self._simulate(node)
            
            # Backpropagation
            self._backpropagate(node, value)
            
            simulation_count += 1
        
        # Select best action
        best_action = self._select_best_action()
        return best_action
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a node using UCB1."""
        while node.children:
            # Find child with highest UCB score
            best_child = None
            best_score = float('-inf')
            
            for child in node.children.values():
                score = child.ucb_score
                if score > best_score:
                    best_score = score
                    best_child = child
            
            node = best_child
        
        return node
    
    def _expand_node(self, node: MCTSNode) -> MCTSNode:
        """Expand a node by creating child nodes for all possible actions."""
        if node.is_terminal:
            return node
        
        # Get valid actions (piece placements)
        engine = TetrisEngine()
        engine.set_game_state(node.state)
        
        if not engine.current_piece:
            node.is_terminal = True
            return node
        
        # Get all valid placements
        valid_placements = engine.get_all_valid_placements()
        
        if not valid_placements:
            node.is_terminal = True
            return node
        
        # Get neural network predictions
        board = engine.board.board
        current_piece = engine.current_piece.piece_type.value
        game_stats = engine.get_stats()
        
        try:
            policy, value = self.network_manager.predict(board, current_piece, game_stats)
        except:
            # Fallback to uniform policy if network fails
            policy = np.ones(len(valid_placements)) / len(valid_placements)
            value = 0.0
        
        # Create child nodes
        for i, placement in enumerate(valid_placements):
            action = (placement.x, placement.y, placement.rotation)
            
            # Create new game state
            new_engine = TetrisEngine()
            new_engine.set_game_state(node.state)
            
            # Place piece
            new_piece = Piece(engine.current_piece.piece_type, placement)
            if new_engine.board.place_piece(new_piece):
                new_state = new_engine.get_game_state()
                
                # Create child node
                child = MCTSNode(
                    state=new_state,
                    parent=node,
                    action=action,
                    children={},
                    visit_count=0,
                    total_value=0.0,
                    prior_probability=policy[i] if i < len(policy) else 0.1,
                    is_terminal=new_state.game_over
                )
                
                node.children[action] = child
        
        # Return a random child for simulation
        if node.children:
            return np.random.choice(list(node.children.values()))
        
        return node
    
    def _simulate(self, node: MCTSNode) -> float:
        """Simulate a random playout from the given node."""
        if node.is_terminal:
            return self._evaluate_terminal_state(node.state)
        
        # Create a copy of the game state for simulation
        engine = TetrisEngine()
        engine.set_game_state(node.state)
        
        # Simulate random playout
        max_moves = 50  # Limit simulation length
        move_count = 0
        
        while not engine.game_over and move_count < max_moves:
            # Get valid placements
            valid_placements = engine.get_all_valid_placements()
            
            if not valid_placements:
                break
            
            # Choose random placement
            placement = np.random.choice(valid_placements)
            
            # Place piece
            piece = Piece(engine.current_piece.piece_type, placement)
            if not engine.board.place_piece(piece):
                break
            
            move_count += 1
        
        # Evaluate final state
        return self._evaluate_state(engine.get_game_state())
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate the simulation result up the tree."""
        while node:
            node.visit_count += 1
            node.total_value += value
            node = node.parent
    
    def _select_best_action(self) -> Tuple[int, int, int]:
        """Select the best action based on visit counts."""
        if not self.root or not self.root.children:
            return (3, 0, 0)  # Default spawn position
        
        # Find child with highest visit count
        best_child = None
        best_visits = -1
        
        for child in self.root.children.values():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_child = child
        
        return best_child.action
    
    def _evaluate_state(self, state: GameState) -> float:
        """Evaluate a game state."""
        # Use neural network for evaluation
        engine = TetrisEngine()
        engine.set_game_state(state)
        
        if state.game_over:
            return -1.0  # Game over is bad
        
        board = engine.board.board
        current_piece = engine.current_piece.piece_type.value if engine.current_piece else 0
        game_stats = engine.get_stats()
        
        try:
            _, value = self.network_manager.predict(board, current_piece, game_stats)
            return value
        except:
            # Fallback evaluation
            return self._heuristic_evaluation(state)
    
    def _evaluate_terminal_state(self, state: GameState) -> float:
        """Evaluate a terminal game state."""
        if state.game_over:
            return -1.0  # Loss
        else:
            return 1.0  # Win (unlikely in Tetris)
    
    def _heuristic_evaluation(self, state: GameState) -> float:
        """Heuristic evaluation function as fallback."""
        engine = TetrisEngine()
        engine.set_game_state(state)
        
        stats = engine.get_stats()
        
        # Normalize features
        height_penalty = -stats['board_height'] / 20.0
        holes_penalty = -stats['holes'] / 50.0
        bumpiness_penalty = -stats['bumpiness'] / 50.0
        lines_bonus = stats['lines_cleared'] / 100.0
        combo_bonus = stats['combo'] / 10.0
        t_spin_bonus = stats['t_spin_potential'] / 10.0
        
        # Weighted sum
        evaluation = (
            height_penalty * 0.3 +
            holes_penalty * 0.4 +
            bumpiness_penalty * 0.2 +
            lines_bonus * 0.1 +
            combo_bonus * 0.1 +
            t_spin_bonus * 0.1
        )
        
        return np.tanh(evaluation)  # Bound to [-1, 1]
    
    def get_action_probabilities(self) -> Dict[Tuple[int, int, int], float]:
        """Get action probabilities based on visit counts."""
        if not self.root or not self.root.children:
            return {}
        
        total_visits = sum(child.visit_count for child in self.root.children.values())
        
        probabilities = {}
        for action, child in self.root.children.items():
            probabilities[action] = child.visit_count / total_visits
        
        return probabilities
    
    def get_search_info(self) -> Dict[str, Any]:
        """Get information about the search."""
        if not self.root:
            return {"error": "No search performed"}
        
        return {
            "root_visits": self.root.visit_count,
            "root_value": self.root.value,
            "num_children": len(self.root.children),
            "best_action": self._select_best_action(),
            "action_probabilities": self.get_action_probabilities()
        }


class MCTSPlayer:
    """Player that uses MCTS for decision making."""
    
    def __init__(self, network_manager: NeuralNetworkManager, 
                 num_simulations: int = 800, time_limit: Optional[float] = None):
        self.mcts = MCTS(network_manager, num_simulations)
        self.time_limit = time_limit
        self.network_manager = network_manager
    
    def get_move(self, game_state: GameState) -> Tuple[int, int, int]:
        """Get the best move for the current game state."""
        return self.mcts.search(game_state, self.time_limit)
    
    def get_move_with_probabilities(self, game_state: GameState) -> Tuple[Tuple[int, int, int], Dict]:
        """Get the best move and action probabilities."""
        best_action = self.mcts.search(game_state, self.time_limit)
        probabilities = self.mcts.get_action_probabilities()
        return best_action, probabilities
    
    def update_search_parameters(self, num_simulations: Optional[int] = None, 
                                time_limit: Optional[float] = None):
        """Update MCTS search parameters."""
        if num_simulations is not None:
            self.mcts.num_simulations = num_simulations
        if time_limit is not None:
            self.time_limit = time_limit


class MCTSConfig:
    """Configuration for MCTS."""
    
    def __init__(self, num_simulations: int = 800, time_limit: Optional[float] = None,
                 exploration_constant: float = 1.414, temperature: float = 1.0):
        self.num_simulations = num_simulations
        self.time_limit = time_limit
        self.exploration_constant = exploration_constant
        self.temperature = temperature  # For action selection temperature


class AdaptiveMCTS(MCTS):
    """MCTS that adapts its search based on game situation."""
    
    def __init__(self, network_manager: NeuralNetworkManager, config: MCTSConfig):
        super().__init__(network_manager, config.num_simulations, config.exploration_constant)
        self.config = config
        self.game_phase = "opening"  # opening, midgame, endgame
    
    def search(self, game_state: GameState, time_limit: Optional[float] = None) -> Tuple[int, int, int]:
        """Adaptive search based on game phase."""
        # Determine game phase
        self._update_game_phase(game_state)
        
        # Adjust search parameters based on phase
        if self.game_phase == "opening":
            # More simulations for opening moves
            self.num_simulations = int(self.config.num_simulations * 1.5)
        elif self.game_phase == "endgame":
            # Even more simulations for critical endgame moves
            self.num_simulations = int(self.config.num_simulations * 2.0)
        else:
            # Standard simulations for midgame
            self.num_simulations = self.config.num_simulations
        
        return super().search(game_state, time_limit)
    
    def _update_game_phase(self, game_state: GameState):
        """Update the current game phase."""
        lines_cleared = game_state.lines_cleared
        
        if lines_cleared < 40:
            self.game_phase = "opening"
        elif lines_cleared < 120:
            self.game_phase = "midgame"
        else:
            self.game_phase = "endgame"
    
    def get_phase_info(self) -> Dict[str, Any]:
        """Get information about the current game phase."""
        return {
            "phase": self.game_phase,
            "num_simulations": self.num_simulations,
            "search_info": self.get_search_info()
        } 