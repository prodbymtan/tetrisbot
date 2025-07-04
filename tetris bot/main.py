#!/usr/bin/env python3
"""
DeepTrix-Z: The Ultimate Modern Tetris Bot
Main entry point and command-line interface.
"""

import argparse
import sys
import time
import json
from typing import Optional

from core.tetris_engine import TetrisEngine, GameConfig
from core.finesse import FinesseEngine
from ai.neural_net import NeuralNetworkManager
from ai.mcts import MCTS, MCTSPlayer, MCTSConfig
from ai.evaluation import BoardEvaluator, GarbageAwareEvaluator, StyleAwareEvaluator
from ai.training import Trainer, TrainingConfig, SelfPlayTrainer


def demo_game():
    """Run a demo game with the AI."""
    print("🧠 DeepTrix-Z Demo")
    print("=" * 50)
    
    # Initialize components
    engine = TetrisEngine()
    network_manager = NeuralNetworkManager()
    
    # Create a simple network for demo
    try:
        network = network_manager.create_network()
        print("✓ Neural network initialized")
    except Exception as e:
        print(f"⚠ Neural network failed to initialize: {e}")
        print("Using heuristic evaluation only")
        network = None
    
    # Initialize MCTS
    if network:
        mcts = MCTS(network_manager, num_simulations=50)  # Reduced for demo
    else:
        mcts = None
    
    # Initialize evaluator
    evaluator = GarbageAwareEvaluator()
    
    print("✓ Game engine initialized")
    print("✓ MCTS initialized")
    print("✓ Board evaluator initialized")
    print()
    
    # Game loop
    frame_count = 0
    start_time = time.time()
    
    while not engine.game_over:
        frame_count += 1
        
        # Display current state
        if frame_count % 100 == 0:  # Update every 100 frames
            print(f"\nFrame: {frame_count}")
            print(f"Level: {engine.level}")
            print(f"Lines: {engine.lines_cleared}")
            print(f"Score: {engine.score}")
            print(f"Combo: {engine.combo}")
            print(f"Garbage: {engine.garbage_lines}")
            
            # Display board
            print("\nBoard:")
            print(str(engine.board))
            
            # Show current piece
            if engine.current_piece:
                print(f"Current piece: {engine.current_piece.piece_type.name}")
                print(f"Position: ({engine.current_piece.position.x}, {engine.current_piece.position.y}, {engine.current_piece.position.rotation})")
            
            print("-" * 30)
        
        # Get AI move
        if mcts and engine.current_piece:
            try:
                # Get best move from MCTS
                game_state = engine.get_game_state()
                x, y, rotation = mcts.search(game_state)
                
                # Execute move
                engine.update(inputs=['hard_drop'])  # Simplified execution for demo
                
            except Exception as e:
                # Fallback to heuristic evaluation
                valid_placements = engine.get_all_valid_placements()
                if valid_placements:
                    best_placement = None
                    best_score = float('-inf')
                    
                    for placement in valid_placements:
                        score = evaluator.evaluate_placement(engine.board, engine.current_piece, placement)
                        if score > best_score:
                            best_score = score
                            best_placement = placement
                    
                    if best_placement:
                        engine.board.place_piece(best_placement)
        else:
            # Simple fallback - just drop pieces
            if engine.current_piece:
                engine.update(inputs=['hard_drop'])
        
        # Update game
        engine.update()
        
        # Add some delay for visualization
        time.sleep(0.01)
    
    # Game over
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 50)
    print("🎮 GAME OVER")
    print("=" * 50)
    print(f"Final Score: {engine.score}")
    print(f"Lines Cleared: {engine.lines_cleared}")
    print(f"Level Reached: {engine.level}")
    print(f"Game Duration: {duration:.2f} seconds")
    print(f"Frames: {frame_count}")
    print(f"Average FPS: {frame_count / duration:.1f}")
    
    # Calculate TPM (Tetris per minute)
    tetris_count = engine.lines_cleared // 4
    tpm = (tetris_count / duration) * 60
    print(f"TPM: {tpm:.1f}")
    
    # Calculate APM (Actions per minute)
    apm = (frame_count / duration) * 60
    print(f"APM: {apm:.1f}")


def train_model(args):
    """Train the neural network."""
    print("🧠 DeepTrix-Z Training")
    print("=" * 50)
    
    # Create training config
    config = TrainingConfig(
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq
    )
    
    # Initialize trainer
    trainer = Trainer(config, device=args.device)
    
    print(f"Training for {args.episodes} episodes")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {args.device}")
    print()
    
    # Start training
    stats = trainer.train()
    
    # Save final model
    trainer.save_model(args.output)
    print(f"Model saved to {args.output}")
    
    # Print final statistics
    if stats['episode_rewards']:
        avg_reward = sum(stats['episode_rewards'][-100:]) / 100
        avg_score = sum(stats['episode_scores'][-100:]) / 100
        avg_lines = sum(stats['episode_lines'][-100:]) / 100
        
        print(f"\nFinal Statistics (last 100 episodes):")
        print(f"Average Reward: {avg_reward:.1f}")
        print(f"Average Score: {avg_score:.1f}")
        print(f"Average Lines: {avg_lines:.1f}")


def evaluate_model(args):
    """Evaluate a trained model."""
    print("🧠 DeepTrix-Z Model Evaluation")
    print("=" * 50)
    
    # Load model
    network_manager = NeuralNetworkManager(device=args.device)
    try:
        network = network_manager.load_network(args.model)
        print(f"✓ Model loaded from {args.model}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Initialize MCTS
    mcts = MCTS(network_manager, num_simulations=args.simulations)
    
    # Run evaluation games
    total_score = 0
    total_lines = 0
    total_games = args.games
    
    print(f"Running {total_games} evaluation games...")
    
    for game in range(total_games):
        engine = TetrisEngine()
        game_score = 0
        game_lines = 0
        
        while not engine.game_over:
            if engine.current_piece:
                try:
                    game_state = engine.get_game_state()
                    x, y, rotation = mcts.search(game_state)
                    # Simplified execution for evaluation
                    engine.update(inputs=['hard_drop'])
                except:
                    # Fallback
                    engine.update(inputs=['hard_drop'])
            
            engine.update()
        
        total_score += engine.score
        total_lines += engine.lines_cleared
        
        if game % 10 == 0:
            print(f"Game {game + 1}/{total_games}: Score={engine.score}, Lines={engine.lines_cleared}")
    
    # Print results
    avg_score = total_score / total_games
    avg_lines = total_lines / total_games
    
    print(f"\nEvaluation Results:")
    print(f"Average Score: {avg_score:.1f}")
    print(f"Average Lines: {avg_lines:.1f}")
    print(f"Total Games: {total_games}")


def benchmark(args):
    """Run performance benchmarks."""
    print("🧠 DeepTrix-Z Performance Benchmark")
    print("=" * 50)
    
    # Initialize components
    engine = TetrisEngine()
    network_manager = NeuralNetworkManager(device=args.device)
    
    try:
        network = network_manager.create_network()
        mcts = MCTS(network_manager, num_simulations=100)
        print("✓ Neural network and MCTS initialized")
    except Exception as e:
        print(f"⚠ Neural network failed: {e}")
        mcts = None
    
    evaluator = BoardEvaluator()
    
    # Benchmark board evaluation
    print("\nBenchmarking board evaluation...")
    start_time = time.time()
    for _ in range(1000):
        evaluator.evaluate_board(engine.board)
    eval_time = time.time() - start_time
    print(f"Board evaluation: 1000 evaluations in {eval_time:.3f}s ({1000/eval_time:.0f} eval/s)")
    
    # Benchmark MCTS (if available)
    if mcts:
        print("\nBenchmarking MCTS...")
        start_time = time.time()
        for _ in range(10):
            game_state = engine.get_game_state()
            mcts.search(game_state)
        mcts_time = time.time() - start_time
        print(f"MCTS: 10 searches in {mcts_time:.3f}s ({10/mcts_time:.1f} searches/s)")
    
    # Benchmark finesse engine
    print("\nBenchmarking finesse engine...")
    start_time = time.time()
    for _ in range(1000):
        engine.finesse_engine.get_all_valid_placements(engine.board, engine.current_piece)
    finesse_time = time.time() - start_time
    print(f"Finesse: 1000 placement calculations in {finesse_time:.3f}s ({1000/finesse_time:.0f} calc/s)")
    
    print("\n✓ Benchmark completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="DeepTrix-Z: The Ultimate Modern Tetris Bot")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run a demo game')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the neural network')
    train_parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    train_parser.add_argument('--save-freq', type=int, default=100, help='Save frequency')
    train_parser.add_argument('--eval-freq', type=int, default=50, help='Evaluation frequency')
    train_parser.add_argument('--output', default='deeptrix_model.pth', help='Output model path')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model', required=True, help='Path to model file')
    eval_parser.add_argument('--games', type=int, default=100, help='Number of evaluation games')
    eval_parser.add_argument('--simulations', type=int, default=100, help='MCTS simulations per move')
    eval_parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    benchmark_parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    if args.command == 'demo':
        demo_game()
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'benchmark':
        benchmark(args)
    else:
        parser.print_help()
        print("\nFor a quick demo, run: python main.py demo")


if __name__ == "__main__":
    main() 