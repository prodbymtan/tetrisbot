import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv # For parallel envs

from deeptrix_z_project.deeptrix_z.game import TetrisEnv # Adjusted import path

# --- Configuration ---
LOG_DIR = "ppo_tetris_logs"
MODEL_SAVE_PATH = "ppo_tetris_model"
TOTAL_TIMESTEPS = 1_000_000  # Adjust as needed (e.g., 10M for serious training)
N_ENVS = 4 # Number of parallel environments, adjust based on CPU cores
LEARNING_RATE = 0.0003 # Common default for PPO
N_STEPS = 2048       # Number of steps to run for each environment per update
BATCH_SIZE = 64        # Minibatch size
N_EPOCHS = 10          # Number of epochs when optimizing the surrogate loss
GAMMA = 0.99           # Discount factor
GAE_LAMBDA = 0.95      # Factor for trade-off of bias vs variance for GAE
CLIP_RANGE = 0.2       # Clipping parameter PPO
ENT_COEF = 0.0         # Entropy coefficient (can tune this, 0 to 0.01 is common)
VF_COEF = 0.5          # Value function coefficient

# Ensure log and model save directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    """
    def _init():
        # Use a different seed for each environment
        env = TetrisEnv(width=10, height=20, hidden_rows=20, num_next_pieces=5)
        # Important: Seed the environment. Each SubprocVecEnv instance needs its own seed.
        # However, gym.Env doesn't always have a direct seed method in reset or init in older gym.
        # Gymnasium's reset takes a seed.
        # For custom env, ensure reset uses the seed. TetrisEnv's reset already calls super().reset(seed=seed)
        return env
    return _init

if __name__ == "__main__":
    print("Starting Tetris PPO training script...")

    # --- Create the Tetris Environment ---
    # Check the custom environment (optional, but good for debugging)
    # print("Checking custom environment...")
    # single_env = TetrisEnv()
    # check_env(single_env, warn=True) # Check if it adheres to Gym API
    # print("Environment check passed.")
    # del single_env # Clean up

    # Vectorized environments for parallel processing
    print(f"Creating {N_ENVS} parallel environments...")
    if N_ENVS > 1:
        vec_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    else:
        vec_env = DummyVecEnv([make_env(0)])

    print("Environments created.")

    # --- Define Callbacks ---
    # Checkpoint callback to save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // N_ENVS, 1), # Save every 100k steps, adjusted by N_ENVS
        save_path=LOG_DIR,
        name_prefix="tetris_ppo_model"
    )

    # Evaluation callback (optional, but highly recommended)
    # It will periodically evaluate the model on a separate test environment
    # eval_env = TetrisEnv(width=10, height=20, hidden_rows=20, num_next_pieces=5)
    # eval_vec_env = DummyVecEnv([lambda: eval_env]) # Wrap for SB3

    # eval_callback = EvalCallback(
    #     eval_vec_env,
    #     best_model_save_path=os.path.join(LOG_DIR, "best_model"),
    #     log_path=LOG_DIR,
    #     eval_freq=max(50_000 // N_ENVS, 1), # Evaluate every 50k steps
    #     deterministic=True,
    #     render=False
    # )
    # print("Callbacks defined.")

    # --- Define the PPO Model ---
    # For Dict observation spaces, "MultiInputPolicy" is used.
    # SB3 will automatically create a CombinedExtractor.
    # The policy_kwargs can be used to customize the network architecture (e.g., net_arch, cnn_extractor).
    # Default CNN for images is usually good enough to start.
    # Features from "board" and "ghost_piece_map" will go to CNN.
    # Other 1D features will be concatenated.

    policy_kwargs = dict(
        # Example of customizing the CNN feature extractor if needed:
        # features_extractor_class=CombinedExtractor, # Default for Dict space
        # features_extractor_kwargs=dict(
        #     cnn_output_dim=256, # Example
        # ),
        net_arch=dict(pi=[128, 128], vf=[128, 128]) # Customize actor and critic networks
                                                  # Default is [64,64] for both if not specified for MLP part
    )

    print("Defining PPO model...")
    model = PPO(
        "MultiInputPolicy",       # Use MultiInputPolicy for Dict observation spaces
        vec_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        policy_kwargs=policy_kwargs,
        verbose=1,                # 0 for no output, 1 for info, 2 for debug
        tensorboard_log=LOG_DIR
    )
    print("PPO Model defined.")
    print(f"Model policy network: {model.policy}")


    # --- Start Training ---
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            # callback=[checkpoint_callback, eval_callback], # Add eval_callback if defined
            callback=[checkpoint_callback],
            progress_bar=True # Requires tqdm to be installed
        )
        print("Training finished.")

        # --- Save the final model ---
        model.save(MODEL_SAVE_PATH)
        print(f"Final model saved to {MODEL_SAVE_PATH}.zip")

        # Example of loading and using the model (optional)
        # print("Loading trained model...")
        # loaded_model = PPO.load(MODEL_SAVE_PATH)
        # print("Model loaded.")

        # obs = vec_env.reset()
        # for _ in range(1000):
        #     action, _states = loaded_model.predict(obs, deterministic=True)
        #     obs, rewards, dones, info = vec_env.step(action)
        #     # vec_env.render(mode="human") # May not work well with SubprocVecEnv directly
        #     if any(dones):
        #         print("An episode finished during post-training test.")
        #         obs = vec_env.reset() # Reset only the environments that are done

    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close the environment
        vec_env.close()
        # if 'eval_vec_env' in locals():
        #     eval_vec_env.close()
        print("Environments closed.")

    print("Training script finished.")

```

**Key aspects of this `train_ppo.py` script:**

*   **Environment Setup:**
    *   Uses `TetrisEnv` from the `deeptrix_z` module.
    *   Includes `make_env` helper for creating multiple environments for `SubprocVecEnv` (for parallel training, significantly speeds up wall-clock time for training).
    *   `check_env` is commented out but is useful for debugging the custom environment.
*   **PPO Model Configuration:**
    *   Uses `"MultiInputPolicy"` which is designed for `Dict` observation spaces. `stable-baselines3` automatically handles the feature extraction:
        *   Keys from the observation `Dict` that are `Box` spaces with 2 or 3 dimensions (like our "board" and "ghost_piece_map" which are `(1, H, W)`) are assumed to be image-like and are passed to a CNN.
        *   Other `Box` or `Discrete` features (like "current_piece_id", "b2b_state", etc.) are typically concatenated to the CNN's output before being fed into the main policy/value networks (MLPs).
    *   `policy_kwargs` can be used to customize network architectures (e.g., number of layers, units per layer, CNN output dimensions). I've added a basic `net_arch` for the policy (pi) and value (vf) function MLPs.
    *   Includes common hyperparameters for PPO (`LEARNING_RATE`, `N_STEPS`, `BATCH_SIZE`, `N_EPOCHS`, `GAMMA`, `GAE_LAMBDA`, `CLIP_RANGE`, `ENT_COEF`, `VF_COEF`). These often require tuning.
*   **Callbacks:**
    *   `CheckpointCallback`: Saves the model periodically during training.
    *   `EvalCallback` (commented out for now as it requires a separate eval env setup): Periodically evaluates the model on a separate environment and saves the best performing model. This is crucial for tracking actual performance improvement.
*   **Training and Saving:**
    *   `model.learn()` starts the training process.
    *   `model.save()` saves the final trained model.
*   **Logging:**
    *   `tensorboard_log=LOG_DIR` enables logging of training progress to TensorBoard.

**Next Steps within this plan item:**
The script is written. The main part of "implementing the PPO model" is using the `stable-baselines3` PPO class with the correct policy type for our `Dict` observation space. The `MultiInputPolicy` and the underlying `CombinedExtractor` handle the complex input structure.

The crucial parts are:
1.  Correctly setting up the `TetrisEnv` with its `Dict` observation space. (Done in previous steps)
2.  Using `"MultiInputPolicy"` with the PPO agent. (Done in the script)
3.  Ensuring the shapes and types of observations match what the policy expects. `stable-baselines3` usually handles this well.

The design is to leverage SB3's built-in capabilities for handling dictionary observation spaces with mixed image-like and vector features. The "CnnPolicy" mentioned in the prompt is implicitly part of what "MultiInputPolicy" will use for the image-like parts of our observation space.

The script `train_ppo.py` now provides the framework for training the PPO agent on the `TetrisEnv`.
