# DeepTrix-Z - A Competitive Tetris RL Bot
# exceptions.py - Custom exceptions for the Tetris game engine

class CollisionException(Exception):
    """Custom exception for piece collisions."""
    pass

class InvalidMoveException(Exception):
    """Custom exception for invalid piece movements or rotations."""
    pass

class GameOverException(Exception):
    """Custom exception for game over condition."""
    pass
