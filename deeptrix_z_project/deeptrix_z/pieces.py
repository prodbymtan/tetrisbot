# DeepTrix-Z - A Competitive Tetris RL Bot
# pieces.py - Defines Tetrominoes, rotations (SRS), and piece generation

import random

class Piece:
    """
    Represents a single Tetris piece (Tetromino).
    """
    def __init__(self, shape_name: str, color: tuple, rotations: list[list[list[int]]]):
        self.name = shape_name
        self.color = color  # RGB tuple
        self.rotations = rotations  # List of 4 rotation matrices
        self.rotation_state = 0  # Current rotation index (0-3)
        self.shape = self.rotations[self.rotation_state]
        self.x = 0  # Column position of the top-left of the piece's bounding box
        self.y = 0  # Row position of the top-left of the piece's bounding box

    def rotate(self, direction: int):
        """
        Rotates the piece.
        :param direction: 1 for clockwise, -1 for counter-clockwise.
        """
        self.rotation_state = (self.rotation_state + direction) % 4
        self.shape = self.rotations[self.rotation_state]

    def get_block_positions(self) -> list[tuple[int, int]]:
        """
        Returns a list of (row, col) for each filled block of the current piece shape,
        relative to the board coordinates.
        """
        positions = []
        for r_idx, row in enumerate(self.shape):
            for c_idx, cell in enumerate(row):
                if cell == 1:
                    positions.append((self.y + r_idx, self.x + c_idx))
        return positions

    def get_bounding_box_size(self) -> tuple[int, int]:
        """Returns (height, width) of the current piece's shape matrix."""
        return len(self.shape), len(self.shape[0]) if self.shape else (0,0)

# Standard Tetromino shapes and their SRS rotation states
# 1 represents a block, 0 represents empty space within the piece's bounding box.
# Shapes are defined for their initial spawn orientation.
TETROMINOES = {
    'I': {
        'color': (0, 255, 255),  # Cyan
        'rotations': [
            [[0,0,0,0], [1,1,1,1], [0,0,0,0], [0,0,0,0]],
            [[0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0]],
            [[0,0,0,0], [0,0,0,0], [1,1,1,1], [0,0,0,0]],
            [[0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0]],
        ]
    },
    'O': {
        'color': (255, 255, 0),  # Yellow
        'rotations': [
            [[1,1], [1,1]],
            [[1,1], [1,1]],
            [[1,1], [1,1]],
            [[1,1], [1,1]],
        ]
    },
    'T': {
        'color': (128, 0, 128),  # Purple
        'rotations': [
            [[0,1,0], [1,1,1], [0,0,0]],
            [[0,1,0], [0,1,1], [0,1,0]],
            [[0,0,0], [1,1,1], [0,1,0]],
            [[0,1,0], [1,1,0], [0,1,0]],
        ]
    },
    'S': {
        'color': (0, 255, 0),  # Green
        'rotations': [
            [[0,1,1], [1,1,0], [0,0,0]],
            [[0,1,0], [0,1,1], [0,0,1]],
            [[0,0,0], [0,1,1], [1,1,0]],
            [[1,0,0], [1,1,0], [0,1,0]],
        ]
    },
    'Z': {
        'color': (255, 0, 0),  # Red
        'rotations': [
            [[1,1,0], [0,1,1], [0,0,0]],
            [[0,0,1], [0,1,1], [0,1,0]],
            [[0,0,0], [1,1,0], [0,1,1]],
            [[0,1,0], [1,1,0], [1,0,0]],
        ]
    },
    'J': {
        'color': (0, 0, 255),  # Blue
        'rotations': [
            [[1,0,0], [1,1,1], [0,0,0]],
            [[0,1,1], [0,1,0], [0,1,0]],
            [[0,0,0], [1,1,1], [0,0,1]],
            [[0,1,0], [0,1,0], [1,1,0]],
        ]
    },
    'L': {
        'color': (255, 165, 0),  # Orange
        'rotations': [
            [[0,0,1], [1,1,1], [0,0,0]],
            [[0,1,0], [0,1,0], [0,1,1]],
            [[0,0,0], [1,1,1], [1,0,0]],
            [[1,1,0], [0,1,0], [0,1,0]],
        ]
    }
}

# SRS Wall Kick Data (from https://tetris.wiki/Super_Rotation_System#Wall_Kicks)
# (offset_col, offset_row)
# For J, L, S, T, Z pieces
JLSTZ_WALL_KICKS = {
    # 0->1 (Clockwise)
    (0, 1): [(0,0), (-1,0), (-1,+1), (0,-2), (-1,-2)],
    # 1->0 (Counter-Clockwise)
    (1, 0): [(0,0), (+1,0), (+1,-1), (0,+2), (+1,+2)],
    # 1->2 (Clockwise)
    (1, 2): [(0,0), (+1,0), (+1,-1), (0,+2), (+1,+2)],
    # 2->1 (Counter-Clockwise)
    (2, 1): [(0,0), (-1,0), (-1,+1), (0,-2), (-1,-2)],
    # 2->3 (Clockwise)
    (2, 3): [(0,0), (+1,0), (+1,+1), (0,-2), (+1,-2)],
    # 3->2 (Counter-Clockwise)
    (3, 2): [(0,0), (-1,0), (-1,-1), (0,+2), (-1,+2)],
    # 3->0 (Clockwise)
    (3, 0): [(0,0), (-1,0), (-1,-1), (0,+2), (-1,+2)],
    # 0->3 (Counter-Clockwise)
    (0, 3): [(0,0), (+1,0), (+1,+1), (0,-2), (+1,-2)],
}

# For I piece
I_WALL_KICKS = {
    # 0->1 (Clockwise)
    (0, 1): [(0,0), (-2,0), (+1,0), (-2,-1), (+1,+2)],
    # 1->0 (Counter-Clockwise)
    (1, 0): [(0,0), (+2,0), (-1,0), (+2,+1), (-1,-2)],
    # 1->2 (Clockwise)
    (1, 2): [(0,0), (-1,0), (+2,0), (-1,+2), (+2,-1)],
    # 2->1 (Counter-Clockwise)
    (2, 1): [(0,0), (+1,0), (-2,0), (+1,-2), (-2,+1)],
    # 2->3 (Clockwise)
    (2, 3): [(0,0), (+2,0), (-1,0), (+2,+1), (-1,-2)],
    # 3->2 (Counter-Clockwise)
    (3, 2): [(0,0), (-2,0), (+1,0), (-2,-1), (+1,+2)],
    # 3->0 (Clockwise)
    (3, 0): [(0,0), (+1,0), (-2,0), (+1,-2), (-2,+1)],
    # 0->3 (Counter-Clockwise)
    (0, 3): [(0,0), (-1,0), (+2,0), (-1,+2), (+2,-1)],
}

class PieceFactory:
    """
    Generates sequences of Tetris pieces using a 7-bag system.
    """
    def __init__(self):
        self.bag = []
        self._shuffle_bag()

    def _shuffle_bag(self):
        self.bag = list(TETROMINOES.keys())
        random.shuffle(self.bag)

    def next_piece(self) -> Piece:
        if not self.bag:
            self._shuffle_bag()
        piece_name = self.bag.pop(0)
        piece_data = TETROMINOES[piece_name]
        return Piece(piece_name, piece_data['color'], piece_data['rotations'])

if __name__ == '__main__':
    # Example usage:
    factory = PieceFactory()
    for _ in range(14):
        p = factory.next_piece()
        print(f"Generated piece: {p.name}, Color: {p.color}")
        print(f"Initial shape (rotation 0):")
        for row in p.shape:
            print(row)

        p.rotate(1) # Clockwise
        print(f"Shape after 1 clockwise rotation (rotation 1):")
        for row in p.shape:
            print(row)
        print("-" * 10)

    # Test I piece rotations specifically
    print("\nTesting I piece rotations:")
    i_piece = Piece('I', TETROMINOES['I']['color'], TETROMINOES['I']['rotations'])
    for i in range(5):
        print(f"I piece rotation {i_piece.rotation_state}:")
        for r in i_piece.shape: print(r)
        kick_data_key = (i_piece.rotation_state, (i_piece.rotation_state + 1) % 4)
        print(f"Kicks for {i_piece.name} {kick_data_key}: {I_WALL_KICKS.get(kick_data_key)}")
        i_piece.rotate(1)
        print("---")

    # Test T piece rotations specifically
    print("\nTesting T piece rotations:")
    t_piece = Piece('T', TETROMINOES['T']['color'], TETROMINOES['T']['rotations'])
    for i in range(5):
        print(f"T piece rotation {t_piece.rotation_state}:")
        for r in t_piece.shape: print(r)
        kick_data_key = (t_piece.rotation_state, (t_piece.rotation_state + 1) % 4)
        print(f"Kicks for {t_piece.name} {kick_data_key}: {JLSTZ_WALL_KICKS.get(kick_data_key)}")
        t_piece.rotate(1)
        print("---")

```
