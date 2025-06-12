"""
go9_refactor.py – Modern, commented, and more efficient rewrite of the original
`go9Data.py`, `go9ML.py`, and `go9MLWorkingCopy.py` files. The module focuses on
three concerns that were previously inter‑mixed:

1. **Data loading / preprocessing** – `Go9Dataset` provides an iterator that
   loads raw 9 × 9 Go positions from the original binary‑like "*.dat" files,
   converts them to `numpy` arrays, and exposes a ready‑to‑use
   `tf.data.Dataset` pipeline.
2. **Model definitions** – Two Keras factory helpers build either a simple
   linear model (logistic‐regression‑style) or a deeper CNN that mirrors the
   original TensorFlow 1 graph but in idiomatic TF 2.x.
3. **Training / evaluation utilities** – Convenience wrappers for compiling,
   training, and evaluating the chosen model and for suggesting a next move on a
   given board state.

The design is intentionally *modular* so you can drop‐in replace any one part
(e.g. swap the CNN with a Transformer) without touching the data layer or the
CLI entry‑point.
"""
from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import tensorflow as tf

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

###############################################################################
# Constants & type aliases                                                    #
###############################################################################

BOARD_SIZE = 9  # 9 × 9 boards only – extend easily for 19 × 19 if needed
NUM_INTERSECTIONS = BOARD_SIZE * BOARD_SIZE

# Mapping raw *.dat characters to numeric plane values (black = 1, white = −1)
CHAR_TO_VALUE = {b".": 0.0, b"#": 1.0, b"O": -1.0}  # Empty, black, white stones

# Typed alias for clarity when returning a pair of NDArrays
ArrayPair = Tuple[np.ndarray, np.ndarray]

###############################################################################
# Data‑loading                                                                #
###############################################################################

class Go9Dataset:
    """Helper that parses raw *.dat files into train / validation sets.

    The original code mixed low‑level byte handling with TensorFlow bookkeeping
    and wrote pickle caches to disk. This version keeps everything in‑memory
    (or lets TF 2 stream directly from disk) using a clean generator + tf.data.

    Parameters
    ----------
    data_dir : str | Path
        Directory that contains the original `input_positions.dat`,
        `zobrist_board_81.dat`, and `fuego_chinese.dat` files.
    max_records : int | None, default None
        Optional cap for quick experimentation or unit tests. Pass `None` to
        exhaust the entire file.
    """

    def __init__(self, data_dir: str | Path, *, max_records: int | None = None):
        self._data_dir = Path(data_dir)
        self._max_records = max_records

        pos_file = self._data_dir / "input_positions.dat"
        score_file = self._data_dir / "fuego_chinese.dat"

        if not pos_file.exists() or not score_file.exists():
            raise FileNotFoundError(
                "Expected Go data files in the directory but they were not found."
            )

        LOGGER.info("Loading board positions from %s…", pos_file)
        self._boards, self._scores = self._load_positions(pos_file, score_file)
        LOGGER.info(
            "Loaded %d positions (%s, %.1f MB).",
            len(self._boards),
            self._boards.dtype,
            self._boards.nbytes / 2 ** 20,
        )

    # ---------------------------------------------------------------------
    # Public API                                                            
    # ---------------------------------------------------------------------

    @property
    def numpy(self) -> ArrayPair:
        """Return `(boards, scores)` as two np.ndarray values."""
        return self._boards, self._scores

    def to_tf_dataset(self, batch_size: int = 128, shuffle: bool = True) -> tf.data.Dataset:
        """Convert the entire corpus into a `tf.data.Dataset` pipeline."""
        boards, scores = self.numpy
        ds = tf.data.Dataset.from_tensor_slices((boards, scores))
        if shuffle:
            ds = ds.shuffle(len(boards), reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    # ------------------------------------------------------------------
    # Internal helpers                                                   
    # ------------------------------------------------------------------

    def _load_positions(self, pos_file: Path, score_file: Path) -> ArrayPair:
        """Read `input_positions.dat` + companion score file into NDArrays."""
        # Build a fast lookup dict from zobrist hash → final score (two bytes)
        score_lookup: dict[bytes, int] = {}
        with score_file.open("rb") as sf:
            for line in sf:
                if len(line) < 12:  # Expect ≥ 8 (zobrist) + 2 (bytes) + 1 (player) +…
                    continue
                zobrist, result = line[:8], line[10:12]
                score_lookup[zobrist] = int.from_bytes(result, "big", signed=False)

        boards: list[np.ndarray] = []
        scores: list[int] = []

        # Each record in `input_positions.dat` has the structure:
        #   8 B zobrist hash
        #   1 B delimiter (ignored)
        #   1 B player turn ("W" or other)
        #  81 B board chars {'.', '#', 'O'}
        #   1 B newline  (optional) / 8 B something else (ignored)
        record_size = 8 + 1 + 1 + NUM_INTERSECTIONS
        raw = pos_file.read_bytes()
        if len(raw) % record_size != 0:
            LOGGER.warning(
                "File size (%d) is not a multiple of single‑record size (%d). "
                "Some trailing bytes may be ignored.",
                len(raw),
                record_size,
            )

        max_records = (len(raw) // record_size) if self._max_records is None else self._max_records
        offset = 0
        for _ in range(max_records):
            segment = raw[offset : offset + record_size]
            if len(segment) < record_size:
                break  # EOF
            zobrist = segment[:8]
            player_turn = segment[9:10]  # one after delimiter char
            board_bytes = segment[10:10 + NUM_INTERSECTIONS]

            # Decode board into float vector of length 81
            board = np.frombuffer(board_bytes, dtype="S1")
            try:
                board = np.vectorize(lambda b: CHAR_TO_VALUE[b])(board).astype(np.float32)
            except KeyError:
                # If any unknown char appears, skip this record for safety
                offset += record_size
                continue

            # Adjust sign depending on whose turn it is (replicates original logic)
            sign = -1 if player_turn == b"W" else 1
            score = score_lookup.get(zobrist)
            if score is None:
                offset += record_size
                continue  # Skip positions without known outcome

            boards.append(board)
            scores.append(sign * score)
            offset += record_size

        return np.stack(boards), np.asarray(scores, dtype=np.int32).reshape(-1, 1)

###############################################################################
# Model helpers                                                               #
###############################################################################

def build_linear_model(input_dim: int = NUM_INTERSECTIONS) -> tf.keras.Model:
    """Return a simple single‑layer logistic regression model (baseline)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(1, activation="sigmoid", name="logits"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e‑3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def build_cnn_model(input_shape: Tuple[int, int, int] = (BOARD_SIZE, BOARD_SIZE, 1)) -> tf.keras.Model:
    """Recreate the original two‑conv‑layer architecture in TF 2.x/Keras."""
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, kernel_size=5, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="go9_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e‑4),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

###############################################################################
# Convenience utilities                                                       #
###############################################################################

def suggest_move(model: tf.keras.Model, board: np.ndarray) -> int:
    """Return the index (0‑80) of the principal recommended move.

    The function tries every legal move (empty intersection), asks the model for
    a score, and returns the move with the highest predicted win probability.
    """
    if board.shape != (NUM_INTERSECTIONS,):
        raise ValueError("Board must be a flat (81,) vector in \u22121/0/1 encoding.")

    empty_points = np.where(board == 0)[0]
    if len(empty_points) == 0:
        raise ValueError("No legal moves on a full board.")

    boards_aug = np.tile(board, (len(empty_points), 1))
    boards_aug[np.arange(len(empty_points)), empty_points] = 1  # Assume current player = black

    preds = model.predict(boards_aug, verbose=0).flatten()
    best_idx = int(empty_points[np.argmax(preds)])
    return best_idx

###############################################################################
# CLI entry‑point                                                             #
###############################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or eval Go 9×9 models.")
    parser.add_argument("data_dir", type=str, help="Directory containing *.dat files")
    parser.add_argument("--cnn", action="store_true", help="Use CNN instead of linear model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Minibatch size")
    args = parser.parse_args()

    # 1) Data
    ds = Go9Dataset(args.data_dir)
    train_ds = ds.to_tf_dataset(batch_size=args.batch_size)

    # 2) Model
    model_fn = build_cnn_model if args.cnn else build_linear_model
    model = model_fn()

    model.summary()
    model.fit(train_ds, epochs=args.epochs)

    # 3) Demo inference on a blank board
    empty_board = np.zeros(NUM_INTERSECTIONS, dtype=np.float32)
    best_move = suggest_move(model, empty_board)
    print("Suggested opening move: (flat index)", best_move)
