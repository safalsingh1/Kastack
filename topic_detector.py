"""
topic_detector.py
-----------------
Detects topic boundaries in a stream of messages using sentence embeddings
and cosine similarity on a sliding window.

Algorithm:
  1. Embed every message text with sentence-transformers (all-MiniLM-L6-v2)
  2. Use a sliding window of WINDOW_SIZE messages
  3. Compute cosine similarity between consecutive windows
  4. When similarity drops below THRESHOLD → new topic boundary

Returns: list of (start_global_id, end_global_id) ranges per topic
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from data_processor import Message, load_messages
from tqdm import tqdm

WINDOW_SIZE = 5        # messages per window for comparison
THRESHOLD = 0.40       # cosine similarity below this = topic change
MIN_TOPIC_MSGS = 8     # minimum messages to form its own topic
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_CACHE = Path(__file__).parent / "data" / "embeddings.npy"


def load_or_compute_embeddings(messages: List[Message], model: SentenceTransformer) -> np.ndarray:
    if EMBEDDINGS_CACHE.exists():
        print("[topic_detector] Loading cached embeddings...")
        return np.load(str(EMBEDDINGS_CACHE))

    print(f"[topic_detector] Computing embeddings for {len(messages)} messages...")
    texts = [m.text for m in messages]

    # Batch encode in chunks to avoid memory issues
    batch_size = 512
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings)
    np.save(str(EMBEDDINGS_CACHE), embeddings)
    print(f"[topic_detector] Saved embeddings to {EMBEDDINGS_CACHE}")
    return embeddings


def compute_window_embeddings(embeddings: np.ndarray, window_size: int) -> np.ndarray:
    """Average-pool message embeddings into windows."""
    n = len(embeddings)
    window_embs = []
    for i in range(0, n, window_size):
        chunk = embeddings[i:i + window_size]
        window_embs.append(chunk.mean(axis=0))
    return np.array(window_embs)


def detect_topic_boundaries(messages: List[Message], embeddings: np.ndarray) -> List[int]:
    """
    Returns a list of global message indices where a new topic starts.
    Index 0 is always included (first topic start).
    """
    n = len(messages)
    window_embs = compute_window_embeddings(embeddings, WINDOW_SIZE)

    boundaries = [0]  # always start a topic at message 0

    for i in range(1, len(window_embs)):
        sim = cosine_similarity(
            window_embs[i - 1].reshape(1, -1),
            window_embs[i].reshape(1, -1)
        )[0][0]

        if sim < THRESHOLD:
            # The boundary is at the start of window i
            boundary_msg_idx = i * WINDOW_SIZE
            if boundary_msg_idx < n:
                # Merge tiny topic segments
                if boundary_msg_idx - boundaries[-1] >= MIN_TOPIC_MSGS:
                    boundaries.append(boundary_msg_idx)

    return boundaries


def get_topic_segments(messages: List[Message], boundaries: List[int]) -> List[Tuple[int, int]]:
    """
    Convert boundary indices into (start_global_id, end_global_id) pairs.
    Returns list of (start_idx, end_idx) into the messages list.
    """
    segments = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(messages)
        segments.append((start, end - 1))  # inclusive end
    return segments


def detect_topics(messages: List[Message]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Main entry point. Returns (embeddings, topic_segments).
    topic_segments: list of (start_idx, end_idx) in messages list.
    """
    model = SentenceTransformer(MODEL_NAME)
    embeddings = load_or_compute_embeddings(messages, model)
    boundaries = detect_topic_boundaries(messages, embeddings)
    segments = get_topic_segments(messages, boundaries)
    print(f"[topic_detector] Detected {len(segments)} topic segments")
    return embeddings, segments


if __name__ == "__main__":
    messages = load_messages()
    embeddings, segments = detect_topics(messages)
    print(f"First 5 topic segments:")
    for i, (s, e) in enumerate(segments[:5]):
        print(f"  Topic {i+1}: msgs {s}–{e} ({e - s + 1} messages)")
        print(f"    First: {messages[s].text[:60]}")
        print(f"    Last:  {messages[e].text[:60]}")
