"""
build_index.py
--------------
One-time script to process conversations.csv and build all data artifacts:
  1. Parse CSV → messages.json
  2. Detect topics → checkpoints.json
  3. Build FAISS indexes
  4. Extract persona → persona.json

Run: python build_index.py

Subsequent runs are fast because embeddings are cached in data/embeddings.npy.
To force a full rebuild, delete the data/ directory.
"""

import sys
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def main():
    start = time.time()

    print("=" * 60)
    print("  ConvoRAG - Building Index")
    print("=" * 60)

    # Step 1: Parse CSV
    print("\n[Step 1/5] Parsing conversations.csv...")
    from data_processor import parse_csv, save_messages
    messages = parse_csv()
    save_messages(messages)
    print(f"  -> {len(messages):,} messages parsed")

    # Step 2: Detect topics
    print("\n[Step 2/5] Detecting topic boundaries...")
    from topic_detector import detect_topics
    embeddings, topic_segments = detect_topics(messages)
    print(f"  -> {len(topic_segments)} topic segments detected")

    # Step 3: Build checkpoints
    print("\n[Step 3/5] Building checkpoints...")
    from checkpoint_manager import build_all_checkpoints, save_checkpoints
    checkpoints = build_all_checkpoints(messages, topic_segments)
    save_checkpoints(checkpoints)
    topic_cps = [c for c in checkpoints if c["type"] == "topic"]
    hundred_cps = [c for c in checkpoints if c["type"] == "hundred"]
    print(f"  -> {len(topic_cps)} topic checkpoints, {len(hundred_cps)} hundred-msg checkpoints")

    # Step 4: Build FAISS indexes
    print("\n[Step 4/5] Building FAISS indexes...")
    from sentence_transformers import SentenceTransformer
    from rag_engine import build_checkpoint_index, build_message_index
    from data_processor import load_messages

    model = SentenceTransformer("all-MiniLM-L6-v2")
    messages_raw = [
        {"global_id": m.global_id, "conv_id": m.conv_id,
         "local_id": m.local_id, "speaker": m.speaker, "text": m.text}
        for m in messages
    ]
    build_checkpoint_index(checkpoints, model)
    build_message_index(messages_raw, model)

    # Step 5: Extract persona
    print("\n[Step 5/5] Extracting user persona...")
    from persona_extractor import extract_persona, save_persona
    persona = extract_persona(messages)
    save_persona(persona)
    print(f"  -> Persona traits: {persona.get('personality_traits', [])}")
    print(f"  -> Habits: {persona.get('habits', [])}")

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  [DONE] Build complete in {elapsed:.1f}s")
    print(f"  Files saved in: {DATA_DIR}")
    print(f"{'=' * 60}")
    print("\nTo start the chatbot: python app.py")


if __name__ == "__main__":
    main()
