"""
checkpoint_manager.py
---------------------
Creates two types of checkpoints from the message stream:

1. TOPIC CHECKPOINTS: one per detected topic segment
   { id, type:"topic", topic_id, start_msg, end_msg, msg_count, keywords, summary }

2. 100-MESSAGE CHECKPOINTS: one every 100 global messages (independent of topics)
   { id, type:"hundred", checkpoint_num, start_msg, end_msg, summary }

Summaries use an extractive approach: pick the most representative sentences
using TF-IDF scores (no external API needed).
"""

import json
import math
import re
import string
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Any

from data_processor import Message

CHECKPOINTS_OUT = Path(__file__).parent / "data" / "checkpoints.json"
HUNDRED_INTERVAL = 100


# ---------------------------------------------------------------------------
# Extractive summariser (TF-IDF sentence ranking)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [w for w in text.split() if len(w) > 2]


STOPWORDS = set([
    "the", "and", "for", "that", "this", "with", "are", "was", "you",
    "your", "have", "but", "not", "its", "they", "them", "their", "from",
    "just", "all", "like", "been", "would", "could", "should", "what",
    "when", "how", "who", "where", "which", "will", "one", "any", "more",
    "also", "get", "got", "can", "yes", "yeah", "okay", "know", "think",
    "love", "really", "don't", "i'm", "it's", "that's", "i've", "i'll",
    "we're", "there", "here", "do", "did", "does", "she", "him", "her",
    "too", "very", "well", "going", "about", "some", "been", "had",
    "has", "him", "his", "our", "out", "use", "way", "may", "now",
    "want", "tell", "said", "ask", "say", "make", "made", "come", "came",
    "take", "look", "see", "know", "think", "feel", "mean", "try", "let",
    "good", "great", "nice", "sure", "glad", "hope", "thank", "thanks",
    "hello", "hey", "bye", "hi", "sounds", "lot", "much", "bit"
])


def _tf_idf_sentences(sentences: List[str], top_n: int = 3) -> List[str]:
    if not sentences:
        return []
    if len(sentences) <= top_n:
        return sentences

    # TF per sentence
    tf = []
    for s in sentences:
        tokens = [t for t in _tokenize(s) if t not in STOPWORDS]
        freq = Counter(tokens)
        total = max(len(tokens), 1)
        tf.append({w: c / total for w, c in freq.items()})

    # IDF across sentences
    df: Counter = Counter()
    for tfs in tf:
        for w in tfs:
            df[w] += 1
    n = len(sentences)
    idf = {w: math.log((n + 1) / (cnt + 1)) for w, cnt in df.items()}

    # Score each sentence
    scores = []
    for tfs in tf:
        score = sum(tfs[w] * idf.get(w, 0) for w in tfs)
        scores.append(score)

    # Pick top_n by score
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_indices = sorted([i for i, _ in ranked[:top_n]])
    return [sentences[i] for i in top_indices]


def summarise(messages: List[Message], top_n: int = 4) -> str:
    """Generate an extractive summary from a list of messages."""
    # Collect all message texts as sentences
    sentences = [m.text for m in messages if len(m.text.split()) >= 5]
    if not sentences:
        sentences = [m.text for m in messages]
    top = _tf_idf_sentences(sentences, top_n=top_n)
    return " | ".join(top) if top else (messages[0].text if messages else "")


def extract_keywords(messages: List[Message], top_n: int = 8) -> List[str]:
    all_tokens = []
    for m in messages:
        all_tokens.extend([t for t in _tokenize(m.text) if t not in STOPWORDS])
    return [w for w, _ in Counter(all_tokens).most_common(top_n)]


# ---------------------------------------------------------------------------
# Checkpoint builders
# ---------------------------------------------------------------------------

def build_topic_checkpoints(
    messages: List[Message],
    topic_segments: List[Tuple[int, int]]
) -> List[Dict[str, Any]]:
    """
    Build one checkpoint per topic segment.
    topic_segments: list of (start_idx, end_idx) into messages list.
    """
    checkpoints = []
    for topic_id, (start_idx, end_idx) in enumerate(topic_segments):
        seg_msgs = messages[start_idx:end_idx + 1]
        if not seg_msgs:
            continue
        summary = summarise(seg_msgs, top_n=4)
        keywords = extract_keywords(seg_msgs, top_n=8)
        cp = {
            "id": f"topic_{topic_id}",
            "type": "topic",
            "topic_id": topic_id,
            "start_msg": messages[start_idx].global_id,
            "end_msg": messages[end_idx].global_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "msg_count": len(seg_msgs),
            "keywords": keywords,
            "summary": summary,
            # Human-readable range label
            "label": f"Topic {topic_id + 1} (msgs {messages[start_idx].global_id}–{messages[end_idx].global_id})"
        }
        checkpoints.append(cp)
    print(f"[checkpoint_manager] Built {len(checkpoints)} topic checkpoints")
    return checkpoints


def build_hundred_checkpoints(messages: List[Message]) -> List[Dict[str, Any]]:
    """
    Build one checkpoint every 100 messages (by global order / index).
    """
    checkpoints = []
    n = len(messages)
    checkpoint_num = 0

    for start_idx in range(0, n, HUNDRED_INTERVAL):
        end_idx = min(start_idx + HUNDRED_INTERVAL - 1, n - 1)
        seg_msgs = messages[start_idx:end_idx + 1]
        summary = summarise(seg_msgs, top_n=3)
        keywords = extract_keywords(seg_msgs, top_n=6)
        cp = {
            "id": f"hundred_{checkpoint_num}",
            "type": "hundred",
            "checkpoint_num": checkpoint_num,
            "start_msg": messages[start_idx].global_id,
            "end_msg": messages[end_idx].global_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "msg_count": len(seg_msgs),
            "keywords": keywords,
            "summary": summary,
            "label": f"Messages {messages[start_idx].global_id}–{messages[end_idx].global_id}"
        }
        checkpoints.append(cp)
        checkpoint_num += 1

    print(f"[checkpoint_manager] Built {len(checkpoints)} hundred-message checkpoints")
    return checkpoints


def build_all_checkpoints(
    messages: List[Message],
    topic_segments: List[Tuple[int, int]]
) -> List[Dict[str, Any]]:
    topic_cps = build_topic_checkpoints(messages, topic_segments)
    hundred_cps = build_hundred_checkpoints(messages)
    all_cps = topic_cps + hundred_cps
    return all_cps


def save_checkpoints(checkpoints: List[Dict[str, Any]], out_path: Path = CHECKPOINTS_OUT):
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(checkpoints, f, indent=2)
    print(f"[checkpoint_manager] Saved {len(checkpoints)} checkpoints to {out_path}")


def load_checkpoints(path: Path = CHECKPOINTS_OUT) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    from data_processor import load_messages
    from topic_detector import detect_topics

    msgs = load_messages()
    _, segments = detect_topics(msgs)
    cps = build_all_checkpoints(msgs, segments)
    save_checkpoints(cps)

    # Preview
    topic_cps = [c for c in cps if c["type"] == "topic"]
    for cp in topic_cps[:3]:
        print(f"\n{cp['label']} ({cp['msg_count']} msgs)")
        print(f"  Keywords: {cp['keywords']}")
        print(f"  Summary: {cp['summary'][:120]}...")
