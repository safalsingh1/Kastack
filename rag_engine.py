"""
rag_engine.py
-------------
Retrieval-Augmented Generation engine using:
  • FAISS for vector retrieval  (sentence-transformers / all-MiniLM-L6-v2)
  • Groq API for LLM answer generation  (llama-3.3-70b-versatile)

Two FAISS indexes:
  checkpoint_index  – one vector per checkpoint summary (topic + 100-msg)
  message_index     – one vector per sampled message

Query pipeline:
  embed(query)
    → top-K checkpoints + top-K messages
    → build context string
    → Groq LLM generates a grounded answer
"""

import os
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

DATA_DIR          = Path(__file__).parent / "data"
TOPIC_INDEX_PATH  = DATA_DIR / "topic_index.faiss"
TOPIC_META_PATH   = DATA_DIR / "topic_index_meta.json"
MSG_INDEX_PATH    = DATA_DIR / "msg_index.faiss"
MSG_META_PATH     = DATA_DIR / "msg_index_meta.json"

EMBED_MODEL   = "all-MiniLM-L6-v2"
GROQ_MODEL    = "llama-3.3-70b-versatile"          # fastest Groq model
MSG_SAMPLE_EVERY = 5   # index every Nth message

SYSTEM_PROMPT = """You are ConvoRAG, an intelligent conversation analyst.
You are given:
  1. Relevant topic/checkpoint summaries extracted from a large conversation dataset.
  2. Relevant individual message snippets.

Your job is to answer the user's question accurately and concisely, grounded only in the provided context.
If the context doesn't contain enough information, say so honestly.
Format your answer clearly. Use bullet points or bold text where helpful.
Never make up information not present in the context."""


# ─────────────────────────────────────────────────────────────────────────────
# Index builders  (called once from build_index.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_checkpoint_index(checkpoints: List[Dict], model: SentenceTransformer) -> None:
    texts  = [cp["summary"] for cp in checkpoints]
    ids    = [cp["id"]      for cp in checkpoints]
    labels = [cp["label"]   for cp in checkpoints]
    types  = [cp["type"]    for cp in checkpoints]

    print(f"[rag_engine] Embedding {len(texts)} checkpoint summaries…")
    embeddings = model.encode(texts, normalize_embeddings=True,
                              show_progress_bar=True, batch_size=256)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)          # cosine similarity on unit vecs
    index.add(embeddings)
    faiss.write_index(index, str(TOPIC_INDEX_PATH))

    meta = [
        {"idx": k, "cp_id": ids[k], "label": labels[k],
         "type": types[k], "summary": texts[k]}
        for k in range(len(texts))
    ]
    with open(TOPIC_META_PATH, "w") as f:
        json.dump(meta, f)
    print(f"[rag_engine] Checkpoint index saved  ({len(texts)} vectors)")


def build_message_index(messages: List[Dict], model: SentenceTransformer) -> None:
    sampled = [m for i, m in enumerate(messages) if i % MSG_SAMPLE_EVERY == 0]
    texts   = [m["text"]      for m in sampled]
    gids    = [m["global_id"] for m in sampled]
    convs   = [m["conv_id"]   for m in sampled]

    print(f"[rag_engine] Embedding {len(texts)} message samples…")
    embeddings = model.encode(texts, normalize_embeddings=True,
                              show_progress_bar=True, batch_size=512)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(MSG_INDEX_PATH))

    meta = [
        {"idx": i, "global_id": gids[i], "conv_id": convs[i], "text": texts[i]}
        for i in range(len(texts))
    ]
    with open(MSG_META_PATH, "w") as f:
        json.dump(meta, f)
    print(f"[rag_engine] Message index saved  ({len(texts)} vectors)")


# ─────────────────────────────────────────────────────────────────────────────
# RAG Engine
# ─────────────────────────────────────────────────────────────────────────────

class RAGEngine:
    def __init__(self):
        self._model:      Optional[SentenceTransformer] = None
        self._cp_index:   Optional[faiss.Index]         = None
        self._msg_index:  Optional[faiss.Index]         = None
        self._cp_meta:    List[Dict] = []
        self._msg_meta:   List[Dict] = []
        self._groq_client = None
        self._loaded = False

    # ── bootstrap ────────────────────────────────────────────────────────────
    def load(self) -> None:
        if self._loaded:
            return

        print("[rag_engine] Loading sentence-transformer model…")
        self._model = SentenceTransformer(EMBED_MODEL)

        print("[rag_engine] Loading FAISS indexes…")
        self._cp_index  = faiss.read_index(str(TOPIC_INDEX_PATH))
        self._msg_index = faiss.read_index(str(MSG_INDEX_PATH))

        with open(TOPIC_META_PATH) as f:
            self._cp_meta = json.load(f)
        with open(MSG_META_PATH) as f:
            self._msg_meta = json.load(f)

        # Groq client — only init if key looks real (not placeholder)
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        is_real_key = api_key and not api_key.startswith("your_") and len(api_key) > 20
        if is_real_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=api_key)
                print(f"[rag_engine] Groq client ready  (model: {GROQ_MODEL})")
            except Exception as e:
                print(f"[rag_engine] WARNING: Groq init failed ({e}). Using template fallback.")
                self._groq_client = None
        else:
            print("[rag_engine] INFO: No valid GROQ_API_KEY found. Using template answers.")
            print("[rag_engine] Set GROQ_API_KEY in .env to enable LLM-powered answers.")

        self._loaded = True
        print("[rag_engine] Ready.")

    # ── embedding ─────────────────────────────────────────────────────────────
    def _embed(self, text: str) -> np.ndarray:
        return self._model.encode(
            [text], normalize_embeddings=True
        ).astype("float32")

    # ── retrieval ─────────────────────────────────────────────────────────────
    def retrieve_checkpoints(self, query: str, top_k: int = 5) -> List[Dict]:
        scores, indices = self._cp_index.search(self._embed(query), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self._cp_meta[idx])
            item["score"] = round(float(score), 4)
            results.append(item)
        return results

    def retrieve_messages(self, query: str, top_k: int = 8) -> List[Dict]:
        scores, indices = self._msg_index.search(self._embed(query), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self._msg_meta[idx])
            item["score"] = round(float(score), 4)
            results.append(item)
        return results

    # ── context builder ───────────────────────────────────────────────────────
    def _build_context(self, cp_results: List[Dict], msg_results: List[Dict]) -> str:
        parts = []
        if cp_results:
            parts.append("### Topic / Checkpoint Summaries")
            for r in cp_results:
                parts.append(f"[{r['label']}] {r['summary']}")
        if msg_results:
            parts.append("\n### Relevant Message Snippets")
            for r in msg_results:
                parts.append(f'  • (msg {r["global_id"]}) "{r["text"]}"')
        return "\n".join(parts)

    # ── Groq LLM call ─────────────────────────────────────────────────────────
    def _groq_answer(self, query: str, context: str) -> str:
        user_msg = (
            f"Context from conversation analysis:\n{context}\n\n"
            f"Question: {query}"
        )
        completion = self._groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=700,
        )
        return completion.choices[0].message.content.strip()

    # ── template fallback (no API key) ────────────────────────────────────────
    def _template_answer(self, query: str, cp_results: List, msg_results: List) -> str:
        if not cp_results and not msg_results:
            return "I couldn't find relevant information in the conversation history."

        parts = [f'Based on the conversation history for **"{query}"**:\n']
        if cp_results:
            top = cp_results[0]
            parts.append(f"📌 **{top['label']}**\n{top['summary']}")
        if msg_results:
            parts.append("\n💬 **Relevant snippets:**")
            for r in msg_results[:3]:
                parts.append(f'  • "{r["text"]}"')
        return "\n".join(parts)

    # ── persona shortcuts (bypass LLM for direct persona facts) ───────────────
    def _persona_field(self, persona: Dict, field: str) -> Optional[Any]:
        val = persona
        for part in field.split("."):
            val = val.get(part) if isinstance(val, dict) else None
            if val is None:
                return None
        return val

    def _persona_direct(self, query: str, persona: Dict, field: str) -> Dict:
        val = self._persona_field(persona, field)
        if not val or (isinstance(val, (list, dict)) and not val):
            text = "I don't have enough data about that aspect from the conversations."
        elif isinstance(val, list):
            text = "Based on the conversations:\n• " + "\n• ".join(str(v) for v in val)
        elif isinstance(val, dict):
            text = "Based on the conversations:\n" + "\n".join(
                f"• **{k.replace('_',' ').title()}**: {v}" for k, v in val.items()
            )
        else:
            text = f"Based on the conversations: {val}"
        return {"answer": text, "sources": {"persona_field": field}}

    def _persona_full(self, persona: Dict) -> Dict:
        p = []
        traits = persona.get("personality_traits", [])
        habits = persona.get("habits", [])
        facts  = persona.get("personal_facts", {})
        style  = persona.get("communication_style", {})
        topics = persona.get("topics_of_interest", [])

        if traits:   p.append(f"**Personality:** {', '.join(traits)}")
        if habits:   p.append(f"**Habits:** {', '.join(habits)}")
        if facts.get("mentioned_occupations"):
            p.append(f"**Occupations:** {', '.join(facts['mentioned_occupations'])}")
        if facts.get("hobbies"):
            p.append(f"**Hobbies:** {', '.join(facts['hobbies'])}")
        if facts.get("pets"):
            p.append(f"**Pets:** {', '.join(facts['pets'])}")
        if facts.get("family"):
            p.append(f"**Family signals:** {', '.join(facts['family'])}")
        if style:
            p.append(
                f"**Communication:** {style.get('tone','?')} tone · "
                f"avg {style.get('avg_message_length_words','?')} words/msg · "
                f"emoji {style.get('emoji_usage','?')}"
            )
        if topics:
            p.append(f"**Top topics:** {', '.join(topics[:10])}")

        return {
            "answer": "\n".join(p) if p else "No persona data available.",
            "sources": {"persona_field": "full"}
        }

    # ── main entry point ──────────────────────────────────────────────────────
    def answer(self, query: str, persona: Optional[Dict] = None) -> Dict[str, Any]:
        self.load()
        q = query.lower()

        # ── Persona shortcut routing ──
        # Only intercept DIRECT questions about the persona itself.
        # Questions that include "conversations", "people", "talk about X",
        # "discuss", "mention" are open RAG queries — let them through.
        RAG_SIGNALS = ["conversation", "people", "discuss", "mention", "talk about",
                       "said", "bring up", "come up", "in the", "from the",
                       "do they talk", "what do they say", "what topics"]
        is_rag_query = any(sig in q for sig in RAG_SIGNALS)

        if persona and not is_rag_query:
            # Habits
            if any(k in q for k in ["habit", "routine", "daily routine", "sleep", "eating", "exercise habit"]):
                return self._persona_direct(query, persona, "habits")
            # Communication style — only when HOW they communicate is asked
            if any(k in q for k in ["communication style", "how do they communicat", "how do they talk",
                                     "how do they speak", "message style", "writing style",
                                     "tone of", "use emoji", "emoji usage"]):
                return self._persona_direct(query, persona, "communication_style")
            # Personality
            if any(k in q for k in ["personality", "kind of person", "what kind of person",
                                     "character", "personality trait", "who is this user",
                                     "describe this user", "what type of person"]):
                if any(k in q for k in ["full", "overall", "everything", "all about", "complete"]):
                    return self._persona_full(persona)
                return self._persona_direct(query, persona, "personality_traits")
            # Jobs
            if any(k in q for k in ["occupation", "career", "profession",
                                     "what is their job", "what do they do for work",
                                     "what do they do for a living"]):
                return self._persona_direct(query, persona, "personal_facts.mentioned_occupations")
            # Family
            if any(k in q for k in ["their family", "do they have kids", "are they married",
                                     "their relationship", "their partner", "their children"]):
                return self._persona_direct(query, persona, "personal_facts.family")
            # Locations
            if any(k in q for k in ["where do they live", "their location", "what city",
                                     "where are they from", "where do they come from"]):
                return self._persona_direct(query, persona, "personal_facts.mentioned_locations")
            # Pets
            if any(k in q for k in ["their pet", "do they have a dog", "do they have a cat",
                                     "what pets", "their animals"]):
                return self._persona_direct(query, persona, "personal_facts.pets")
            # Full overview
            if any(k in q for k in ["overall summary", "full summary", "everything about this user",
                                     "tell me about this user", "give me a full picture"]):
                return self._persona_full(persona)

        # ── RAG retrieval ──
        cp_results  = self.retrieve_checkpoints(query, top_k=5)
        msg_results = self.retrieve_messages(query,    top_k=8)
        context     = self._build_context(cp_results, msg_results)

        # ── LLM answer ──
        if self._groq_client:
            try:
                answer_text = self._groq_answer(query, context)
            except Exception as e:
                print(f"[rag_engine] Groq error: {e}. Falling back to template.")
                answer_text = self._template_answer(query, cp_results, msg_results)
        else:
            answer_text = self._template_answer(query, cp_results, msg_results)

        return {
            "answer": answer_text,
            "sources": {
                "checkpoints": cp_results,
                "messages":    msg_results,
            }
        }


# ── Singleton ──────────────────────────────────────────────────────────────────
_engine: Optional[RAGEngine] = None

def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine


# ── CLI smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from persona_extractor import load_persona
    engine  = get_engine()
    engine.load()
    persona = load_persona()

    questions = [
        "What kind of person is this user?",
        "What are their main habits?",
        "How do they communicate?",
        "What hobbies come up most often in the conversations?",
        "Do they talk about sports?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        res = engine.answer(q, persona=persona)
        print(f"A: {res['answer'][:400]}")
