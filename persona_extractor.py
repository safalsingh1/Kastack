"""
persona_extractor.py
--------------------
Extracts a structured persona from the conversation messages.

Categories:
  habits           – recurring behavioral patterns (sleep, food, exercise, etc.)
  personal_facts   – jobs, locations, family, relationships
  personality      – traits inferred from language
  communication_style – message length, emoji, punctuation, tone

All logic is local: regex + keyword matching + simple stats.
No external API or LLM used.
"""

import re
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Any

from data_processor import Message

PERSONA_OUT = Path(__file__).parent / "data" / "persona.json"

# ---------------------------------------------------------------------------
# Keyword libraries
# ---------------------------------------------------------------------------

HABIT_PATTERNS = {
    "early_riser": [r"\bwake up early\b", r"\bmorning person\b", r"\bup at \d", r"\bget up early\b"],
    "late_sleeper": [r"\bstay up late\b", r"\bnight owl\b", r"\bsleep in\b", r"\bup all night\b"],
    "exercises_regularly": [r"\bwork out\b", r"\bgym\b", r"\bjog\b", r"\brun(ning)?\b", r"\byoga\b", r"\bhike\b", r"\bbike\b", r"\bswim\b"],
    "reader": [r"\bread(ing)?\b", r"\bbook club\b", r"\bnovel\b", r"\blibrary\b"],
    "cook": [r"\bcook(ing)?\b", r"\bbake\b", r"\brecipe\b", r"\bmeal\b", r"\bchef\b"],
    "gamer": [r"\bvideo game\b", r"\bgaming\b", r"\bplaystation\b", r"\bxbox\b", r"\bpc game\b", r"\bskyrim\b", r"\bfallout\b"],
    "music_lover": [r"\bplay guitar\b", r"\bplay piano\b", r"\bsing\b", r"\bband\b", r"\bmusic\b"],
    "outdoorsy": [r"\bhike\b", r"\bcamping\b", r"\bfishing\b", r"\bnature\b", r"\boutsid\b", r"\bpark\b"],
    "social": [r"\bfriends\b", r"\bfamily\b", r"\bparty\b", r"\bget together\b", r"\bhang out\b"],
    "pet_owner": [r"\bmy dog\b", r"\bmy cat\b", r"\bpet\b", r"\bpuppy\b", r"\bkitten\b"],
    "traveller": [r"\btravel\b", r"\bvacation\b", r"\btrip\b", r"\bvisit\b", r"\bcountry\b", r"\bcity\b"],
    "movie_watcher": [r"\bwatch\b.*\bmovie\b", r"\bfilm\b", r"\bnetflix\b", r"\bcinema\b"],
}

JOB_PATTERNS = [
    (r"\b(software engineer|programmer|developer|coder)\b", "software engineer"),
    (r"\b(nurse|nursing)\b", "nurse"),
    (r"\b(teacher|teaching)\b", "teacher"),
    (r"\b(firefighter|fire fighter)\b", "firefighter"),
    (r"\b(doctor|physician)\b", "doctor"),
    (r"\b(chef|cook professionally)\b", "chef"),
    (r"\b(artist|muralist|painter)\b", "artist"),
    (r"\b(musician|guitarist|singer)\b", "musician"),
    (r"\b(student|studying)\b", "student"),
    (r"\b(writer|blogger|author)\b", "writer"),
    (r"\b(personal trainer|fitness coach)\b", "personal trainer"),
    (r"\b(park ranger)\b", "park ranger"),
    (r"\b(librarian)\b", "librarian"),
    (r"\b(barista)\b", "barista"),
    (r"\b(EMT|paramedic)\b", "EMT"),
    (r"\b(dental assistant)\b", "dental assistant"),
    (r"\b(stay.at.home mom|stay at home mom)\b", "stay-at-home mom"),
]

FAMILY_PATTERNS = [
    (r"\b(my kids?|my children)\b", "has_children"),
    (r"\bmy (son|daughter)\b", "has_children"),
    (r"\bmy (husband|wife|partner|boyfriend|girlfriend)\b", "in_relationship"),
    (r"\bsingle (mom|dad|parent)\b", "single_parent"),
    (r"\bmy (mom|dad|mother|father|parents)\b", "has_parents"),
    (r"\bmy (brother|sister|sibling)\b", "has_siblings"),
]

LOCATION_PATTERNS = [
    r"\bPortland\b", r"\bNew York\b", r"\bLos Angeles\b", r"\bChicago\b",
    r"\bTexas\b", r"\bCalifornia\b", r"\bFlorida\b", r"\bMichigan\b",
    r"\bGeorgia\b", r"\bOregon\b", r"\bMaldives\b", r"\bJapan\b",
]

PERSONALITY_SIGNALS = {
    "positive": [r"\bgreat\b", r"\bamazing\b", r"\bawesome\b", r"\blove\b", r"\bhappy\b", r"\bexcited\b", r"\bwonderful\b"],
    "empathetic": [r"\bsorry\b", r"\bunderstand\b", r"\bfeel\b", r"\bcare\b", r"\bsupport\b"],
    "humorous": [r"\bhaha\b", r"\blol\b", r"\bfunny\b", r"\bjoke\b", r"\blaugh\b"],
    "curious": [r"\bwonder\b", r"\bcurious\b", r"\binteresting\b", r"\blearn\b", r"\bfascinating\b"],
    "adventurous": [r"\badventure\b", r"\bexplore\b", r"\bnew experience\b", r"\btry new\b"],
    "family_oriented": [r"\bfamily\b", r"\bkids\b", r"\bparents\b", r"\bhome\b"],
    "ambitious": [r"\bgoal\b", r"\bachieve\b", r"\bdream\b", r"\baspire\b", r"\bwork hard\b"],
    "introverted": [r"\balone\b", r"\bquiet\b", r"\bsolitude\b", r"\bby myself\b"],
}

# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

def extract_habits(messages: List[Message]) -> List[str]:
    all_text = " ".join(m.text.lower() for m in messages)
    detected = []
    for habit, patterns in HABIT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, all_text):
                detected.append(habit.replace("_", " "))
                break
    return list(set(detected))


def extract_personal_facts(messages: List[Message]) -> Dict[str, Any]:
    all_text = " ".join(m.text for m in messages)
    lower_text = all_text.lower()

    facts: Dict[str, Any] = {}

    # Jobs
    job_counts: Counter = Counter()
    for pattern, label in JOB_PATTERNS:
        count = len(re.findall(pattern, lower_text, re.IGNORECASE))
        if count > 0:
            job_counts[label] += count
    if job_counts:
        facts["mentioned_occupations"] = [j for j, _ in job_counts.most_common(3)]

    # Family
    family_flags = set()
    for pattern, flag in FAMILY_PATTERNS:
        if re.search(pattern, all_text, re.IGNORECASE):
            family_flags.add(flag)
    if family_flags:
        facts["family"] = list(family_flags)

    # Locations
    locations = []
    for loc_pattern in LOCATION_PATTERNS:
        matches = re.findall(loc_pattern, all_text)
        locations.extend(matches)
    if locations:
        loc_counts = Counter(locations)
        facts["mentioned_locations"] = [loc for loc, _ in loc_counts.most_common(5)]

    # Pets
    pets = []
    for animal in ["dog", "cat", "horse", "bird", "fish", "rabbit", "turtle"]:
        if re.search(rf"\bmy {animal}\b", lower_text):
            pets.append(animal)
    if pets:
        facts["pets"] = list(set(pets))

    # Hobbies explicitly stated
    hobbies = []
    hobby_words = ["hiking", "fishing", "cooking", "reading", "gaming", "dancing",
                   "singing", "yoga", "photography", "gardening", "painting", "writing",
                   "cycling", "running", "swimming", "skiing", "surfing", "archery",
                   "juggling", "karaoke"]
    for h in hobby_words:
        if re.search(rf"\b{h}\b", lower_text):
            hobbies.append(h)
    if hobbies:
        facts["hobbies"] = hobbies

    return facts


def extract_personality(messages: List[Message]) -> List[str]:
    all_text = " ".join(m.text.lower() for m in messages)
    scores: Dict[str, int] = {}
    for trait, patterns in PERSONALITY_SIGNALS.items():
        count = sum(len(re.findall(p, all_text)) for p in patterns)
        if count >= 3:  # threshold: must appear at least 3 times
            scores[trait] = count

    # Sort by strength and return top traits
    sorted_traits = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
    return sorted_traits[:6]


def extract_communication_style(messages: List[Message]) -> Dict[str, Any]:
    if not messages:
        return {}

    texts = [m.text for m in messages]
    lengths = [len(t.split()) for t in texts]
    avg_len = sum(lengths) / len(lengths)

    # Emoji detection (basic unicode range)
    emoji_re = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        "\u2600-\u26FF\u2700-\u27BF]+",
        flags=re.UNICODE
    )
    emoji_msgs = sum(1 for t in texts if emoji_re.search(t))
    emoji_ratio = emoji_msgs / len(texts)

    # Question ratio
    question_msgs = sum(1 for t in texts if "?" in t)
    question_ratio = question_msgs / len(texts)

    # Exclamation ratio
    exclaim_msgs = sum(1 for t in texts if "!" in t)
    exclaim_ratio = exclaim_msgs / len(texts)

    # Capitalization (all-caps words)
    all_caps_words = sum(
        len([w for w in t.split() if w.isupper() and len(w) > 1])
        for t in texts
    )

    # Tone
    casual_markers = ["lol", "haha", "omg", "yeah", "yep", "nope", "gonna", "wanna", "kinda"]
    casual_count = sum(
        sum(1 for m in casual_markers if m in t.lower())
        for t in texts
    )
    tone = "casual" if casual_count > 10 else "formal"

    return {
        "avg_message_length_words": round(avg_len, 1),
        "emoji_usage": "frequent" if emoji_ratio > 0.2 else ("occasional" if emoji_ratio > 0.05 else "rare"),
        "emoji_ratio": round(emoji_ratio, 3),
        "question_ratio": round(question_ratio, 3),
        "exclamation_ratio": round(exclaim_ratio, 3),
        "tone": tone,
        "all_caps_words": all_caps_words,
    }


def extract_topics_of_interest(messages: List[Message]) -> List[str]:
    """Frequent meaningful keywords across all messages."""
    stopwords = set([
        "the", "and", "for", "that", "this", "with", "are", "was", "you",
        "your", "have", "but", "not", "its", "they", "them", "their", "from",
        "just", "all", "like", "been", "would", "could", "should", "what",
        "when", "how", "who", "where", "which", "will", "one", "any", "more",
        "also", "get", "got", "can", "yes", "yeah", "okay", "know", "think",
        "love", "really", "don't", "i'm", "it's", "that's", "i've", "i'll",
        "there", "do", "did", "does", "she", "him", "her", "too", "very",
        "well", "going", "about", "some", "had", "has", "our", "out", "use",
        "way", "may", "now", "want", "tell", "said", "ask", "say", "make",
        "made", "come", "came", "take", "look", "see", "feel", "mean", "try",
        "good", "great", "nice", "sure", "glad", "hope", "thank", "thanks",
        "hello", "hey", "bye", "hi", "sounds", "lot", "much", "bit", "i'll",
        "we're", "it's", "don't", "that's", "i've", "let", "let's",
    ])
    all_words = []
    for m in messages:
        words = re.findall(r"\b[a-zA-Z]{4,}\b", m.text.lower())
        all_words.extend([w for w in words if w not in stopwords])
    return [w for w, _ in Counter(all_words).most_common(15)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract_persona(messages: List[Message]) -> Dict[str, Any]:
    print(f"[persona_extractor] Analysing {len(messages)} messages...")
    persona = {
        "total_messages_analysed": len(messages),
        "habits": extract_habits(messages),
        "personal_facts": extract_personal_facts(messages),
        "personality_traits": extract_personality(messages),
        "communication_style": extract_communication_style(messages),
        "topics_of_interest": extract_topics_of_interest(messages),
    }
    return persona


def save_persona(persona: Dict[str, Any], out_path: Path = PERSONA_OUT):
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(persona, f, indent=2)
    print(f"[persona_extractor] Saved persona to {out_path}")


def load_persona(path: Path = PERSONA_OUT) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    from data_processor import load_messages
    msgs = load_messages()
    persona = extract_persona(msgs)
    save_persona(persona)
    import json
    print(json.dumps(persona, indent=2))
