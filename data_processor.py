"""
data_processor.py
-----------------
Parses conversations.csv and produces a flat, chronological list of Message objects.

CSV format: each row is a multi-line conversation (quoted), e.g.
  "User 1: Hi!\nUser 2: Hello!\n..."

Each message in the conversation is a line starting with "User 1:" or "User 2:".
"""

import csv
import re
import json
from dataclasses import dataclass, asdict
from typing import List
from pathlib import Path

CSV_PATH = Path(__file__).parent / "conversations.csv"
MESSAGES_OUT = Path(__file__).parent / "data" / "messages.json"


@dataclass
class Message:
    global_id: int      # Monotonically increasing across all conversations
    conv_id: int        # Which conversation (row) this came from
    local_id: int       # Position within the conversation
    speaker: str        # "User 1" or "User 2"
    text: str           # Message content


def parse_csv(csv_path: Path = CSV_PATH) -> List[Message]:
    """
    Read the CSV file and return a flat chronological list of Message objects.
    """
    messages: List[Message] = []
    global_id = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for conv_id, row in enumerate(reader):
            if not row:
                continue
            # The whole conversation is in row[0] (single-column CSV)
            conversation_text = row[0]
            lines = conversation_text.split("\n")

            local_id = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Match "User 1: ..." or "User 2: ..."
                match = re.match(r"^(User \d+):\s*(.+)", line)
                if match:
                    speaker = match.group(1)
                    text = match.group(2).strip()
                    if text:
                        messages.append(Message(
                            global_id=global_id,
                            conv_id=conv_id,
                            local_id=local_id,
                            speaker=speaker,
                            text=text
                        ))
                        global_id += 1
                        local_id += 1

    return messages


def save_messages(messages: List[Message], out_path: Path = MESSAGES_OUT):
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(m) for m in messages], f)
    print(f"[data_processor] Saved {len(messages)} messages to {out_path}")


def load_messages(path: Path = MESSAGES_OUT) -> List[Message]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Message(**d) for d in data]


if __name__ == "__main__":
    msgs = parse_csv()
    save_messages(msgs)
    print(f"Total messages: {len(msgs)}")
    # Quick preview
    for m in msgs[:5]:
        print(f"  [{m.global_id}] conv={m.conv_id} {m.speaker}: {m.text[:60]}")
