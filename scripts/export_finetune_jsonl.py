import json
import argparse
from pathlib import Path

def to_ft_example(history):
    # Expect: list of {'role': 'user'|'assistant', 'content': '...'}
    messages = []
    for m in history:
        role = "user" if m["role"] == "user" else "assistant"
        messages.append({"role": role, "content": m["content"]})
    return {"messages": messages}

def main(src_json, out_jsonl):
    src = Path(src_json)
    out = Path(out_jsonl)
    data = json.loads(src.read_text())
    # data is list[history]; each history is list of messages
    with out.open("w") as f:
        for hist in data:
            f.write(json.dumps(to_ft_example(hist)) + "\n")
    print(f"Wrote {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_json", required=True, help="JSON file with a list of chat histories")
    ap.add_argument("--out_jsonl", default="finetune_dataset.jsonl")
    args = ap.parse_args()
    main(args.src_json, args.out_jsonl)