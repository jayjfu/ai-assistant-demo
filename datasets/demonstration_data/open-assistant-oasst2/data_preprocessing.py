import os
import json
import argparse


# ChatML special tokens
IM_START = "<|im_start|>"
IM_END   = "<|im_end|>"

# Configurable system prompt (set to None to omit)
SYSTEM_PROMPT = "You are a helpful assistant."

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Convert OA2 trees.jsonl → ChatML format")
parser.add_argument("--input_file", default="./2023-11-05_oasst2_ready.trees.jsonl", type=str)
parser.add_argument('--output_dir', default="./processed", type=str)
parser.add_argument("--output_file", default="oasst2_chatml.jsonl", type=str)
args = parser.parse_args()

def dfs_collect(node, msgs):
    role_mapping = {"prompter": 'user', "assistant": 'assistant'}
    msgs.append({"role": role_mapping[node['role']], "content": node['text']})
    for child in node.get('replies', []):
        dfs_collect(child, msgs)

def tree_to_chatml(tree) -> str:
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": 'system', "content": SYSTEM_PROMPT})
    dfs_collect(tree['prompt'], messages)

    chatml_blocks = []
    for m in messages:
        chatml_blocks.append(f"{IM_START}{m['role']}\n{m['content']}{IM_END}")
    return "\n".join(chatml_blocks) + "\n"

def main():
    os.makedirs(os.path.join(SCRIPT_DIR, args.output_dir), exist_ok=True)

    with (open(args.input_file, 'r') as f_in, \
          open(os.path.join(SCRIPT_DIR, args.output_dir, args.output_file), 'w') as f_out):

        for line_no, line in enumerate(f_in, 1):
            line = line.strip()
            tree = json.loads(line)

            chatml = tree_to_chatml(tree)

            json.dump({"text": chatml}, f_out, ensure_ascii=False)
            f_out.write("\n")

if __name__ == "__main__":
    main()