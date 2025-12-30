import argparse
import os
from transformers import AutoTokenizer, pipeline


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Test the dpo model")
parser.add_argument('--template_model', default='unsloth/Llama-3.2-1B-Instruct', type=str)
parser.add_argument('--saved_model', default="../checkpoints/Llama-3.2-1B-trl-dpo-model/checkpoint-30568", type=str)
parser.add_argument('--prompt', default="How can I build a daily drawing habit?", type=str)
args = parser.parse_args()

def main():
    tokenizer = AutoTokenizer.from_pretrained(args.template_model)

    model_path = os.path.join(str(SCRIPT_DIR), args.saved_model)

    pipe = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
    )

    prompt = args.prompt
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    outputs = pipe(input_text, max_new_tokens=512, do_sample=True, temperature=0.7)
    print(outputs[0]["generated_text"])

if __name__ == "__main__":
    main()