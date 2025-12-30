import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from trl import DPOTrainer, DPOConfig


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="direct preference optimization (DPO)")
parser.add_argument('--base_model', default='unsloth/Llama-3.2-1B', type=str)
parser.add_argument('--template_model', default='unsloth/Llama-3.2-1B-Instruct', type=str)
parser.add_argument('--sft_model',
                    default="./checkpoints/Llama-3.2-1B-fine-tuned-model/checkpoint-6928", type=str)
parser.add_argument('--hf_dataset_name', default='HuggingFaceH4/ultrafeedback_binarized', type=str)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--max_length', default=512, type=int)
parser.add_argument('--logging_steps', default=1_000, type=int)
parser.add_argument('--save_steps', default=5_000, type=int)
parser.add_argument('--eval_steps', default=2_000, type=int)
parser.add_argument('--no_resume', action='store_true')
parser.add_argument('--output_dir', default="./checkpoints/", type=str)
parser.add_argument('--output_name', default="trl-dpo-model", type=str)
args = parser.parse_args()

def train(args):
    dataset = datasets.load_dataset(args.hf_dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(args.template_model)

    def apply_chat_template(example):
        prompt, chosen, rejected = example['prompt'], example['chosen'], example['rejected']

        user_msg = [{"role": 'user', "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(user_msg, add_generation_prompt=True, tokenize=False)

        chosen_msg = [
            {"role": 'user', "content": prompt},
            {"role": 'assistant', "content": chosen[1]['content']}
        ]
        chosen_text = tokenizer.apply_chat_template(chosen_msg, add_generation_prompt=False, tokenize=False)

        chosen_msg[-1]['content'] = rejected[1]['content']
        rejected_msg = chosen_msg
        rejected_text = tokenizer.apply_chat_template(rejected_msg, add_generation_prompt=False, tokenize=False)

        return {
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
        }

    train_dataset = dataset['train_prefs'].map(
        apply_chat_template,
        batched=False,
        remove_columns=dataset['train_prefs'].column_names
    )
    test_dataset = dataset['test_prefs'].map(
        apply_chat_template,
        batched=False,
        remove_columns=dataset['test_prefs'].column_names
    )

    model = AutoModelForCausalLM.from_pretrained(args.sft_model)

    os.makedirs(os.path.join(SCRIPT_DIR, args.output_dir), exist_ok=True)
    output_dir = os.path.join(str(SCRIPT_DIR), args.output_dir, f"{args.base_model.split('/')[-1]}-{args.output_name}")

    dpo_config = DPOConfig(
        output_dir=output_dir,
        eval_strategy='steps',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=True,
        eval_steps=args.eval_steps,
        max_length=args.max_length,
    )
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    dpo_trainer.train(resume_from_checkpoint=not args.no_resume)

def main():
    train(args)

if __name__ == "__main__":
    main()