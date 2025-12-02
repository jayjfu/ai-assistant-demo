import argparse
import os
from transformers import AutoModelForCausalLM
import datasets
from trl import SFTConfig, SFTTrainer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="supervised fine-tuning")
parser.add_argument('--base_model', default='unsloth/Llama-3.2-1B', type=str)
parser.add_argument('--dataset_path', default="../../datasets/demonstration_data/open-assistant-oasst2/processed", type=str)
parser.add_argument('--dataset_name', default="oasst2_chatml.jsonl", type=str)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--num_epochs', default=2, type=int)
parser.add_argument('--max_length', default=512, type=str)
parser.add_argument('--no_resume', action='store_true')
parser.add_argument('--output_dir', default="./checkpoints", type=str)
parser.add_argument('--output_name', default="fine-tuned-model", type=str)
args = parser.parse_args()

def train(args):
    data_files = {"train": os.path.join(SCRIPT_DIR, args.dataset_path, args.dataset_name)}
    dataset = datasets.load_dataset('json', data_files=data_files)

    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    os.makedirs(os.path.join(SCRIPT_DIR, args.output_dir), exist_ok=True)
    output_dir = os.path.join(str(SCRIPT_DIR), args.output_dir, f"{args.base_model.split('/')[-1]}-{args.output_name}")

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        bf16=True,
        max_length=args.max_length,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset['train'],
    )
    trainer.train(resume_from_checkpoint=not args.no_resume)

def main():
    train(args)

if __name__ == "__main__":
    main()