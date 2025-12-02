import argparse
import os
from transformers import AutoModelForSequenceClassification
import datasets
from trl import RewardConfig, RewardTrainer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="reward model training")
parser.add_argument('--base_model', default='unsloth/Llama-3.2-1B', type=str)
parser.add_argument('--hf_dataset_name', default='Anthropic/hh-rlhf', type=str)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--max_length', default=512, type=str)
parser.add_argument('--no_resume', action='store_true')
parser.add_argument('--output_dir', default="./checkpoints", type=str)
parser.add_argument('--output_name', default="reward-model", type=str)
args = parser.parse_args()

def train(args):
    dataset = datasets.load_dataset(args.hf_dataset_name)

    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=1)

    os.makedirs(os.path.join(SCRIPT_DIR, args.output_dir), exist_ok=True)
    output_dir = os.path.join(str(SCRIPT_DIR), args.output_dir, f"{args.base_model.split('/')[-1]}-{args.output_name}")

    reward_config = RewardConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        logging_steps=1_000,
        bf16=True,
        max_length=args.max_length,
    )
    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
    )
    trainer.train(resume_from_checkpoint=not args.no_resume)

def main():
    train(args)

if __name__ == "__main__":
    main()