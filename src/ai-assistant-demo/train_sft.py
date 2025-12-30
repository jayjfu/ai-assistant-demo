import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import datasets

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="supervised fine-tuning")
parser.add_argument('--base_model', default='unsloth/Llama-3.2-1B', type=str)
parser.add_argument('--dataset_path', default="../../datasets/demonstration_data/open-assistant-oasst2/processed", type=str)
parser.add_argument('--dataset_name', default="oasst2_chatml.jsonl", type=str)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--num_epochs', default=2, type=int)
parser.add_argument('--max_length', default=512, type=int)
parser.add_argument('--save_steps', default=2_000, type=int)
parser.add_argument('--no_resume', action='store_true')
parser.add_argument('--output_dir', default="./checkpoints", type=str)
parser.add_argument('--output_name', default="fine-tuned-model", type=str)
args = parser.parse_args()

def train(args):
    data_files = {"train": os.path.join(SCRIPT_DIR, args.dataset_path, args.dataset_name)}
    raw_dataset = datasets.load_dataset('json', data_files=data_files)

    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize_function(examples):
        tokenized_text = tokenizer(examples['text'], padding=False, truncation=True, max_length=args.max_length)

        return tokenized_text

    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_dataset['train'].column_names,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    os.makedirs(os.path.join(SCRIPT_DIR, args.output_dir), exist_ok=True)
    output_dir = os.path.join(str(SCRIPT_DIR), args.output_dir, f"{args.base_model.split('/')[-1]}-{args.output_name}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        logging_steps=500,
        save_steps=args.save_steps,
        bf16=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=not args.no_resume)

def main():
    train(args)

if __name__ == "__main__":
    main()