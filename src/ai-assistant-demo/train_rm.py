import argparse
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
import datasets


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="reward model training")
parser.add_argument('--base_model', default='unsloth/Llama-3.2-1B', type=str)
parser.add_argument('--hf_dataset_name', default='Anthropic/hh-rlhf', type=str)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--max_length', default=512, type=int)
parser.add_argument('--save_steps', default=5_000, type=int)
parser.add_argument('--no_resume', action='store_true')
parser.add_argument('--output_dir', default="./checkpoints", type=str)
parser.add_argument('--output_name', default="reward-model", type=str)
args = parser.parse_args()

class RewardModelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs_chosen = model(
            input_ids=inputs['input_ids_chosen'],
            attention_mask=inputs['attention_mask_chosen'],
        )
        outputs_rejected = model(
            input_ids=inputs['input_ids_rejected'],
            attention_mask=inputs['attention_mask_rejected'],
        )

        reward_chosen = outputs_chosen.logits.squeeze(-1)
        reward_rejected = outputs_rejected.logits.squeeze(-1)

        # Pairwise ranking loss
        loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()

        return (loss, {"loss": loss, "reward_chosen": reward_chosen, "reward_rejected": reward_rejected}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)

        if prediction_loss_only:
            return loss, None, None
        else:
            batch_size = inputs['input_ids_chosen'].shape[0]
            dummy_labels = torch.zeros(batch_size, device=loss.device)
            return loss, None, dummy_labels

def train(args):
    raw_dataset = datasets.load_dataset(args.hf_dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def preprocess_function(examples):
        chosen_texts = examples['chosen']
        rejected_texts = examples['rejected']

        chosen_tokenized = tokenizer(chosen_texts, padding=False, truncation=True, max_length=args.max_length)
        rejected_tokenized = tokenizer(rejected_texts, padding=False, truncation=True, max_length=args.max_length)

        return {
            "input_ids_chosen": chosen_tokenized['input_ids'],
            "attention_mask_chosen": chosen_tokenized['attention_mask'],
            "input_ids_rejected": rejected_tokenized['input_ids'],
            "attention_mask_rejected": rejected_tokenized['attention_mask'],
        }

    tokenized_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_dataset['train'].column_names,
    )

    def paired_collator(features):
        chosen_ids = [f['input_ids_chosen'] for f in features]
        chosen_masks = [f['attention_mask_chosen'] for f in features]
        rejected_ids = [f['input_ids_rejected'] for f in features]
        rejected_masks = [f['attention_mask_rejected'] for f in features]

        # Compute the global max length
        all_lengths = [len(seq) for seq in chosen_ids + rejected_ids]
        max_len = max(all_lengths)

        chosen_padded = tokenizer.pad(
            {"input_ids": chosen_ids, "attention_mask": chosen_masks},
            padding='max_length',
            max_length=max_len,
            return_tensors='pt',
        )
        rejected_padded = tokenizer.pad(
            {"input_ids": rejected_ids, "attention_mask": rejected_masks},
            padding='max_length',
            max_length=max_len,
            return_tensors='pt',
        )

        return {
            "input_ids_chosen": chosen_padded['input_ids'],
            "attention_mask_chosen": chosen_padded['attention_mask'],
            "input_ids_rejected": rejected_padded['input_ids'],
            "attention_mask_rejected": rejected_padded['attention_mask'],
        }

    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=1)

    os.makedirs(os.path.join(SCRIPT_DIR, args.output_dir), exist_ok=True)
    output_dir = os.path.join(str(SCRIPT_DIR), args.output_dir, f"{args.base_model.split('/')[-1]}-{args.output_name}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy='steps',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        logging_steps=1_000,
        save_steps=args.save_steps,
        bf16=True,
        remove_unused_columns=False,
    )
    trainer = RewardModelTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=paired_collator,
    )
    trainer.train(resume_from_checkpoint=not args.no_resume)

def main():
    train(args)

if __name__ == "__main__":
    main()