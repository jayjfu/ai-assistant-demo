import argparse
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import datasets


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="direct preference optimization (DPO) without TRL")
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
parser.add_argument('--beta', default=0.1, type=float)
parser.add_argument('--output_dir', default="./checkpoints/", type=str)
parser.add_argument('--output_name', default="dpo-model", type=str)
args = parser.parse_args()

def compute_dpo_loss(model_chosen_logps, model_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta):
    logits = (model_chosen_logps - model_rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
    losses = -F.logsigmoid(beta * logits)
    return losses.mean()

class DPOTrainer(Trainer):
    def __init__(self, ref_model, beta, processing_class, max_length, **kwargs):
        super().__init__(**kwargs)
        self._signature_columns = ['prompt', 'chosen', 'rejected']

        self.ref_model = ref_model
        self.beta = beta
        self.processing_class = processing_class
        self.max_length = max_length

        self.ref_model.requires_grad_(False)
        self.ref_model.eval()
        self.ref_model.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt, chosen, rejected = inputs['prompt'], inputs['chosen'], inputs['rejected']

        chosen_inputs = self.processing_class(
            [p + c for p, c in zip(prompt, chosen)],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        ).to(model.device)
        rejected_inputs = self.processing_class(
            [p + r for p, r in zip(prompt, rejected)],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        ).to(model.device)

        prompt_lengths = []
        for p in prompt:
            tokenized_prompt = self.processing_class(p, add_special_tokens=True, return_tensors='pt')
            prompt_lengths.append(tokenized_prompt['input_ids'].shape[1])

        prompt_lengths = torch.tensor(prompt_lengths).to(model.device)

        model_chosen_logps = self.get_batch_logps(model, chosen_inputs, prompt_lengths)
        model_rejected_logps = self.get_batch_logps(model, rejected_inputs, prompt_lengths)

        with torch.no_grad():
            ref_chosen_logps = self.get_batch_logps(self.ref_model, chosen_inputs, prompt_lengths)
            ref_rejected_logps = self.get_batch_logps(self.ref_model, rejected_inputs, prompt_lengths)

        loss = compute_dpo_loss(
            model_chosen_logps,
            model_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            self.beta,
        )

        return (loss,) if return_outputs else loss

    @staticmethod
    def get_batch_logps(model, batch_inputs, prompt_lengths):
        outputs = model(**batch_inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch_inputs['input_ids'][..., 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logps = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        response_logps = []
        for i, prompt_len in enumerate(prompt_lengths):
            response_start = prompt_len.item()
            if response_start >= token_logps.shape[1]:# or token_logps[i, response_start:].numel() == 0:
                response_logps.append(torch.tensor(0.0, device=model.device, requires_grad=True))
            else:
                response_logps.append(token_logps[i, response_start:].mean())

        return torch.stack(response_logps)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss, None, None)  # no logits, no labels

def train(args):
    raw_dataset = datasets.load_dataset(args.hf_dataset_name)

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

    train_dataset = raw_dataset['train_prefs'].map(
        apply_chat_template,
        batched=False,
        remove_columns=raw_dataset['train_prefs'].column_names
    )
    eval_dataset = raw_dataset['test_prefs'].map(
        apply_chat_template,
        batched=False,
        remove_columns=raw_dataset['test_prefs'].column_names
    )

    model = AutoModelForCausalLM.from_pretrained(args.sft_model, dtype=torch.bfloat16)
    ref_model = AutoModelForCausalLM.from_pretrained(args.sft_model, dtype=torch.bfloat16)
    ref_model.requires_grad_(False)
    ref_model.eval()

    # Custom collator
    data_collator = lambda features: {
        "prompt": [f["prompt"] for f in features],
        "chosen": [f["chosen"] for f in features],
        "rejected": [f["rejected"] for f in features],
    }

    os.makedirs(os.path.join(SCRIPT_DIR, args.output_dir), exist_ok=True)
    output_dir = os.path.join(str(SCRIPT_DIR), args.output_dir, f"{args.base_model.split('/')[-1]}-{args.output_name}")

    training_args = TrainingArguments(
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
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # tokenizer=tokenizer,
        processing_class=tokenizer,
        max_length=args.max_length,
        beta=args.beta,
    )
    trainer.train(resume_from_checkpoint=not args.no_resume)

def main():
    train(args)

if __name__ == "__main__":
    main()