import argparse
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import datasets
from tqdm import tqdm


# ChatML special tokens
IM_START = "<|im_start|>"
IM_END   = "<|im_end|>"

# Configurable system prompt (set to None to omit)
SYSTEM_PROMPT = "You are a helpful assistant."

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="reinforcement learning (PPO)")
parser.add_argument('--base_model', default='unsloth/Llama-3.2-1B', type=str)
parser.add_argument('--sft_model', default="./checkpoints/Llama-3.2-1B-fine-tuned-model/checkpoint-738", type=str)
parser.add_argument('--reward_model', default="./checkpoints/Llama-3.2-1B-reward-model/checkpoint-9000", type=str)
parser.add_argument('--hf_dataset_name', default='HuggingFaceH4/cherry_picked_prompts', type=str)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--lr', default=1e-6, type=float)
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--max_length', default=512, type=str)
parser.add_argument('--gen_max_length', default=512, type=str)
parser.add_argument('--clip_range', default=0.2, type=float)
parser.add_argument('--kl_coef', default=0.05, type=str)
parser.add_argument('--logging_steps', default=10, type=int)
parser.add_argument('--output_dir', default="./checkpoints/", type=str)
parser.add_argument('--output_name', default="rlhf-model", type=str)
args = parser.parse_args()

def get_batches(dataset, batch_size, shuffle=True):
    if shuffle:
        dataset = dataset.shuffle()

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        yield batch

def gen_response(model, tokenizer, prompt_ids):
    with torch.no_grad():
        output = model.generate(
            input_ids=prompt_ids,
            max_new_tokens=args.gen_max_length,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return output[:, prompt_ids.shape[1]:]  # new tokens only

def compute_rewards(reward_model, tokenizer, prompts, responses):
    texts = [p + r for p, r in zip(prompts, responses)]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt')
    inputs = inputs.to(reward_model.device)
    with torch.no_grad():
        rewards = reward_model(**inputs).logits.squeeze(-1)
    return rewards

def gae_advantages(rewards, values, gamma=0.99, lam=0.95):
    batch_size, seq_length = values.shape
    device = values.device

    last_rewards = torch.zeros(batch_size, seq_length, device=device)
    last_rewards[:, -1] = rewards

    next_values = torch.cat([values[:, 1:], torch.zeros(batch_size, 1, device=device)], dim=1)
    deltas = last_rewards + gamma * next_values - values

    # Backward GAE
    advantages = torch.zeros_like(values)
    gae = 0.0
    for t in reversed(range(seq_length)):
        gae = deltas[:, t] + gamma * lam * gae
        advantages[:, t] = gae

    returns = advantages + values
    return advantages, returns

def train(args):
    dataset = datasets.load_dataset(args.hf_dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)

    model = AutoModelForCausalLM.from_pretrained(args.sft_model, dtype=torch.bfloat16)
    value_head = nn.Linear(model.config.hidden_size, 1, dtype=torch.bfloat16)
    ref_model = AutoModelForCausalLM.from_pretrained(args.sft_model, dtype=torch.bfloat16)
    reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model, num_labels=1, dtype=torch.bfloat16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    value_head.to(device)
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    reward_model.to(device)
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(value_head.parameters()), lr=args.lr)
    global_step = 0

    model.train()
    value_head.train()

    total_batches = (len(dataset['train']) + args.batch_size - 1) // args.batch_size * args.num_epochs
    pbar =tqdm(total=total_batches, desc='Training')

    for epoch in range(args.num_epochs):
        for batch in get_batches(dataset['train'], args.batch_size):
            global_step += 1

            batch_ids = tokenizer(batch['prompt'], padding=True, truncation=True,
                                  max_length=args.max_length, return_tensors='pt').input_ids
            batch_ids = batch_ids.to(device)

            with torch.no_grad():
                response_ids = gen_response(model, tokenizer, batch_ids)
            response_texts = [tokenizer.decode(r, skip_special_tokens=True) for r in response_ids]

            rewards = compute_rewards(reward_model, tokenizer, batch['prompt'], response_texts)
            rewards = rewards - rewards.mean()

            full_ids = torch.cat([batch_ids, response_ids], dim=1)
            attention_mask = torch.ones_like(full_ids)

            outputs = model(full_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits = outputs.logits
            logprobs = torch.log_softmax(logits, dim=-1)

            with torch.no_grad():
                ref_outputs = ref_model(full_ids, attention_mask=attention_mask, output_hidden_states=True)
                ref_logits = ref_outputs.logits
                ref_logprobs = torch.log_softmax(ref_logits, dim=-1)

            # Gather logprobs of taken actions
            actions = response_ids[:, :-1]
            logprobs_taken = torch.gather(logprobs[:, :-1, :], 2, actions.unsqueeze(-1)).squeeze(-1)
            ref_logprobs_taken = torch.gather(ref_logprobs[:, :-1, :], 2, actions.unsqueeze(-1)).squeeze(-1)

            response_mask = response_ids != tokenizer.pad_token_id
            values = value_head(outputs.hidden_states[-1][:, batch_ids.shape[1]:-1, :]).squeeze(-1)
            values = values * response_mask[:, :-1].float()

            # GAE
            advantages, returns = gae_advantages(rewards, values)
            # Or simple way
            # advantages = (rewards.unsqueeze(1) - values.detach()) * response_mask[:, :-1].float()
            # returns = rewards.unsqueeze(1).expand(-1, values.size(1)) * response_mask[:, :-1].float()

            # Clipped surrogate objective (PPO)
            logratio = logprobs_taken - ref_logprobs_taken
            ratio = torch.exp(logratio)
            ratio = ratio * response_mask[:, :-1].float() + (1 - response_mask[:, :-1].float())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = ((values - returns) ** 2).mean()

            kl_penalty = (logratio * response_mask[:, :-1].float()).sum() / response_mask[:, :-1].sum()

            loss = policy_loss + 0.5 * value_loss + args.kl_coef * kl_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % args.logging_steps == 0:
                print(f"Step [{global_step}], Loss: {loss.item():.4f}, KL: {kl_penalty.item():.4f}, Reward: {rewards.mean().item():.3f}")

            pbar.update(1)

    os.makedirs(os.path.join(SCRIPT_DIR, args.output_dir), exist_ok=True)
    output_dir = os.path.join(str(SCRIPT_DIR), args.output_dir, f"{args.base_model.split('/')[-1]}-{args.output_name}")
    model.save_pretrained(output_dir)

def main():
    train(args)

if __name__ == "__main__":
    main()