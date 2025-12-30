# AI Assistant Demo

### This demo showcases a simple workflow for fine-tuning a pretrained LLM and turning it into a basic AI assistant.

## Introduction

This repository implements two popular post-training alignment approaches:

1. **Classic RLHF pipeline**
   - **Supervised fine-tuning (SFT)** on high-quality demonstrations
   - **Reward model (RM)** on comparison data  
   - Reinforcement learning from human feedback with **Proximal Policy Optimization (PPO)**

2. **Direct preference optimization (DPO) pipeline**  
   - **Supervised fine-tuning (SFT)** on high-quality demonstrations  
   - **Direct preference optimization (DPO)**

## Project Structure
```
ai-assistant-demo
├── datasets
│   ├── demonstration_data
│   │   └── open-assistant-oasst2
│   │       └── data_preprocessing.py
│   ├── get_dataset.sh
│   └── README.md
├── README.md
├── requirements.txt
├── scripts
│   ├── dpo_training_pipeline.sh
│   └── rlhf_training_pipeline.sh
└── src
    └── ai-assistant-demo
        ├── inference
        │   ├── dpo_chat.py
        │   └── rlhf_chat.py
        ├── train_dpo.py
        ├── train_rlhf.py
        ├── train_rm.py
        ├── train_sft.py
        ├── trl_train_dpo.py
        ├── trl_train_rm.py
        └── trl_train_sft.py
```

## Data Preparation

### OpenAssistant Conversations (OASST2)
Download the file:
```bash
cd datasets
bash get_dataset.sh
```

Convert into ChatML prompt format:
```bash
cd demonstration/open-assistant-oasst2
python data_preprocessing.py
```

## Supervised Fine-Tuning
Supervised fine-tuning on the above OASST2 dataset, using **Llama-3.2-1B** as the base model:
```bash
python ./src/ai-assistant-demo/trl_train_sft.py \
  --batch_size 4 \
  --lr 2e-4 \
  --num_epochs 2 \
  --max_length 512 \
  --save_steps 2_000 \
  --no_resume
```

<details><summary> Custom supervised fine-tuning without using the TRL package: </summary>

```bash
python ./src/ai-assistant-demo/train_sft.py \
  --batch_size 4 \
  --lr 2e-4 \
  --num_epochs 2 \
  --max_length 512 \
  --save_steps 2_000 \
  --no_resume
```
</details>

## Reward Model Training
Reward model training on preference datasets (e.g., Anthropic/hh-rlhf):
```bash
python ./src/ai-assistant-demo/trl_train_rm.py \
  --batch_size 4 \
  --lr 1e-5 \
  --num_epochs 1 \
  --max_length 512 \
  --save_steps 5_000 \
  --no_resume
```

<details><summary> Custom reward model training without using the TRL package: </summary>

```bash
python ./src/ai-assistant-demo/train_rm.py \
  --batch_size 4 \
  --lr 1e-5 \
  --num_epochs 1 \
  --max_length 512 \
  --save_steps 5_000 \
  --no_resume
```
</details>

## Reinforcement Learning (PPO)
Reinforcement learning from human feedback with PPO:
```bash
python ./src/ai-assistant-demo/train_rlhf.py \
  --batch_size 2 \
  --lr 1e-6 \
  --num_epochs 10 \
  --max_length 512 \
  --gen_max_length 512 \
  --clip_range 0.2 \
  --kl_coef 0.05 \
  --logging_steps 10
```

## Direct Preference Optimization (DPO)
Direct preference optimization, trained directly from the SFT model:
```bash
python ./src/ai-assistant-demo/trl_train_dpo.py \
  --batch_size 2 \
  --lr 5e-5 \
  --num_epochs 1 \
  --max_length 512 \
  --logging_steps 1_000 \
  --save_steps 5_000 \
  --eval_steps 2_000 \
  --no_resume
```

<details><summary> Direct preference optimization without using the TRL package: </summary>

```bash
python ./src/ai-assistant-demo/train_dpo.py \
  --batch_size 2 \
  --lr 5e-5 \
  --num_epochs 1 \
  --max_length 512 \
  --logging_steps 1_000 \
  --save_steps 5_000 \
  --eval_steps 2_000 \
  --no_resume
```

</details>

## Inference
Test the RLHF/DPO model:
```bash
# RLHF model
python ./src/ai-assistant-demo/inference/rlhf_chat.py \
  --prompt "How can I build a daily drawing habit?"
```

```bash
# DPO model
python ./src/ai-assistant-demo/inference/dpo_chat.py \
  --prompt "How can I build a daily drawing habit?"
```

## Scripts:
- Training (RLHF): `scripts/rlhf_training_pipeline.sh`
- Training (DPO): `scripts/dpo_training_pipeline.sh`

## Reference & Acknowledgement:
- InstructGPT: [[arXiv](https://arxiv.org/abs/2203.02155)]
- OpenAssistant: [[GitHub](https://github.com/LAION-AI/Open-Assistant)] [[arXiv](https://arxiv.org/abs/2304.07327)]
- TRL - Transformer Reinforcement Learning: [[GitHub](https://github.com/huggingface/trl)]