# AI Assistant Demo

### This demo showcases a simple workflow for fine-tuning a pretrained LLM and turning it into a basic AI assistant.

*(Work in Progress: This repository is currently under active development.)*

## Introduction

This repository includes two popular post-training alignment approaches:

1. **Classic RLHF pipeline**
   - **Supervised fine-tuning (SFT)** on high-quality demonstrations
   - **Reward model (RM)** on comparison data  
   - Reinforcement learning from human feedback with **Proximal Policy Optimization (PPO)**

2. **Direct preference optimization (DPO) pipeline**  
   - **Supervised fine-tuning (SFT)** on high-quality demonstrations  
   - **Direct preference optimization (DPO)**

## Project Structure
```bash
# TODO
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
```bash
python ./src/ai-assistant-demo/trl_train_sft.py
# Or
# python ./src/ai-assistant-demo/train_sft.py
```

## Reward Model Training
```bash
python ./src/ai-assistant-demo/trl_train_rm.py
# Or
# python ./src/ai-assistant-demo/train_rm.py
```

## Reinforcement Learning (PPO)
```bash
python ./src/ai-assistant-demo/train_rlhf.py
```

## Direct Preference Optimization (DPO)
```bash
python ./src/ai-assistant-demo/trl_train_dpo.py
# Or
# python ./src/ai-assistant-demo/train_dpo.py
```

## Inference
TODO
```bash
python inference/dpo_chat.py
```

## Scripts:
- Training (RLHF): `scripts/rlhf_training_pipeline.sh`
- Training (DPO): `scripts/dpo_training_pipeline.sh`

## Reference & Acknowledgement:
- InstructGPT: [[arXiv](https://arxiv.org/abs/2203.02155)]
- OpenAssistant: [[GitHub](https://github.com/LAION-AI/Open-Assistant)] [[arXiv](https://arxiv.org/abs/2304.07327)]
- TRL - Transformer Reinforcement Learning: [[GitHub](https://github.com/huggingface/trl)]