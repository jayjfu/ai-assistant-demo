# AI Assistant Demo

### This demo shows how to build a simple AI assistant by fine-tuning a pretrained LLM.

*(Work in Progress: This repository is currently under active development.)*

## Introduction

- supervised fine-tuning LLM on an instruction dataset (SFT + Reward Model (for Ranking) + RLHF)
  - SFT = supervised-finetuning, RLHF = reinforcement learning form human feedback
  - ~~ -> multi turn conversations?~~

Dataset:
- OpenAssistant Conversations
- Dolly 2.0 Dataset
- ShareGPT
- Alpaca
- Orca-style datasets

## Supervised Fine-Tuning
```bash
```

## Reward Model Training
```bash
```

## Reinforcement Learning (PPO)
```bash
```

## Direct Preference Optimization (DPO)
```bash
```

## Inference
TODO
```bash
python inference/dpo_chat.py
```

## Evaluation
~~Evaluation Options?~~

## Project Structure
```bash
TODO
```

## Reference & Acknowledgement:
- InstructGPT: [[arXiv](https://arxiv.org/abs/2203.02155)]
- OpenAssistant: [[GitHub](https://github.com/LAION-AI/Open-Assistant)] [[arXiv](https://arxiv.org/abs/2304.07327)]
- TRL - Transformer Reinforcement Learning: [[GitHub](https://github.com/huggingface/trl)]