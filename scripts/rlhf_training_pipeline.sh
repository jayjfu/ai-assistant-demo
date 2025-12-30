cd ../src/ai-assistant-demo

: "${USE_TRL:=true}"
if [[ "$USE_TRL" = true ]]; then
  echo "Training w/ TRL: "
  # Supervised fine-tuning
  python trl_train_sft.py --no_resume
  # Reward model training
  python trl_train_rm.py --no_resume
else
  echo "Training: "
  # Supervised fine-tuning
  python train_sft.py --no_resume
  # Reward model training
  python train_rm.py --no_resume
fi

# Reinforcement learning from human feedback (PPO)
python train_rlhf.py --no_resume

cd -