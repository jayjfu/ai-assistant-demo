cd ../src/ai-assistant-demo

: "${USE_TRL:=true}"
if [[ "$USE_TRL" = true ]]; then
  echo "Training w/ TRL: "
  # Supervised fine-tuning
  python trl_train_sft.py --batch_size 4 --lr 2e-4 --num_epochs 2 --max_length 512 --save_steps 2_000 --no_resume
  # Reward model training
  python trl_train_rm.py --batch_size 4 --lr 1e-5 --num_epochs 1 --max_length 512 --save_steps 5_000 --no_resume
else
  echo "Training: "
  # Supervised fine-tuning
  python train_sft.py --batch_size 4 --lr 2e-4 --num_epochs 2 --max_length 512 --save_steps 2_000 --no_resume
  # Reward model training
  python train_rm.py --batch_size 4 --lr 1e-5 --num_epochs 1 --max_length 512 --save_steps 5_000 --no_resume
fi

# Reinforcement learning from human feedback (PPO)
python train_rlhf.py --batch_size 2 --lr 1e-6 --num_epochs 10 --max_length 512 --gen_max_length 512 --clip_range 0.2 --kl_coef 0.05 --logging_steps 10

cd -