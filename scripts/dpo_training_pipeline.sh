cd ../src/ai-assistant-demo

: "${USE_TRL:=true}"
if [[ "$USE_TRL" = true ]]; then
  echo "Training w/ TRL: "
  # Supervised fine-tuning
  python trl_train_sft.py --batch_size 4 --lr 2e-4 --num_epochs 2 --max_length 512 --save_steps 2_000 --no_resume
  # Direct preference optimization (DPO)
  python trl_train_dpo.py --batch_size 2 --lr 5e-5 --num_epochs 1 --max_length 512 --logging_steps 1_000 --save_steps 5_000 --eval_steps 2_000 --no_resume
else
  echo "Training: "
  # Supervised fine-tuning
  python train_sft.py --batch_size 4 --lr 2e-4 --num_epochs 2 --max_length 512 --save_steps 2_000 --no_resume
  # Direct preference optimization (DPO)
  python train_dpo.py --batch_size 2 --lr 5e-5 --num_epochs 1 --max_length 512 --logging_steps 1_000 --save_steps 5_000 --eval_steps 2_000 --no_resume
fi

cd -