cd ../src/ai-assistant-demo

: "${USE_TRL:=true}"
if [[ "$USE_TRL" = true ]]; then
  echo "Training w/ TRL: "
  # Supervised fine-tuning
  python trl_train_sft.py --no_resume
  # Direct preference optimization (DPO)
  python trl_train_dpo.py --no_resume
else
  echo "Training: "
  # Supervised fine-tuning
  python train_sft.py --no_resume
  # Direct preference optimization (DPO)
  python train_dpo.py --no_resume
fi

cd -