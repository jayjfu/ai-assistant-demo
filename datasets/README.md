## Datasets
We use the `oasst_ready-trees` file (2023-11-05_oasst2_all.trees.jsonl.gz), and convert it into OpenAI's ChatML prompt format.
- OpenAssistant/oasst2: [[HF](https://huggingface.co/datasets/OpenAssistant/oasst2)]

Download the file: 
```bash
bash get_dataset.sh
```

Convert into ChatML prompt format:
```bash
cd demonstration/open-assistant-oasst2
python data_preprocessing.py
```