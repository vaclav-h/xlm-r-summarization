# Extractive summarization using XLM-RoBERTa

Code for fine-tuning XLM-RoBERTa (XLM-R) [1] for extractive summarization. Can be used with any of the 100 supported languages by XLM-R.


## Fine-tuning instruction

1. **Build dataset**
The dataset can be built using a dataset for abstractive summarization in JSON-lines format with fields 'text' (the text to be summarized) and 'abstract' (the target summary).
Example: `python3 build_ext_dataset.py --file abstractive_dataset.jsonl -output output_file.jsonl --lang cs`
2. **Run training**
Edit hyperparameters in `config.py` and run `train.py`.
Example: `python3 train.py --gpus 1` to train on single GPU or `python -m torch.distributed.launch train.py --gpus N` to train on N GPUs. Training on CPU is not supported.

We experimented with training only the [xlm-roberta-large](https://huggingface.co/xlm-roberta-large) but the smaller version [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) should also work by changing the 'model_type' field in `config.py`.

## Example

Demo usage of the fine-tuned model is [here](https://github.com/vaclav-h/xlm-r-summarization/blob/main/inference_demo.ipynb).

## References

[1] Conneau, Alexis, et al. "Unsupervised cross-lingual representation learning at scale." _arXiv preprint arXiv:1911.02116_ (2019).
