def get_cfg():
    return {
        "trn_data" : "my_preprocessed_train_data.jsonl", # Dataset built using build_ext_dataset.py
        "eval_data" : "my_preprocessed_eval_data.jsonl", # Dataset built using build_ext_dataset.py
        "model_type" : "xlm-roberta-large", # Should also work with 'xlm-roberta-base'
        "epochs" : 3,
        "warmup" : 10000, # Warm-up steps
        "batch_size" : 8,
        "accumulations" : 4, # Gradient accumulation steps
        "report_freq" : 50, # Report frequency in update steps
        "eval_freq" : 1000, # Evaluation frequency in update steps
        "eval_size" : 5000, # Number of samples to use for evaluation
        "max_src_sentences" : 32, # Maximum number of sentences to consider from the source text
        "pos_embed" : 512, # Number of positional embeddings
        "max_lr" : 2e-3, # (* 1e-2)
        "wandb_project" : "",
        "wandb_run" : "",
        "save_path" : "/path/to/my/model.pt"
    }
