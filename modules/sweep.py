import wandb
from run_trial import main

if __name__ == "__main__":
    sweep_config = {
        "method": "grid",
        "parameters": {
            "learning_rate": {"values": [1e-5, 5e-4]},
            "model_name_or_path": {
                "values": [
                    "xlm-roberta-base",
                    "adalbertojunior/distilbert-portuguese-cased",
                ]
            },
            "per_device_train_batch_size": {"values": [4, 8]},
        },
    }

    run_config = {
        "parameters": {
            "task_name": "ner",
            "max_steps": 15000,
            "output_dir": "runs/v17-xlmrobertabase",
            "seed": 2,
            "train_file": "processed_data/v17_80_0_512/fold-0/train.json",
            "validation_file": "processed_data/v17_80_0_512/fold-0/dev.json",
            "test_file": "processed_data/v17_80_0_512/test.json",
            "do_train": True,
            "do_eval": True,
            "do_predict": True,
            "report_to": ["mlflow", "wandb"],
            "evaluation_strategy": "epoch",
            "max_seq_length": 512,
            "return_entity_level_metrics": True,
            "save_total_limit": 1,
            "lr_scheduler_type": "linear",
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "fp16": True,
            "metric_for_best_model": "overall_f1",
        }
    }
    # sweep_config.update(run_config)
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=main)
