from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForTokenClassification,
    set_seed,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
)


import wandb
from datasets import load_dataset
import json
import numpy as np
import evaluate
import os
from dataclasses import dataclass, field
from typing import Optional


# @dataclass
# class ModelArguments:
#     """
#     Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
#     """

#     model_name_or_path: str = field(
#         metadata={
#             "help": "Path to pretrained model or model identifier from huggingface.co/models"
#         }
#     )
#     config_name: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": "Pretrained config name or path if not the same as model_name"
#         },
#     )
#     tokenizer_name: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": "Pretrained tokenizer name or path if not the same as model_name"
#         },
#     )
#     cache_dir: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
#         },
#     )
#     model_revision: str = field(
#         default="main",
#         metadata={
#             "help": "The specific model version to use (can be a branch name, tag name or commit id)."
#         },
#     )
#     use_auth_token: bool = field(
#         default=False,
#         metadata={
#             "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
#             "with private models)."
#         },
#     )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(
        default="ner", metadata={"help": "The name of the task (ner, pos...)."}
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a csv or JSON file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to predict on (a csv or JSON file)."
        },
    )
    text_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The column name of text to input in the file (a csv or JSON file)."
        },
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The column name of label to input in the file (a csv or JSON file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If set, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={
            "help": "Whether to return all the entity levels during evaluation or just the overall ones."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()


def train(config=None):
    with wandb.init(config=config):
        # set sweep configuration
        config = wandb.config
        open_json_file = open("modules/run_configs.json")
        run_configs = json.loads(open_json_file.read())
        print(run_configs)
        # config = {**sweep_config, **run_configs}
        # print("Configs--->: ", config)
        parser = HfArgumentParser((DataTrainingArguments, TrainingArguments))

        data_args, training_args = parser.parse_dict(run_configs)
        set_seed(training_args.seed)

        data_files = {}
        data_files["train"] = data_args.train_file
        data_files["validation"] = data_args.validation_file
        data_files["test"] = data_args.test_file
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
        )

        def get_label_list(labels):
            unique_labels = set()
            for label in labels:
                unique_labels = unique_labels | set(label)
            label_list = list(unique_labels)
            label_list.sort()
            return label_list

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            use_fast=True,
            do_lower_case=False,
        )
        column_names = raw_datasets["train"].column_names
        label_column_name = column_names[1]

        text_column_name = column_names[0]
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}

        num_labels = len(label_list)

        # Map that sends B-Xxx label to its I-Xxx counterpart
        b_to_i_label = []
        for idx, label in enumerate(label_list):
            if label.startswith("B-") and label.replace("B-", "I-") in label_list:
                b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
            else:
                b_to_i_label.append(idx)

        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples[text_column_name],
                padding="max_length",
                truncation=True,
                max_length=data_args.max_seq_length,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True,
            )
            labels = []
            for i, label in enumerate(examples[label_column_name]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        label_ids.append(-100)
                    # We set the label for the first token of each word.
                    elif word_idx != previous_word_idx:
                        label_ids.append(label_to_id[(label[word_idx])])
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        if data_args.label_all_tokens:
                            label_ids.append(
                                b_to_i_label[label_to_id[(label[word_idx])]]
                            )
                        else:
                            label_ids.append(-100)
                    previous_word_idx = word_idx

                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        print(train_dataset)

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(test_dataset), data_args.max_predict_samples)
            test_dataset = test_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            test_dataset = test_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

        # set training arguments
        training_args = TrainingArguments(
            output_dir="ttttt",
            report_to=["wandb"],  # Turn on Weights & Biases logging
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=16,
            save_strategy="epoch",
            evaluation_strategy="steps",
            eval_steps=150,
            logging_strategy="epoch",
            # load_best_model_at_end=True,
            remove_unused_columns=True,
            fp16=True,
        )

        metric = evaluate.load("seqeval")

        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = metric.compute(
                predictions=true_predictions, references=true_labels
            )
            if data_args.return_entity_level_metrics:
                # Unpack nested dictionaries
                final_results = {}
                for key, value in results.items():
                    if isinstance(value, dict):
                        for n, v in value.items():
                            final_results[f"{key}_{n}"] = v
                    else:
                        final_results[key] = value
                return final_results
            else:
                return {
                    "precision": results["overall_precision"],
                    "recall": results["overall_recall"],
                    "f1": results["overall_f1"],
                    "accuracy": results["overall_accuracy"],
                }

        def model_init():
            model = AutoModelForTokenClassification.from_pretrained(
                config.model_name_or_path,
                # config=model_config,
                num_labels=num_labels,
            )
            return model

        data_collator = DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
        # define training loop
        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        # start training loop
        train_result = trainer.train()
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        if training_args.do_eval:

            metrics = trainer.evaluate()
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        # Predict
        if training_args.do_predict:

            predictions, labels, metrics = trainer.predict(
                test_dataset, metric_key_prefix="predict"
            )
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

            # Save predictions
            output_predictions_file = os.path.join(
                training_args.output_dir, "predictions.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_predictions_file, "w") as writer:
                    for prediction in true_predictions:
                        writer.write(" ".join(prediction) + "\n")


if __name__ == "__main__":
    # method
    sweep_config = {"method": "random"}

    # hyperparameters
    parameters_dict = {
        "epochs": {"value": 1},
        "batch_size": {"values": [2, 4]},
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-7,
            "max": 1e-4,
        },
        "weight_decay": {"values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
        "model_name_or_path": {
            "values": [
                "xlm-roberta-base",
                "adalbertojunior/distilbert-portuguese-cased",
                "microsoft/deberta-v3-small",
            ]
        },
    }

    sweep_config["parameters"] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="corejur_v18")
    wandb.agent(sweep_id, train, count=1)
