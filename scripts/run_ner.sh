python modules/run_ner_transformers.py \
  --task_name ner \
  --per_device_train_batch_size 8 \
  --learning_rate 5e-5 \
  --max_steps 15000 \
  --output_dir runs/v17-xlmrobertabase \
  --seed 2 \
  --model_name_or_path xlm-roberta-base \
  --train_file processed_data/v17_80_0_512/fold-0/train.json  \
  --validation_file processed_data/v17_80_0_512/fold-0/dev.json \
  --test_file processed_data/v17_80_0_512/test.json \
  --do_train \
  --do_eval \
  --do_predict \
  --report_to mlflow wandb \
  --evaluation_strategy epoch \
  --max_seq_length 512 \
  --return_entity_level_metrics True \
  --save_total_limit 1 \
  --lr_scheduler_type linear \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing False \
  --fp16 True \
  --metric_for_best_model overall_f1 