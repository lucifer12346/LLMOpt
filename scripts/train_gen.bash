YOUR_ACCELERATE_CONFIG=""
PRETRAINED_MODEL=""

accelerate launch --config_file ${YOUR_ACCELERATE_CONFIG} ../LLM_training/train.py --per_device_train_batch_size 1 --block_size 8192 --seed 42 --pretrained_model_name_or_path ${PRETRAINED_MODEL} --epochs 8 --lr 5e-6 --checkpointing_steps 1000000000000000 --tensorboard_log_dir ../models/train_logs/job_gen --output_ckpt_dir ../models/ckpts/job_gen --training_data_dir ../data/train_job --model_type generate 