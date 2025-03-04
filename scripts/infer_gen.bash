ckpt=""

lists=(1 2 3 4 5 6 7 8)
for epoch in ${lists[@]}; do
    ((test_ckpt=${ckpt}*${epoch}))

    CUDA_VISIBLE_DEVICES=1 python ../LLM_training/inference_vllm.py --pretrained_model_name_or_path ../models/ckpts/job_gen/ckpt-${test_ckpt} --input_file ../data/val_job.json --output_file ../result/job_val_epoch${epoch}.json --tensor_parallel_size 1 --n 16 --temperature 1 --model_mode generate --cont 0 

    CUDA_VISIBLE_DEVICES=1 python ../LLM_training/inference_vllm.py --pretrained_model_name_or_path ../models/ckpts/job_gen/ckpt-${test_ckpt} --input_file ../data/test_job.json --output_file ../result/job_epoch${epoch}.json --tensor_parallel_size 1 --n 16 --temperature 1 --model_mode generate --cont 0 
    
done