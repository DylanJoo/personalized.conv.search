MODEL=DylanJHJ/gtr-t5-base
export CUDA_VISIBLE_DEVICES=2 
python3 train/train_retriever.py \
        --model_name_or_path ${MODEL} \
        --tokenizer_name ${MODEL} \
        --train_file data/start/train.jsonl \
        --config_name ${MODEL} \
        --output_dir models/ckpt/start-base \
        --max_p_length 32 \
        --max_q_length 8 \
        --per_device_train_batch_size 2 \
        --learning_rate 1e-5 \
        --evaluation_strategy steps \
        --max_steps 10000 \
        --save_steps 2000 \
        --eval_steps 500 \
        --freeze_document_encoder true \
        --do_train \
        --do_eval 
