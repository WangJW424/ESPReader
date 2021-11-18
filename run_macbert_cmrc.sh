#intensive module
export CMRC_DIR=./cmrc
python run_cmrc.py \
    --model_type bert \
    --model_name_or_path chinese-macbert-base \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file cmrc2018_train.json \
    --predict_file cmrc2018_dev.json \
    --data_dir $CMRC_DIR \
    --learning_rate 5e-5 \
    --num_train_epochs 2 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --max_query_length=64 \
    --per_gpu_train_batch_size=24 \
    --per_gpu_eval_batch_size=24 \
    --max_sentence_num=32  \
    --warmup_steps=0.1 \
    --ILF_rate 0.1 \
    --output_dir macbert/cmrc_chinese-macbert-base_lr5e-5_len512_bs24_ep2_wm01_fp24_gpu \
    --save_steps 500 \
    --n_best_size=20 \
    --max_answer_length=30 \
    --gradient_accumulation_steps=1