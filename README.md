# ESPReader (Reader with Explicit Span-sentence Predication)

See our paper: [What If Sentence-hood is Hard to Define: A Case Study in Chinese Reading Comprehension](https://aclanthology.org/2021.findings-emnlp.202)

The code is based on Pytorch version Transformer implemented by Huggingface.
Check: https://github.com/huggingface/transformers/tree/master/src/transformers

## Usage
###  Step 1: Download Dataset: CMRC 2018

- Training set: [cmrc2018_train.json](https://worksheets.codalab.org/rest/bundles/0x296baa11dfbc4ab08cdeb5b4adf182e2/contents/blob/)
- Dev set: [cmrc2018_dev.json](https://worksheets.codalab.org/rest/bundles/0x72252619f67b4346a85e122049c3eabd/contents/blob/)


###  Step 2: Download BERT weights 
(You can skip it and our program will download it automatically if the model name is given correctly.) Alternative model names are listed as follows:
- [bert-base-chinese](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin)
- [chinese-macbert-base](https://huggingface.co/hfl/chinese-macbert-base/resolve/main/pytorch_model.bin)
- [chinese-macbert-large](https://huggingface.co/hfl/chinese-macbert-large/resolve/main/pytorch_model.bin)
- [chinese-roberta-wwm-ext-base](https://huggingface.co/hfl/chinese-roberta-wwm-ext/resolve/main/pytorch_model.bin)
- [chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/resolve/main/pytorch_model.bin)

### Step 3: Training and evaluating
We use the following script for training and evaluating ESPReader on CMRC 2018:
```
export CMRC_DIR={The file root of CMRC 2018 datasets}
export Output_DIR={Where to save the trained model and its output files.}
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
    --max_query_length 64 \
    --per_gpu_train_batch_size 3 \
    --per_gpu_eval_batch_size 6 \
    --max_sentence_num 32  \
    --warmup_steps 0.1 \
    --ILF_rate 0.1 \
    --output_dir $Output_DIR/cmrc_chinese-macbert-base_lr5e-5_len512_bs24_ep2_wm01_fp24_gpu \
    --save_steps 5000 \
    --n_best_size 20 \
    --max_answer_length 30 \
    --gradient_accumulation_steps 1
```
You can also run the bash file we provide, using:
```
   sh run_macbert_cmrc.sh
```
(This script is on the assumption that 8 GPUs are available, if you have and want to use mutiple GPUs, be sure that the batch size you are going to set is equal to *per_gpu_train_batch_size x gpu_num x gradient_accumulation_steps*, for example, in this script: 24 = 3 x 8 x 1)

The evaluation script 'cmrc_metrics.py' comes from the official file ['cmrc2018_evaluate.py']( https://worksheets.codalab.org/rest/bundles/0x4747bbd27f894046b15abd894eb0175a/contents/blob/)

If you run this script with all the default parameters we provide, the evaluating results on CMRC 2018 Dev set should be (small fluctuations are acceptable):
- {"AVERAGE": "80.25068", "F1": "88.73993", "EM": "71.76142", "TOTAL": 3219, "SKIP": 0}

##Best Results of ESPReader
The best results of ESPReader on each PrLM on CMRC 2018 banchmark are listed as follows (results on Test set comes from the [CMRC 2018 leaderboard]( http://ymcui.com/cmrc2018/)):

| Model name | Dev set | Test set |
|:----|:----:|:----:|
|  | EM / F1 | EM / F1|  
| bert-base-chinese | 68.7 / 86.3 | -- / -- |
| chinese-roberta-wwm-ext-base | 70.7 / 88.3 | -- /  -- |
| chinese-macbert-base | 71.8 /	88.7 |	75.6 / 90.0 |
| chinese-roberta-wwm-ext-large | 72.3 / 89.4 | -- / -- |
| chinese-macbert-large | 72.3 / 89.6 |	77.2 / 91.5 |

