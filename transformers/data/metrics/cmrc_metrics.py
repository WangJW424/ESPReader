# -*- coding: utf-8 -*-
'''
Evaluation script for CMRC 2018
version: v5 - special
Note:
v5 - special: Evaluate on SQuAD-style CMRC 2018 Datasets
v5: formatted output, add usage description
v4: fixed segmentation issues
'''
from __future__ import print_function
from collections import Counter, OrderedDict
import string
import re
import argparse
import json
import nltk
import sys
from transformers.tokenization_bert import whitespace_tokenize
import pdb


# split Chinese with English
def mixed_segmentation(in_str, rm_punc=False):
    #in_str = str(in_str).decode('utf-8').lower().strip()
    in_str = in_str.lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
               '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
               '「','」','（','）','－','～','『','』']
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                ss = nltk.word_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    #handling last part
    if temp_str != "":
        ss = nltk.word_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


# remove punctuation
def remove_punctuation(in_str):
    #in_str = str(in_str).decode('utf-8').lower().strip()
    in_str = in_str.lower().strip()
    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
               '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
               '「','」','（','）','－','～','『','』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


# find longest common string
def find_lcs(s1, s2):
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1][j+1] = m[i][j]+1
                if m[i+1][j+1] > mmax:
                    mmax=m[i+1][j+1]
                    p=i+1
    return s1[p-mmax:p], mmax

#
def evaluate(ground_truth_file, prediction_file):
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    for instance in ground_truth_file["data"]:
        for para in instance["paragraphs"]:
            for qas in para['qas']:
                total_count += 1
                query_id    = qas['id'].strip()
                query_text  = qas['question'].strip()
                answers 	= [x["text"] for x in qas['answers']]

                if query_id not in prediction_file:
                    sys.stderr.write('Unanswered question: {}\n'.format(query_id))
                    skip_count += 1
                    continue

                #prediction 	= str(prediction_file[query_id]).decode('utf-8')
                prediction = str(prediction_file[query_id])
                f1 += calc_f1_score(answers, prediction)
                em += calc_em_score(answers, prediction)

    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return f1_score, em_score, total_count, skip_count


def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = mixed_segmentation(ans, rm_punc=True)
        prediction_segs = mixed_segmentation(prediction, rm_punc=True)
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision 	= 1.0*lcs_len/len(prediction_segs)
        recall 		= 1.0*lcs_len/len(ans_segs)
        f1 			= (2*precision*recall)/(precision+recall)
        f1_scores.append(f1)
    return max(f1_scores)


def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    # if em!=1:
        # print('{0} vs {1}'.format([remove_punctuation(ans) for ans in answers], remove_punctuation(prediction)))
    return em


def get_error_type(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            return 0
    if em != 1:
        # print('{0} vs {1}'.format([remove_punctuation(ans) for ans in answers], remove_punctuation(prediction)))
        prediction_ = ''.join(i for i in whitespace_tokenize(remove_punctuation(prediction)))
        in_ans = False
        include_ans = False
        cross = False
        for ans in answers:
            ans_ = ''.join(i for i in whitespace_tokenize(remove_punctuation(ans)))
            if prediction_ in ans_:
                in_ans = True
                cross = False
                break
            elif ans_ in prediction_:
                include_ans = True
                cross = False
                break
            shorter = ans_ if len(ans_) < len(prediction_) else prediction_
            longer = prediction_ if shorter == ans_ else ans_
            for i in range(len(shorter)-1):
                if shorter[i:i+2] in longer:
                    cross = True
                    break

        if in_ans:
            print(prediction_, ans_)
            return 1
        elif include_ans:
            return 2
        elif cross:
            return 3
        else:
            return 4

def get_result_type_nums(ground_truth_file, prediction_file):
    total_count = 0
    skip_count = 0
    result_num_list = [0, 0, 0, 0, 0]
    for instance in ground_truth_file["data"]:
        #context_id   = instance['context_id'].strip()
        #context_text = instance['context_text'].strip()
        for para in instance["paragraphs"]:
            for qas in para['qas']:
                total_count += 1
                query_id    = qas['id'].strip()
                query_text  = qas['question'].strip()
                answers 	= [x["text"] for x in qas['answers']]

                if query_id not in prediction_file:
                    sys.stderr.write('Unanswered question: {}\n'.format(query_id))
                    skip_count += 1
                    continue

                #prediction 	= str(prediction_file[query_id]).decode('utf-8')
                prediction = str(prediction_file[query_id])
                error_type = get_error_type(answers, prediction)
                result_num_list[error_type] += 1
    print(result_num_list)
    return result_num_list



def cmrc_evaluate(dataset_file, prediction_file, glob_step=''):
    ground_truth_file = json.load(open(dataset_file, 'r', encoding='utf-8'))
    prediction_file = json.load(open(prediction_file, 'r', encoding='utf-8'))
    F1, EM, TOTAL, SKIP = evaluate(ground_truth_file, prediction_file)
    AVG = (EM + F1) * 0.5
    output_result = OrderedDict()
    output_result['AVERAGE'] = '%.5f' % AVG
    output_result['F1'] = '%.5f' % F1
    output_result['EM'] = '%.5f' % EM
    output_result['TOTAL'] = TOTAL
    output_result['SKIP'] = SKIP
    print(json.dumps(output_result, ensure_ascii=False))
    return output_result

if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='Evaluation Script for CMRC 2018')
    #parser.add_argument('dataset_file', help='Official dataset file')
    #parser.add_argument('prediction_file', help='Your prediction File')
    #args = parser.parse_args()
    dataset_file = 'D:/code/sagbert-master/transformers/data/cmrc/cmrc2018_dev.json'
    #prediction_file = 'D:\code\slignet\cmrc\macbert\cmrc_chinese-macbert-base_lr5e-5_len512_bs24_ep2_wm01_fp24_beta0_82_similarity_gpu/predictions_.json'
    macbert_baseline_file = 'D:\code\sagbert-master\chinese-macbert-base\cmrc_chinese-macbert-base_lr3e-5_len512_bs64_ep2_wm01_fp32_gpu_baseline/predictions_.json'
    bert_baseline_file = 'D:/code/sagbert-master/cmrc/bert_baseline/cmrc_bert-base-chinese_lr5e-5_len512_bs24_ep2_wm01_fp32_gpu/predictions_.json'
    macbert_prediction_file = 'D:\code\slignet\chinese-macbert-base\cmrc_chinese-macbert-base_lr5e-5_len512_bs24_ep2_wm01_fp24_82_similarity2_sentence32_gpu/predictions_.json'
    bert_prediction_file ='D:\code\slignet/bert-base-chinese\cmrc_bert-base-chinese_lr5e-5_len512_bs24_ep2_wm01_fp32_82_similarity2_sentence32_gpu/predictions_.json'
    #prediction_file ='D:\code\sagbert-master\cmrc\macbert\cmrc_chinese-macbert-base_lr5e-5_len512_bs32_ep2_wm128_fp32_gpu/predictions_.json'
    ground_truth_file = json.load(open(dataset_file, 'r', encoding='utf-8'))
    baseline_file = json.load(open(macbert_baseline_file, 'r', encoding='utf-8'))
    prediction_file = json.load(open(macbert_prediction_file, 'r', encoding='utf-8'))

    F1, EM, TOTAL, SKIP = evaluate(ground_truth_file, baseline_file)
    get_result_type_nums(ground_truth_file, baseline_file)
    AVG = (EM+F1)*0.5
    output_result = OrderedDict()
    output_result['AVERAGE'] = '%.5f' % AVG
    output_result['F1'] = '%.5f' % F1
    output_result['EM'] = '%.5f' % EM
    output_result['TOTAL'] = TOTAL
    output_result['SKIP'] = SKIP
    print(json.dumps(output_result, ensure_ascii=False))

