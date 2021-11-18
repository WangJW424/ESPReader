from tqdm import tqdm
import logging
import os
import json
import numpy as np
from transformers.tokenization_bert import whitespace_tokenize
from .utils import DataProcessor
from transformers.file_utils import is_tf_available, is_torch_available

short_set = {'<.', 'co.', 'de.', 'ho.', 'ed.', 'cf.', 'vs.', 'ms.', 'cn.', 'ar.', 'lt.', 'pt.', 'ka.', 'mt.', 'jr.', 'ar.', 'sr.', 'ch.', 'oz.', 'th.', 'sq.','cc.','pp.', 'fr.', 'ss.', 'ie.', 'ft.', 'tr.', 'cf.', 'ca.', 'al.', 'mr.', 'pg.', 'st.', 'pl.', 'nw.', 'gr.', 'bu.', 'dr.', 'ad.', 'sp.', 'rep.', 'ltd.', 'spp.', 'mrs.', 'www.', 'sir.', 'gov.', 'nic.', 'mac.', 'inc.', 'aug.', 'sgt.', 'lit.', 'maj.', 'col.', 'vol.', 'nun.', 'gen.', 'hen.', 'ave.', 'dow.', 'nov.', 'etc.', 'oct.', 'tri.', 'rev.', 'aka.', 'soc.', 'goa.', 'jan.', 'dec.', 'viz.', 'mag.'}

sentence_end_tokens = ['.', '?', '!']
bracket_end_tokens = [']', ')', '}']
comma_tokens = {','}
MIN_SENTENCE_LEN = 6
MAX_SENTENCE_LEN = 64


def _is_decimal(front, rear):
    import re
    full = front[-1]+'.'+rear[0]
    part = '.'+rear
    dnumre = re.compile(r'^[0-9]\.[0-9]+$')
    if dnumre.search(full) or dnumre.search(part):
        return True
    return False



def _is_short(front):
    tmp = front + '.'
    if tmp in short_set:
        return True
    if len(front) == 1 and 'a' <= front <= 'z':
        return True
    if front[:2] == '##':
        return False
    return False


def _is_sentence_end_mark(tok_txt, front, rear):
    if tok_txt == '.':
        if _is_decimal(front, rear):
            return False
        if _is_short(front):
            return False
        if rear == 'com':
            return False
    if rear in sentence_end_tokens+bracket_end_tokens:
        return False
    return True

def _is_sentence_end_bracket(tok_txt, front, rear):
    if front in sentence_end_tokens and rear not in sentence_end_tokens+[',']:
        return True
    return False

if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def squad_convert_examples_to_features(
    examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training, max_sentence_num=32, return_dataset=False, regression=False, pq_end=False,
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset

    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features( 
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """
    # Defining helper methods
    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(tqdm(examples, desc="Converting examples to features")):
        if is_training and not example.is_impossible:
            # Get start and end position
            start_position = example.start_position
            end_position = example.end_position

            # If the answer cannot be found in the text, then skip this example.
            actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
            cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
            if actual_text.find(cleaned_answer_text) == -1:
                logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                continue

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
            )

        spans = []

        truncated_query = tokenizer.encode(
            example.question_text, add_special_tokens=False, max_length=max_query_length
        )
        sequence_added_tokens = (
            tokenizer.max_len - tokenizer.max_len_single_sentence + 1
            if "roberta" in str(type(tokenizer))
            else tokenizer.max_len - tokenizer.max_len_single_sentence
        )
        sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

        span_doc_tokens = all_doc_tokens
        while len(spans) * doc_stride < len(all_doc_tokens):

            encoded_dict = tokenizer.encode_plus(
                truncated_query if tokenizer.padding_side == "right" else span_doc_tokens,
                span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
                max_length=max_seq_length,
                return_overflowing_tokens=True,
                pad_to_max_length=True,
                stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
                truncation_strategy="only_second" if tokenizer.padding_side == "right" else "only_first",
            )

            paragraph_len = min(
                len(all_doc_tokens) - len(spans) * doc_stride,
                max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
            )

            if tokenizer.pad_token_id in encoded_dict["input_ids"]:
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                non_padded_ids = encoded_dict["input_ids"]

            tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

            token_to_orig_map = {}
            for i in range(paragraph_len):
                index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
                token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

            encoded_dict["paragraph_len"] = paragraph_len
            encoded_dict["tokens"] = tokens
            encoded_dict["token_to_orig_map"] = token_to_orig_map
            encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
            encoded_dict["token_is_max_context"] = {}
            encoded_dict["start"] = len(spans) * doc_stride
            encoded_dict["length"] = paragraph_len

            spans.append(encoded_dict)

            if "overflowing_tokens" not in encoded_dict:
                break
            span_doc_tokens = encoded_dict["overflowing_tokens"]

        for doc_span_index in range(len(spans)):
            for j in range(spans[doc_span_index]["paragraph_len"]):
                is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
                index = (
                    j
                    if tokenizer.padding_side == "left"
                    else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
                )
                spans[doc_span_index]["token_is_max_context"][index] = is_max_context

        for span in spans:
            # Identify the position of the CLS token
            cls_index = span["input_ids"].index(tokenizer.cls_token_id)

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = np.array(span["token_type_ids"])

            p_mask = np.minimum(p_mask, 1)

            if tokenizer.padding_side == "right":
                # Limit positive values to one
                p_mask = 1 - p_mask

            p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1

            # Set the CLS index to '0'
            p_mask[cls_index] = 0

            span_is_impossible = example.is_impossible
            # if example.qas_id == "5a8d7bf7df8bba001a0f9ab2":
            #     print("hello")
            # if span_is_impossible:
            #     print("True")

            start_position = 0
            end_position = 0
            answer_sentence_positions = set()
            answer_sentence_positions.add(0)

            token_to_sentence_offset = []
            sentence_split_points = []
            # tokens_txt = tokenizer.convert_ids_to_tokens(span['input_ids'])
            sentence_id = 0
            # quotation_matched = False
            query_checked = False
            sentence_len = 0
            min_sentence_len = MIN_SENTENCE_LEN
            max_sentence_len = MAX_SENTENCE_LEN
            last_is_end = False
            for (idx, tok_id) in enumerate(span['input_ids']):
                if sentence_id >= max_sentence_num-1:
                    if tok_id != tokenizer.sep_token_id:
                        token_to_sentence_offset.append(sentence_id)
                        continue
                    token_to_sentence_offset.append(sentence_id)
                    sentence_split_points.append(idx)
                    break

                front = None
                rear = None
                if idx > 0:
                    front = tokenizer.convert_ids_to_tokens([span['input_ids'][idx - 1]])[0]
                if idx < len(span['input_ids'])-1:
                    rear = tokenizer.convert_ids_to_tokens([span['input_ids'][idx + 1]])[0]
                tok_txt = tokenizer.convert_ids_to_tokens([tok_id])[0]

                if idx == 0:
                    token_to_sentence_offset.append(0)  # set [CLS] as sentence_0
                    sentence_split_points.append(idx)
                    sentence_id = 1
                elif tok_id == tokenizer.sep_token_id:
                    if not query_checked:
                        query_checked = True
                        token_to_sentence_offset.append(1)
                        sentence_split_points.append(idx)      # split after [SEP]
                        sentence_len = 0
                    else:
                        token_to_sentence_offset.append(token_to_sentence_offset[-1])  # end by [SEP]
                        if last_is_end:
                            sentence_split_points[-1] = idx
                        else:
                            sentence_split_points.append(idx)
                        break
                elif not rear:
                    token_to_sentence_offset.append(token_to_sentence_offset[-1])  # reach the end
                    if last_is_end:
                        sentence_split_points[-1] = idx
                    else:
                        sentence_split_points.append(idx)
                    break
                elif tok_txt in comma_tokens and query_checked:
                    if sentence_len >= max_sentence_len:
                        token_to_sentence_offset.append(sentence_id)
                        sentence_split_points.append(idx)
                        last_is_end = True
                        sentence_id += 1
                        sentence_len = 0
                    else:
                        token_to_sentence_offset.append(sentence_id)
                        last_is_end = False
                        sentence_len += 1
                elif tok_txt in sentence_end_tokens and query_checked:
                    if _is_sentence_end_mark(tok_txt, front, rear) and sentence_len >= min_sentence_len:
                        token_to_sentence_offset.append(sentence_id)
                        sentence_split_points.append(idx)
                        last_is_end = True
                        sentence_id += 1
                        sentence_len = 0
                    else:
                        token_to_sentence_offset.append(sentence_id)
                        last_is_end = False
                        sentence_len += 1
                elif tok_txt in bracket_end_tokens and query_checked:
                    if _is_sentence_end_bracket(tok_txt, front, rear) and sentence_len >= min_sentence_len:
                        token_to_sentence_offset.append(sentence_id)
                        sentence_split_points.append(idx)
                        last_is_end = True
                        sentence_id += 1
                        sentence_len = 0
                    else:
                        token_to_sentence_offset.append(sentence_id)
                        last_is_end = False
                        sentence_len += 1
                else:
                    token_to_sentence_offset.append(sentence_id)
                    last_is_end = False
                    sentence_len += 1

            total_sentence_num = token_to_sentence_offset[-1] + 1
            sentence_mask = [1] * total_sentence_num
            while len(sentence_mask) < max_sentence_num:
                sentence_mask.append(0)
                sentence_split_points.append(-1)
            while len(token_to_sentence_offset) < len(span['input_ids']):
                token_to_sentence_offset.append(-1)

            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = span["start"]
                doc_end = span["start"] + span["length"] - 1
                out_of_span = False

                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True

                if out_of_span:
                    start_position = cls_index
                    end_position = cls_index
                    answer_sentence_positions.add(cls_index)
                    span_is_impossible = True
                else:
                    if tokenizer.padding_side == "left":
                        doc_offset = 0
                    else:
                        doc_offset = len(truncated_query) + sequence_added_tokens

                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
                    for pos in range(start_position, end_position+1):
                        answer_sentence_positions.add(token_to_sentence_offset[pos])

            answer_sentence_positions = list(answer_sentence_positions)
            answer_sentence_positions.sort()
            sentence_start_position = sentence_end_position = answer_sentence_positions[0]
            if len(answer_sentence_positions) > 1:
                sentence_start_position = answer_sentence_positions[1]
                sentence_end_position = answer_sentence_positions[-1]
            question_end_index = span["truncated_query_with_special_tokens_length"] - 1 #8
            doc_end_index = question_end_index + span["paragraph_len"]
            pq_end_pos = [question_end_index, doc_end_index]
            if pq_end:
                features.append(
                    SquadFeatures(
                        span["input_ids"],
                        span["attention_mask"],
                        span["token_type_ids"],
                        sentence_mask,
                        cls_index,
                        p_mask.tolist(),
                        example_index=example_index,
                        unique_id=unique_id,
                        paragraph_len=span["paragraph_len"],
                        token_is_max_context=span["token_is_max_context"],
                        tokens=span["tokens"],
                        token_to_orig_map=span["token_to_orig_map"],
                        token_to_sentence_offset=token_to_sentence_offset,
                        sentence_split_points=sentence_split_points,
                        start_position=start_position,
                        end_position=end_position,
                        sentence_start_position=sentence_start_position,
                        sentence_end_position=sentence_end_position,
                        is_impossible=span_is_impossible,
                        pq_end_pos=pq_end_pos
                    )
                )
            else:
                features.append(
                    SquadFeatures(
                        span["input_ids"],
                        span["attention_mask"],
                        span["token_type_ids"],
                        sentence_mask,
                        cls_index,
                        p_mask.tolist(),
                        example_index=example_index,
                        unique_id=unique_id,
                        paragraph_len=span["paragraph_len"],
                        token_is_max_context=span["token_is_max_context"],
                        tokens=span["tokens"],
                        token_to_orig_map=span["token_to_orig_map"],
                        token_to_sentence_offset=token_to_sentence_offset,
                        sentence_split_points=sentence_split_points,
                        start_position=start_position,
                        end_position=end_position,
                        sentence_start_position=sentence_start_position,
                        sentence_end_position=sentence_end_position,
                        is_impossible=span_is_impossible,
                    )

                )
            unique_id += 1

    if return_dataset == "pt":
        if not is_torch_available():
            raise ImportError("Pytorch must be installed to return a pytorch dataset.")

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_sentence_masks = torch.tensor([f.sentence_mask for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_token_to_sentence_offsets = torch.tensor([f.token_to_sentence_offset for f in features], dtype=torch.long)
        all_sentence_split_points = torch.tensor([f.sentence_split_points for f in features], dtype=torch.long)

        if not is_training:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            if regression:
                all_is_impossibles = torch.tensor([int(f.is_impossible) for f in features], dtype=torch.float)
            else:
                all_is_impossibles = torch.tensor([int(f.is_impossible) for f in features], dtype=torch.long)
            if pq_end:
                all_pq_end_pos = torch.tensor([f.pq_end_pos for f in features], dtype=torch.long)
                dataset = TensorDataset(
                    all_input_ids,
                    all_attention_masks,
                    all_token_type_ids,
                    all_sentence_masks,
                    all_token_to_sentence_offsets,
                    all_sentence_split_points,
                    all_example_index,
                    all_is_impossibles,
                    all_pq_end_pos,
                    all_cls_index,
                    all_p_mask,
                )
            else:
                dataset = TensorDataset(
                    all_input_ids,
                    all_attention_masks,
                    all_token_type_ids,
                    all_sentence_masks,
                    all_token_to_sentence_offsets,
                    all_sentence_split_points,
                    all_example_index,
                    all_is_impossibles,
                    all_cls_index,
                    all_p_mask,
                )
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            all_sentence_start_positions = torch.tensor([f.sentence_start_position for f in features], dtype=torch.long)
            all_sentence_end_positions = torch.tensor([f.sentence_end_position for f in features], dtype=torch.long)
            if regression:
                all_is_impossibles = torch.tensor([int(f.is_impossible) for f in features], dtype=torch.float)
            else:
                all_is_impossibles = torch.tensor([int(f.is_impossible) for f in features], dtype=torch.long)
                print(sum(all_is_impossibles == 1), sum(all_is_impossibles == 0))
            if pq_end:
                all_pq_end_pos = torch.tensor([f.pq_end_pos for f in features], dtype=torch.long)
                dataset = TensorDataset(
                    all_input_ids,
                    all_attention_masks,
                    all_token_type_ids,
                    all_sentence_masks,
                    all_token_to_sentence_offsets,
                    all_sentence_split_points,
                    all_start_positions,
                    all_end_positions,
                    all_sentence_start_positions,
                    all_sentence_end_positions,
                    all_is_impossibles,
                    all_pq_end_pos,
                    all_cls_index,
                    all_p_mask,
                )
            else:
                dataset = TensorDataset(
                    all_input_ids,
                    all_attention_masks,
                    all_token_type_ids,
                    all_sentence_masks,
                    all_token_to_sentence_offsets,
                    all_sentence_split_points,
                    all_start_positions,
                    all_end_positions,
                    all_sentence_start_positions,
                    all_sentence_end_positions,
                    all_is_impossibles,
                    all_cls_index,
                    all_p_mask,
                )

        return features, dataset
    elif return_dataset == "tf":
        if not is_tf_available():
            raise ImportError("TensorFlow must be installed to return a TensorFlow dataset.")

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                        "sentence_mask": ex.sentence_mask,
                        # "token_to_sentence_offset": ex.token_to_sentence_offset,
                        "sentence_split_points": ex.sentence_split_points,
                    }, {
                        "start_position": ex.start_position,
                        "end_position": ex.end_position,
                        "sentence_start_position": ex.sentence_start_position,
                        "sentence_end_position": ex.sentence_end_position,
                        "cls_index": ex.cls_index,
                        "p_mask": ex.p_mask,
                    }
                )

        return tf.data.Dataset.from_generator(
            gen,
            (
                {"input_ids": tf.int32, "attention_mask": tf.int32,
                 "token_type_ids": tf.int32, "sentence_mask": tf.int32,
                 #"token_to_sentence_offset": tf.int32,
                 "sentence_split_points": tf.int64},
                {"start_position": tf.int64, "end_position": tf.int64,
                 "sentence_start_position": tf.int64, "sentence_end_position": tf.int64,
                 "cls_index": tf.int64, "p_mask": tf.int32},
            ),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                    "sentence_mask": tf.TensorShape([None]),
                    #"token_to_sentence_offset": tf.TensorShape([None]),
                    "sentence_split_points": tf.TensorShape([None]),
                },
                {
                    "start_position": tf.TensorShape([]),
                    "end_position": tf.TensorShape([]),
                    "sentence_start_position": tf.TensorShape([]),
                    "sentence_end_position": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                },
            ),
        )

    return features


class SquadProcessor(DataProcessor):
    """
    Processor for the SQuAD data set.
    Overriden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and version 2.0 of SQuAD, respectively.
    """

    train_file = None
    dev_file = None

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        if not evaluate:
            answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
            answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
            answers = []
        else:
            answers = [
                {"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
                for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text"])
            ]

            answer = None
            answer_start = None

        return SquadExample(
            qas_id=tensor_dict["id"].numpy().decode("utf-8"),
            question_text=tensor_dict["question"].numpy().decode("utf-8"),
            context_text=tensor_dict["context"].numpy().decode("utf-8"),
            answer_text=answer,
            start_position_character=answer_start,
            title=tensor_dict["title"].numpy().decode("utf-8"),
            answers=answers,
        )

    def get_examples_from_dataset(self, dataset, evaluate=False):
        """
        Creates a list of :class:`~transformers.data.processors.squad.SquadExample` using a TFDS dataset.

        Args:
            dataset: The tfds dataset loaded from `tensorflow_datasets.load("squad")`
            evaluate: boolean specifying if in evaluation mode or in training mode

        Returns:
            List of SquadExample

        Examples::

            import tensorflow_datasets as tfds
            dataset = tfds.load("squad")

            training_examples = get_examples_from_dataset(dataset, evaluate=False)
            evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
        """

        if evaluate:
            dataset = dataset["validation"]
        else:
            dataset = dataset["train"]

        examples = []
        for tensor_dict in tqdm(dataset):
            examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))

        return examples

    def get_train_examples(self, data_dir, filename=None):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """
        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "train")

    def get_dev_examples(self, data_dir, filename=None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "dev")

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False

                    if not is_impossible:
                        if is_training:
                            if len(qa["answers"]) == 0:
                                print("empty answer!!!")
                                continue
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]
                    # if is_impossible:
                    #     print(qas_id)
                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )

                    examples.append(example)
        return examples


class SquadV1Processor(SquadProcessor):
    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"


class SquadV2Processor(SquadProcessor):
    train_file = "train-v2.0.json"
    dev_file = "dev-v2.0.json"


class SquadExample(object):
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        title,
        answers=[],
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start end end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]


class SquadFeatures(object):
    """
    Single squad example features to be fed to a model.
    Those features are model-specific and can be crafted from :class:`~transformers.data.processors.squad.SquadExample`
    using the :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        sentence_mask,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        token_to_sentence_offset,
        sentence_split_points,
        start_position,
        end_position,
        sentence_start_position,
        sentence_end_position,
        is_impossible,
        pq_end_pos=None,
        tag_seq=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.sentence_mask = sentence_mask
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.token_to_sentence_offset = token_to_sentence_offset
        self.sentence_split_points = sentence_split_points
        self.start_position = start_position
        self.end_position = end_position
        self.sentence_start_position = sentence_start_position
        self.sentence_end_position = sentence_end_position
        self.is_impossible = is_impossible
        self.pq_end_pos = pq_end_pos
        self.tag_seq = tag_seq


class SquadResult(object):
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, choice_logits=None, start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id
        if choice_logits:
            self.choice_logits = choice_logits

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
        self.cls_logits = cls_logits
