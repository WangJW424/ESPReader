from tqdm import tqdm
import logging
import os
import json
import numpy as np
from transformers.tokenization_bert import whitespace_tokenize
from .utils import DataProcessor
from transformers import tokenization
from transformers.file_utils import is_tf_available, is_torch_available
import torch
from torch.utils.data import TensorDataset

sentence_end_tokens = ['。', '！', '？', '……', '，']
bracket_end_tokens = ['」','）']
quotation_tokens =['’', '”']
comma_tokens = ['，']

MIN_SENTENCE_LEN = 6
MAX_SENTENCE_LEN = 48
logger = logging.getLogger(__name__)


def _is_sentence_end_mark(tok_txt, front, rear):
  if rear in sentence_end_tokens + bracket_end_tokens:
    return False
  return True


def _is_sentence_end_bracket(tok_txt, front, rear):
  if front in sentence_end_tokens and rear not in sentence_end_tokens:
    return True
  return False

def _is_sentence_end_quotation(front, rear, quotation_matched):
  if quotation_matched:
    quotation_matched=False
    is_end = False
  else:
    quotation_matched=True
    if front in sentence_end_tokens and rear not in sentence_end_tokens:
      is_end = True
  return (is_end, quotation_matched)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""

  # The SQuAD annotations are character based. We first project them to
  # whitespace-tokenized words. But then after WordPiece tokenization, we can
  # often find a "better match". For example:
  #
  #   Question: What year was John Smith born?
  #   Context: The leader was John Smith (1895-1943).
  #   Answer: 1895
  #
  # The original whitespace-tokenized answer will be "(1895-1943).". However
  # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
  # the exact answer, 1895.
  #
  # However, this is not always possible. Consider the following:
  #
  #   Question: What country is the top exporter of electornics?
  #   Context: The Japanese electronics industry is the lagest in the world.
  #   Answer: Japan
  #
  # In this case, the annotator chose "Japan" as a character sub-span of
  # the word "Japanese". Since our WordPiece tokenizer does not split
  # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
  # in SQuAD, but does happen.
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
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

class CmrcExample(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               question_text,
               doc_tokens,
               answer_text=None,
               start_position=None,
               end_position=None,
               answers=[]):
    self.qas_id = qas_id
    self.question_text = question_text
    self.doc_tokens = doc_tokens
    self.answer_text = answer_text
    self.start_position = start_position
    self.end_position = end_position
    self.answers = answers

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position)
    if self.start_position:
      s += ", end_position: %d" % (self.end_position)
    return s


class CmrcFeatures(object):
  """A single set of features of data."""

  def __init__(self,
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
               pq_end_pos=None,
               tag_seq=None,):
    self.unique_id = unique_id
    self.example_index = example_index
    self.tokens = tokens
    self.token_to_orig_map = token_to_orig_map
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.sentence_mask = sentence_mask
    self.cls_index = cls_index
    self.p_mask = p_mask
    self.paragraph_len = paragraph_len
    self.attention_mask = attention_mask
    self.token_type_ids = token_type_ids
    self.token_to_sentence_offset = token_to_sentence_offset
    self.sentence_split_points = sentence_split_points
    self.start_position = start_position
    self.end_position = end_position
    self.sentence_start_position = sentence_start_position
    self.sentence_end_position = sentence_end_position
    self.pq_end_pos = pq_end_pos
    self.tag_seq = tag_seq

def customize_tokenizer(text):
  tokenizer = tokenization.BasicTokenizer(do_lower_case=False)
  temp_x = ""
  text = tokenization.convert_to_unicode(text)
  for c in text:
    if tokenizer._is_chinese_char(ord(c)) or tokenization._is_punctuation(c) or tokenization._is_whitespace(c) or tokenization._is_control(c):
      temp_x += " " + c + " "
    else:
      temp_x += c
  return temp_x.split()


#
class ChineseFullTokenizer(object):
  """Runs end-to-end tokenziation."""

  def __init__(self, vocab_file, do_lower_case=False):
    self.vocab = tokenization.load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.wordpiece_tokenizer = tokenization.WordpieceTokenizer(vocab=self.vocab)
    self.do_lower_case = do_lower_case

  def tokenize(self, text):
    split_tokens = []
    for token in customize_tokenizer(text):
      for sub_token in self.wordpiece_tokenizer.tokenize(token):
        split_tokens.append(sub_token)

    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    return tokenization.convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    return tokenization.convert_by_vocab(self.inv_vocab, ids)


class CmrcProcessor(DataProcessor):
  """
  Processor for the Cmrc data set.
  """
  def __init__(self):
    self.train_file = "cmrc2018_train.json"
    self.dev_file = "cmrc2018_dev.json"

  def read_cmrc_examples(self, input_data, set_type):
    """Read a SQuAD json file into a list of SquadExample."""

    is_training = set_type == "train"
    examples = []
    for entry in tqdm(input_data):
      for paragraph in entry["paragraphs"]:
        paragraph_text = paragraph["context"]
        raw_doc_tokens = customize_tokenizer(paragraph_text)
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        k = 0
        temp_word = ""
        for c in paragraph_text:
          if tokenization._is_whitespace(c):
            char_to_word_offset.append(k - 1)
            continue
          else:
            temp_word += c
            char_to_word_offset.append(k)
          if temp_word == raw_doc_tokens[k]:
            doc_tokens.append(temp_word)
            temp_word = ""
            k += 1

        assert k == len(raw_doc_tokens)

        for qa in paragraph["qas"]:
          qas_id = qa["id"]
          question_text = qa["question"]
          if question_text == "":
            continue
          start_position = None
          end_position = None
          answer_text = None
          answers = []

          if is_training:
            answer = qa["answers"][0]
            answer_text = answer["text"]

            if answer_text not in paragraph_text:
              logger.warning("Could not find answer")
              print(answer_text)
              print(paragraph_text)
            else:
              answer_offset = paragraph_text.index(answer_text)
              answer_length = len(answer_text)
              start_position = char_to_word_offset[answer_offset]
              end_position = char_to_word_offset[answer_offset + answer_length - 1]

              # Only add answers where the text can be exactly recovered from the
              # document. If this CAN'T happen it's likely due to weird Unicode
              # stuff so we will just skip the example.
              #
              # Note that this means for training mode, every example is NOT
              # guaranteed to be preserved.
              actual_text = "".join(
                doc_tokens[start_position:(end_position + 1)])
              cleaned_answer_text = "".join(whitespace_tokenize(answer_text))
              if actual_text.find(cleaned_answer_text) == -1:
                #pdb.set_trace()
                logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                continue
          else:
            answers = qa['answers']

          example = CmrcExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            answer_text=answer_text,
            start_position=start_position,
            end_position=end_position,
            answers=answers)
          examples.append(example)
    logger.info("**********read_cmrc_examples complete!**********")

    return examples


  def get_train_examples(self, data_dir, filename=None):
    """
    Returns the training examples from the data directory.

    Args:
        data_dir: Directory containing the data files used for training and evaluating.
        filename: None by default, specify this if the evaluation file has a different name than the original one
            which is `cmrc2018_train.json`.
    """
    if data_dir is None:
      data_dir = ""

    if self.train_file is None:
      raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

    with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
    ) as reader:
      input_data = json.load(reader)["data"]
    return self.read_cmrc_examples(input_data, "train")


  def get_dev_examples(self, data_dir, filename=None):
    """
    Returns the evaluation example from the data directory.

    Args:
        data_dir: Directory containing the data files used for training and evaluating.
        filename: None by default, specify this if the evaluation file has a different name than the original one
            which is `cmrc2018_dev.json`.
    """

    if data_dir is None:
      data_dir = ""

    if self.dev_file is None:
      raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

    with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
    ) as reader:
      input_data = json.load(reader)["data"]
    return self.read_cmrc_examples(input_data, "dev")

def cmrc_convert_examples_to_features(examples, tokenizer, max_seq_length,
                                      doc_stride, max_query_length, is_training,
                                      max_sentence_num=48, return_dataset=False, pq_end=False
                                      ):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1000000000

  features = []
  for (example_index, example) in enumerate(tqdm(examples, desc="Converting examples to features")):
    if is_training:
      # Get start and end position
      start_position = example.start_position
      end_position = example.end_position

      # If the answer cannot be found in the text, then skip this example.
      actual_text = "".join(example.doc_tokens[start_position: (end_position + 1)])
      cleaned_answer_text = "".join(whitespace_tokenize(example.answer_text))
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

    if is_training:
      tok_start_position = orig_to_tok_index[example.start_position]
      if example.end_position < len(example.doc_tokens) - 1:
        tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
      else:
        tok_end_position = len(all_doc_tokens) - 1
      (tok_start_position, tok_end_position) = _improve_answer_span(
          all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
          example.answer_text)

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
    while len(spans)*doc_stride < len(all_doc_tokens):
      #print('question_text:', example.question_text)
      #print('truncated_query:', truncated_query)
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
      # max_sentence_len = MAX_SENTENCE_LEN
      last_is_end = False
      quotation_matched = True
      true_token_num = 0
      for (idx, tok_id) in enumerate(span['input_ids']):
        if sentence_id >= max_sentence_num - 1:
          # 句子数量达到设定最大值，剩下非[PAD]token视为一个句子
          if tok_id != tokenizer.sep_token_id:
            token_to_sentence_offset.append(sentence_id)
            true_token_num += 1
            continue
          token_to_sentence_offset.append(sentence_id)
          sentence_split_points.append(idx)
          true_token_num += 1
          break

        front = None
        rear = None
        if idx > 0:
          front = tokenizer.convert_ids_to_tokens([span['input_ids'][idx - 1]])[0]
        if idx < len(span['input_ids']) - 1:
          rear = tokenizer.convert_ids_to_tokens([span['input_ids'][idx + 1]])[0]
        tok_txt = tokenizer.convert_ids_to_tokens([tok_id])[0]

        if idx == 0:
          token_to_sentence_offset.append(0)  # set [CLS] as sentence_0
          true_token_num += 1
          sentence_split_points.append(idx)
          sentence_id = 1
        elif tok_id == tokenizer.sep_token_id:
          if not query_checked:
            query_checked = True
            token_to_sentence_offset.append(1)
            sentence_split_points.append(idx)  # split after [SEP]
            sentence_id = 2
            sentence_len = 0
            true_token_num += 1
          else:
            token_to_sentence_offset.append(token_to_sentence_offset[-1])  # end by [SEP]
            true_token_num += 1
            if last_is_end:
              sentence_split_points[-1] = idx
            else:
              sentence_split_points.append(idx)
            break
        elif not rear:
          token_to_sentence_offset.append(token_to_sentence_offset[-1])  # reach the end
          true_token_num += 1
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
            true_token_num += 1
          else:
            token_to_sentence_offset.append(sentence_id)
            last_is_end = False
            true_token_num += 1
            sentence_len += 1
        elif tok_txt in bracket_end_tokens and query_checked:
          if _is_sentence_end_bracket(tok_txt, front, rear) and sentence_len >= min_sentence_len:
            token_to_sentence_offset.append(sentence_id)
            sentence_split_points.append(idx)
            last_is_end = True
            sentence_id += 1
            sentence_len = 0
            true_token_num += 1
          else:
            token_to_sentence_offset.append(sentence_id)
            last_is_end = False
            sentence_len += 1
            true_token_num += 1
        elif tok_txt in quotation_tokens and query_checked:
          is_end, quotation_matched = _is_sentence_end_quotation(front, rear, quotation_matched)
          if is_end and sentence_len >= min_sentence_len:
            token_to_sentence_offset.append(sentence_id)
            sentence_split_points.append(idx)
            last_is_end = True
            sentence_id += 1
            sentence_len = 0
            true_token_num += 1
          else:
            token_to_sentence_offset.append(sentence_id)
            last_is_end = False
            sentence_len += 1
            true_token_num += 1
        else:
          token_to_sentence_offset.append(sentence_id)
          last_is_end = False
          sentence_len += 1
          true_token_num += 1


      total_sentence_num = token_to_sentence_offset[-1] + 1
      sentence_mask = [1] * total_sentence_num
      while len(sentence_mask) < max_sentence_num:
        sentence_mask.append(0)
        sentence_split_points.append(-1)
      while len(token_to_sentence_offset) < len(span['input_ids']):
        token_to_sentence_offset.append(-1)
      for i in range(len(token_to_sentence_offset)):
        if token_to_sentence_offset[i] < 0:
          token_to_sentence_offset[i] = token_to_sentence_offset[i-1]
      span_is_impossible = False
      if is_training:
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
          for pos in range(start_position, end_position + 1):
            answer_sentence_positions.add(token_to_sentence_offset[pos])

      answer_sentence_positions = list(answer_sentence_positions)
      answer_sentence_positions.sort()
      sentence_start_position = sentence_end_position = answer_sentence_positions[0]
      if len(answer_sentence_positions) > 1:
        sentence_start_position = answer_sentence_positions[1]
        sentence_end_position = answer_sentence_positions[-1]
      question_end_index = span["truncated_query_with_special_tokens_length"] - 1  # 8
      doc_end_index = question_end_index + span["paragraph_len"]
      pq_end_pos = [question_end_index, doc_end_index]
      if not span_is_impossible:
        if pq_end:
          features.append(
            CmrcFeatures(
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
              pq_end_pos=pq_end_pos
            )
          )
        else:
          features.append(
            CmrcFeatures(
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
              pq_end_pos=pq_end_pos
            )
          )
        unique_id +=1

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
          all_pq_end_pos,
          all_cls_index,
          all_p_mask
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
          all_cls_index,
          all_p_mask
        )
    else:
      all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
      all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
      all_sentence_start_positions = torch.tensor([f.sentence_start_position for f in features], dtype=torch.long)
      all_sentence_end_positions = torch.tensor([f.sentence_end_position for f in features], dtype=torch.long)
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
          all_cls_index,
          all_p_mask,
        )

    return features, dataset
  return features

class CmrcResult(object):
  """
  Constructs a CmrcResult which can be used to evaluate a model's output on the Cmrc dataset.

  Args:
      unique_id: The unique identifier corresponding to that example.
      start_logits: The logits corresponding to the start of the answer
      end_logits: The logits corresponding to the end of the answer
  """

  def __init__(self, unique_id, start_logits, end_logits, choice_logits=None, start_top_index=None,
               end_top_index=None, cls_logits=None):
    self.start_logits = start_logits
    self.end_logits = end_logits
    self.unique_id = unique_id
    #if choice_logits:
      #self.choice_logits = choice_logits

    if start_top_index:
      self.start_top_index = start_top_index
      self.end_top_index = end_top_index
    self.cls_logits = cls_logits



