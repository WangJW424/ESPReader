__version__ = "2.3.0"

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
    absl.logging.set_verbosity('info')
    absl.logging.set_stderrthreshold('info')
    absl.logging._warn_preinit_stderr = False
except:
    pass

import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# Files and general utilities
from .file_utils import (TRANSFORMERS_CACHE, PYTORCH_TRANSFORMERS_CACHE, PYTORCH_PRETRAINED_BERT_CACHE,
                         cached_path, add_start_docstrings, add_end_docstrings,
                         WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME, CONFIG_NAME, MODEL_CARD_NAME,
                         is_tf_available, is_torch_available)

from transformers.data import (is_sklearn_available,
                               InputExample, InputFeatures, DataProcessor,
                               SingleSentenceClassificationProcessor,
                               squad_convert_examples_to_features, SquadFeatures, SquadExample, SquadV1Processor, SquadV2Processor,
                               cmrc_convert_examples_to_features, CmrcFeatures, CmrcExample, CmrcProcessor
                               )



# Tokenizers
from .tokenization_utils import (PreTrainedTokenizer)
from .tokenization_auto import AutoTokenizer
from .tokenization_bert import BertTokenizer, BasicTokenizer, WordpieceTokenizer



# Configurations
from .configuration_utils import PretrainedConfig
from .configuration_auto import AutoConfig, ALL_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_bert import BertConfig, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

# Modeling
if is_torch_available():
    from .modeling_utils import (PreTrainedModel, prune_layer, Conv1D)
    from .modeling_auto import (AutoModel, AutoModelForSequenceClassification, AutoModelForQuestionAnswering,
                                AutoModelWithLMHead, AutoModelForTokenClassification, ALL_PRETRAINED_MODEL_ARCHIVE_MAP)

    from .modeling_bert import (BertPreTrainedModel, BertModel, BertForPreTraining,
                                BertForMaskedLM, BertForNextSentencePrediction,
                                BertForSequenceClassification, BertForMultipleChoice,
                                BertForTokenClassification, BertForQuestionAnswering,
                                BertForQuestionAnsweringESP, BertForQuestionAnsweringESP_V2,
                                load_tf_weights_in_bert, BERT_PRETRAINED_MODEL_ARCHIVE_MAP)


    # Optimization
    from .optimization import (AdamW, get_constant_schedule, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup,
                               get_cosine_with_hard_restarts_schedule_with_warmup, get_linear_schedule_with_warmup)




if not is_torch_available():
    logger.warning("PyTorch have not  been found."
                   "Models won't be available and only tokenizers, configuration"
                   "and file/data utilities can be used.")
