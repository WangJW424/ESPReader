from .processors import InputExample, InputFeatures, DataProcessor, SquadFeatures, SingleSentenceClassificationProcessor
from .processors import squad_convert_examples_to_features, SquadExample, SquadV1Processor, SquadV2Processor
from .processors import cmrc_convert_examples_to_features, CmrcExample, CmrcFeatures, CmrcProcessor

from .metrics import is_sklearn_available
if is_sklearn_available():
    from .metrics import glue_compute_metrics, xnli_compute_metrics
