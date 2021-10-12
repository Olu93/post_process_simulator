# %%
import tempfile
from typing import Dict, Iterable, List, Tuple
import torch
from models.SimpleClassifier import SimpleClassifier
from readers.AbstractProcessLogReader import DatasetModes, TaskModes
from readers.BPIC12 import BPIC12W
from readers.RequestForPaymentLogReader import RequestForPaymentLogReader
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, CnnEncoder, LstmSeq2VecEncoder
from allennlp.nn import util
from allennlp.training import Trainer, GradientDescentTrainer
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.util import evaluate

from runners.SimpleRunner import SimpleRunner
from runners.BERTRunner import BERT, BERTRunner


task_mode = TaskModes.EXTENSIVE
model_name = BERT.SMALL
# model, reader = BERTRunner(RequestForPaymentLogReader, SimpleClassifier, task_mode=task_mode, model_name=model_name).run_training_loop()
model, reader = BERTRunner(BPIC12W, SimpleClassifier, task_mode=task_mode, model_name=model_name).run_training_loop()
# model, reader = SimpleRunner(RequestForPaymentLogReader, SimpleClassifier, task_mode=task_mode).run_training_loop()
# model, reader = SimpleRunner(BPIC12W, SimpleClassifier, task_mode=TaskModes.SIMPLE).run_training_loop()
# model, reader = SimpleRunner(BPIC12W, SimpleClassifier, task_mode=TaskModes.EXTENSIVE).run_training_loop()
# model, reader = SimpleRunner(BPIC12W, SimpleClassifier, task_mode=TaskModes.EXTENSIVE_RANDOM).run_training_loop()
# %%
model
# %%
