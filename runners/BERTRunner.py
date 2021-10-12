import tempfile
from typing import Iterable, Type

from allennlp.data.token_indexers.pretrained_transformer_mismatched_indexer import PretrainedTransformerMismatchedIndexer
from readers.AbstractProcessLogReader import AbstractProcessLogReader, TaskModes
from runners.SimpleRunner import SimpleRunner
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
from allennlp.modules.token_embedders import Embedding, PretrainedTransformerMismatchedEmbedder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, CnnEncoder, LstmSeq2VecEncoder
from allennlp.nn import util
from allennlp.training import Trainer, GradientDescentTrainer
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.util import evaluate


class BERT:
    TEST = "google/reformer-crime-and-punishment"
    BASE = 'bert-base-uncased'
    SMALL = 'albert-base-v2'


class BERTRunner(SimpleRunner):
    def __init__(self,
                 DatasetReaderClass: Type[AbstractProcessLogReader],
                 ModelClass: Type[Model],
                 task_mode: TaskModes = TaskModes.SIMPLE,
                 model_name=BERT.SMALL) -> None:
        super().__init__(DatasetReaderClass, ModelClass, task_mode=task_mode)
        self.transformer_model = model_name

    def build_dataset_reader(self, DatasetReaderClass: Type[AbstractProcessLogReader]) -> DatasetReader:
        return DatasetReaderClass(
            mode=self.task_mode,
            token_indexers=PretrainedTransformerMismatchedIndexer(model_name=self.transformer_model),
        )

    def build_model(self, vocab: Vocabulary, ModelClassifier: Type[Model]) -> Model:
        print("Building the model")

        vocab_size = vocab.get_vocab_size("tokens")

        embedding_size = {
            BERT.BASE: 768,
            BERT.SMALL: 768,
        }.get(self.transformer_model, 10)
        pretrained_embedding = PretrainedTransformerMismatchedEmbedder(model_name=self.transformer_model)
        embedder = BasicTextFieldEmbedder(token_embedders={"tokens": pretrained_embedding})
        encoder = BagOfEmbeddingsEncoder(embedding_dim=embedding_size)
        # encoder = LstmSeq2VecEncoder(input_size=embedding_size, hidden_size=5, num_layers=1)
        # encoder = CnnEncoder(embedding_dim=embedding_size, num_filters=2, ngram_filter_sizes=(2, 3, 4))
        return ModelClassifier(vocab, embedder, encoder)

    # def build_vocab(self, instances: Iterable[Instance]) -> Vocabulary:
    #     print("Building the vocabulary")
    #     return Vocabulary()

