

from typing import Type

from allennlp.data.token_indexers.pretrained_transformer_mismatched_indexer import PretrainedTransformerMismatchedIndexer
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
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, CnnEncoder, LstmSeq2VecEncoder
from allennlp.nn import util
from allennlp.training import Trainer, GradientDescentTrainer
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.util import evaluate

class BERTRunner(SimpleRunner):
    def build_model(self, vocab: Vocabulary, ModelClassifier:Type[Model]) -> Model:
        print("Building the model")
        transformer_model = "google/reformer-crime-and-punishment"

        vocab_size = vocab.get_vocab_size("tokens")
        embedding_size = 10

        embedder = BasicTextFieldEmbedder(
            {"tokens": Embedding(embedding_dim=embedding_size, num_embeddings=vocab_size)})
        # encoder = BagOfEmbeddingsEncoder(embedding_dim=embedding_size)
        encoder = LstmSeq2VecEncoder(input_size=embedding_size, hidden_size=5, num_layers=1)
        # encoder = CnnEncoder(embedding_dim=embedding_size, num_filters=2, ngram_filter_sizes=(2, 3, 4))
        return ModelClassifier(vocab, embedder, encoder)