# %%
import tempfile
from typing import Dict, Iterable, List, Tuple, Type
import torch
from readers.AbstractProcessLogReader import Dataset, Modes
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


class SimpleRunner():


    def build_dataset_reader(self) -> DatasetReader:
        return RequestForPaymentLogReader(mode=Modes.EXTENSIVE)
        # return BPIC12W(mode=Modes.EXTENSIVE, debug=True)

    def read_data(self, reader: DatasetReader) -> Tuple[List[Instance], List[Instance]]:
        print("Reading data")
        training_data = list(reader.read(Dataset.TRAIN))
        validation_data = list(reader.read(Dataset.VAL))
        return training_data, validation_data

    def build_vocab(self, instances: Iterable[Instance]) -> Vocabulary:
        print("Building the vocabulary")
        return Vocabulary.from_instances(instances)

    def build_model(self, vocab: Vocabulary, ModelClassifier:Type[Model]) -> Model:
        print("Building the model")
        vocab_size = vocab.get_vocab_size("tokens")
        embedding_size = 10
        embedder = BasicTextFieldEmbedder(
            {"tokens": Embedding(embedding_dim=embedding_size, num_embeddings=vocab_size)})
        # encoder = BagOfEmbeddingsEncoder(embedding_dim=embedding_size)
        encoder = LstmSeq2VecEncoder(input_size=embedding_size, hidden_size=5, num_layers=1)
        # encoder = CnnEncoder(embedding_dim=embedding_size, num_filters=2, ngram_filter_sizes=(2, 3, 4))
        return ModelClassifier(vocab, embedder, encoder)

    def build_data_loaders(
        self,
        train_data: List[Instance],
        dev_data: List[Instance],
    ) -> Tuple[DataLoader, DataLoader]:
        train_loader = SimpleDataLoader(train_data, 8, shuffle=True)
        dev_loader = SimpleDataLoader(dev_data, 8, shuffle=False)
        return train_loader, dev_loader

    def build_trainer(
        self,
        model: Model,
        serialization_dir: str,
        train_loader: DataLoader,
        dev_loader: DataLoader,
    ) -> Trainer:
        parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        optimizer = AdamOptimizer(parameters)  # type: ignore
        trainer = GradientDescentTrainer(
            model=model,
            serialization_dir=serialization_dir,
            data_loader=train_loader,
            validation_data_loader=dev_loader,
            num_epochs=10,
            optimizer=optimizer,
        )
        return trainer

    def run_training_loop(self, ModelClassifier:Type[Model]):
        dataset_reader = self.build_dataset_reader()

        train_data, dev_data = self.read_data(dataset_reader)

        vocab = self.build_vocab(train_data + dev_data)
        model = self.build_model(vocab, ModelClassifier)

        train_loader, dev_loader = self.build_data_loaders(train_data, dev_data)
        train_loader.index_with(vocab)
        dev_loader.index_with(vocab)

        # You obviously won't want to create a temporary file for your training
        # results, but for execution in binder for this guide, we need to do this.
        with tempfile.TemporaryDirectory() as serialization_dir:
            trainer = self.build_trainer(model, serialization_dir, train_loader, dev_loader)
            trainer.train()

        return model, dataset_reader