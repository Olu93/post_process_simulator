from enum import Enum, auto
from IPython.display import display
import pathlib
from allennlp.data import Instance
from allennlp.data import Vocabulary, DatasetReader
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.data.fields import Field, LabelField, TextField
import random

import pm4py
from pm4py.util import constants
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.visualization.graphs import visualizer as graphs_visualizer
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.visualization.petrinet import visualizer as petrinet_visualization
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualization
import pandas as pd
from typing import Iterable, List, Dict


class Modes(Enum):
    SIMPLE = auto()
    EXTENSIVE = auto()
    FINAL_OUTCOME = auto()


class Dataset(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


@DatasetReader.register("process-log-xes")
class AbstractProcessLogReader(DatasetReader):
    log = None
    data_path: str = None
    _original_data: pd.DataFrame = None
    data: pd.DataFrame = None
    debug: bool = False
    caseId: str = None
    activityId: str = None
    _vocab: dict = None
    vocabulary: Vocabulary = Vocabulary.empty()
    modes: Modes = Modes.SIMPLE

    def __init__(self,
                 data_path: str,
                 caseId: str = 'case:concept:name',
                 activityId: str = 'concept:name',
                 debug=False,
                 mode: Modes = Modes.SIMPLE,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = None
        self.debug = debug
        self.mode = mode
        self.data_path = pathlib.Path(data_path)
        self.caseId = caseId
        self.activityId = activityId
        self.log = pm4py.read_xes(self.data_path.as_posix())
        if self.debug:
            print(self.log[1])  #prints the first event of the first trace of the given log
        if self.debug:
            print(self.log[1][0])  #prints the first event of the first trace of the given log
        self._original_data = pm4py.convert_to_dataframe(self.log)
        if self.debug:
            display(self._original_data.head())
        self.preprocess_level_general()
        self.preprocess_level_specialized()
        self.compute_sequences()
        self.register_vocabulary()

    def show_dfg(self):
        dfg = dfg_discovery.apply(self.log)
        gviz = dfg_visualization.apply(dfg, log=self.log, variant=dfg_visualization.Variants.FREQUENCY)
        dfg_visualization.view(gviz)

    @property
    def original_data(self):
        return self._original_data.copy()

    @original_data.setter
    def original_data(self, data: pd.DataFrame):
        self._original_data = data

    def preprocess_level_general(self, **kwargs):
        self.data = self.original_data
        remove_cols = kwargs.get('remove_cols')
        if remove_cols:
            self.data = self.original_data.drop(remove_cols, axis=1)

    def preprocess_level_specialized(self, **kwargs):
        self.data = self.data

    def compute_sequences(self):
        grouped_traces = list(self.data.groupby(by=self.caseId))

        self._traces = {
            idx: self.tokenizer.tokenize(" ".join(list(df[self.activityId].values)))
            for idx, df in grouped_traces
        }

        self.instantiate_dataset()

    def instantiate_dataset(self):
        if self.mode == Modes.SIMPLE:
            self.traces = self._traces.values()
            self.train, self.val, self.test = self._train_test_split(self.traces)

        if self.mode == Modes.EXTENSIVE:
            self.traces = ((4 - to) * [Token("<PAD>")] + tr[0:to] for tr in self._traces.values()
                           for to in range(2,
                                           len(tr) + 1) if len(tr) > 2)

        if self.mode == Modes.FINAL_OUTCOME:
            self.traces = ((4 - to) * [Token("<PAD>")] + tr[0:to] + tr[-1] for tr in self._traces.values()
                           for to in range(2, len(tr)) if len(tr) > 2)

        self.trace_data, self.test = self._train_test_split(self.traces)
        self.train, self.val = self.train_val_split(self.trace_data)

    def _train_test_split(self, traces):
        traces = list(traces)
        random.shuffle(traces)
        len_dataset = len(traces)
        len_train_traces = int(len_dataset * 0.9)
        train_traces = traces[:len_train_traces]
        # len_val_traces = int(len_dataset * 0.6)
        # val_traces = traces[:len_val_traces]
        len_test_traces = int(len_dataset * 0.1)
        test_traces = traces[:len_test_traces]
        return train_traces, test_traces

    def train_val_split(self, traces):
        traces = list(traces)
        random.shuffle(traces)
        len_dataset = len(traces)
        len_train_traces = int(len_dataset * 0.6)
        train_traces = traces[:len_train_traces]
        len_val_traces = int(len_dataset * 0.4)
        val_traces = traces[:len_val_traces]
        return train_traces, val_traces

    # Not necessary just add token
    def register_vocabulary(self):

        self.vocabulary.add_tokens_to_namespace(list(self.data[self.activityId].unique()))
        self.vocabulary.add_token_to_namespace("<PAD>")  
        self.max_tokens = self.vocabulary.get_vocab_size() # Entirely wrong even

    @property
    def tokens(self) -> List[str]:
        return self.vocabulary.list_available()

    @property
    def vocab2idx(self) -> List[str]:
        return self.vocabulary.get_token_to_index_vocabulary()

    @property
    def idx2vocab(self) -> List[str]:
        return self.vocabulary.get_index_to_token_vocabulary()

    # def text_to_instance(self, text: str, label: str = None) -> Instance:
    #     tokens = self.tokenizer.tokenize(text)
    #     if self.max_tokens:
    #         tokens = tokens[:self.max_tokens]
    #     text_field = TextField(tokens, self.token_indexers)
    #     fields: Dict[str, Field] = {"text": text_field}
    #     if label:
    #         fields["label"] = LabelField(label)
    #     return Instance(fields)

    # def _read(self, file_path: str) -> Iterable[Instance]:
    #     for line in self.traces:
    #         yield self.text_to_instance(line, 0)

    def _read(self, file_path: str) -> Iterable[Instance]:

        if file_path == Dataset.TRAIN:
            for line in self.train:
                instance = AbstractProcessLogReader.text_to_instance(line, self.token_indexers, self.max_tokens)
                yield instance

        if file_path == Dataset.VAL:
            for line in self.val:
                instance = AbstractProcessLogReader.text_to_instance(line, self.token_indexers, self.max_tokens)
                yield instance

    @staticmethod
    def text_to_instance(line, token_indexers, max_tokens=None):
        prev_events, next_event = line[:-1], str(line[-1])
        if max_tokens:
            prev_events = prev_events[len(prev_events) - max_tokens:]
        text_field = TextField(prev_events, token_indexers)
        label_field = LabelField(next_event)
        return Instance({"text": text_field, "label": label_field})