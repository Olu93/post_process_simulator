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


class ListTokenizer(Tokenizer):
    def tokenize(self, text: List[str]) -> List[Token]:
        return [Token(t) for t in text]

    def batch_tokenize(self, texts: List[List[str]]) -> List[List[Token]]:
        return [self.tokenize(sent) for sent in texts]


class TaskModes(Enum):
    SIMPLE = auto()
    EXTENSIVE = auto()
    EXTENSIVE_RANDOM = auto()
    FINAL_OUTCOME = auto()


class DatasetModes(Enum):
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
    modes: TaskModes = TaskModes.SIMPLE
    padding_token: str = "<PAD>"

    def __init__(self,
                 data_path: str,
                 caseId: str = 'case:concept:name',
                 activityId: str = 'concept:name',
                 debug=False,
                 mode: TaskModes = TaskModes.SIMPLE,
                 tokenizer: Tokenizer = None,
                 token_indexers: TokenIndexer = None,
                 max_tokens: int = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or ListTokenizer()
        self.token_indexers = {"tokens": token_indexers or SingleIdTokenIndexer()}
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

        self._traces = {idx: self.tokenizer.tokenize(list(df[self.activityId].values)) for idx, df in grouped_traces}

        self.instantiate_dataset()

    def instantiate_dataset(self):
        if self.mode == TaskModes.SIMPLE:
            self.traces = self._traces.values()
            self.trace_data, self.test = self._train_test_split(self.traces)
            self.train, self.val = self.train_val_split(self.trace_data)

        if self.mode == TaskModes.EXTENSIVE:
            self.traces = (tr[0:end] for tr in self._traces.values() for end in range(2, len(tr) + 1) if len(tr) > 1)

        if self.mode == TaskModes.EXTENSIVE_RANDOM:
            tmp_traces = (tr[random.randint(0,
                                            len(tr) - 1):] for tr in self._traces.values()
                          for sample in self._heuristic_sample_size(tr) if len(tr) > 1)
            self.traces = (tr[:random.randint(2, len(tr))] for tr in tmp_traces if len(tr) > 1)

        if self.mode == TaskModes.FINAL_OUTCOME:
            self.traces = (tr[0:end] + tr[-1] for tr in self._traces.values() for end in range(2, len(tr))
                           if len(tr) > 1)

        # if self.mode == TaskModes.EXTENSIVE:
        #     self.traces = ((4 - to) * [Token(self.padding_token)] + tr[0:to] for tr in self._traces.values()
        #                    for to in range(2,
        #                                    len(tr) + 1) if len(tr) > 2)

        # if self.mode == TaskModes.FINAL_OUTCOME:
        #     self.traces = ((4 - to) * [Token(self.padding_token)] + tr[0:to] + tr[-1] for tr in self._traces.values()
        #                    for to in range(2, len(tr)) if len(tr) > 2)

        self.trace_data, self.test = self._train_test_split(self.traces)
        self.train, self.val = self.train_val_split(self.trace_data)

    def _heuristic_sample_size(self, sequence):
        return range((len(sequence)**2 + len(sequence)) // 4)

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

    def register_vocabulary(self):
        all_unique_tokens = list(self.data[self.activityId].unique()) + [self.padding_token]
        self.vocabulary.add_tokens_to_namespace(all_unique_tokens)
        self.vocabulary.add_tokens_to_namespace(all_unique_tokens, namespace="labels")
        self.max_tokens = self.vocabulary.get_vocab_size()

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

        if file_path == DatasetModes.TRAIN:
            for line in self.train:
                instance = AbstractProcessLogReader.text_to_instance(line, self.token_indexers, self.max_tokens)
                yield instance

        if file_path == DatasetModes.VAL:
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