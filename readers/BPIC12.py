from readers.AbstractProcessLogReader import AbstractProcessLogReader
import random

class BPIC12W(AbstractProcessLogReader):
    def __init__(self, **kwargs) -> None:
        super().__init__('data/financial_log.xes', **kwargs)

    def preprocess_level_general(self):
        super().preprocess_level_general(remove_cols=[])

    def preprocess_level_specialized(self, **kwargs):
        self.data = self.data[self.data[self.activityId].str.startswith("W_", na=False)]

    def _train_val_test_split(self, traces):
        traces = list(traces)
        random.shuffle(traces)
        len_dataset = len(traces)
        len_train_traces = int(len_dataset * 0.3)
        train_traces = traces[:len_train_traces]
        len_val_traces = int(len_dataset * 0.6)
        val_traces = traces[:len_val_traces]
        len_test_traces = int(len_dataset * 0.1)
        test_traces = traces[:len_test_traces]
        return train_traces, val_traces, test_traces