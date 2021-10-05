from readers.AbstractProcessLogReader import AbstractProcessLogReader


class PermitLogReader(AbstractProcessLogReader):
    def __init__(self, **kwargs) -> None:
        super().__init__('data/PermitLog.xes_', **kwargs)

    def preprocess_level_general(self):
        super().preprocess_level_general(remove_cols=[
            'case:ProjectNumber',
            'case:TaskNumber',
            'case:ActivityNumber',
            'case:RequestedAmount_0',
            'case:DeclarationNumber_0',
        ])