# %%
import pathlib
import pm4py
import matplotlib
import matplotlib.pyplot as plt
from pm4py.util import constants
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.visualization.graphs import visualizer as graphs_visualizer
matplotlib.rcParams['figure.facecolor'] = 'w'

# %%
original_path = pathlib.Path('data/RequestForPayment.xes_')
log = pm4py.read_xes(original_path.as_posix())
print(log[0][0])  #prints the first event of the first trace of the given log
# %%
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.visualization.petrinet import visualizer as petrinet_visualization
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualization

dfg = dfg_discovery.apply(log)
gviz = dfg_visualization.apply(dfg, log=log, variant=dfg_visualization.Variants.FREQUENCY)
dfg_visualization.view(gviz)


# %%
original_df = pm4py.convert_to_dataframe(log)
original_df
# %%

# %%
data = original_df.copy()
data = data.iloc[:, 1:20]
not_required_cols = [
    'case:ProjectNumber',
    'case:TaskNumber',
    'case:ActivityNumber',
    'case:RequestedAmount_0',
    'case:DeclarationNumber_0',
]
data = data.drop(not_required_cols, axis=1)
data

# %%

# %%
list(original_df.columns)