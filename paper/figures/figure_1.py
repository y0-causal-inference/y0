"""Generate Figure 1 for the JOSS manuscript."""

import matplotlib.pyplot as plt

from y0.dsl import Variable
from y0.graph import NxMixedGraph

Cancer = Variable("Cancer")
Smoking = Variable("Smoking")
Tar = Variable("Tar")

layout = {
    Smoking: [-5, 0],
    Tar: [0, 0.5],
    Cancer: [5, 0],
}

cancer_graph_initial = NxMixedGraph.from_edges(
    directed=[(Smoking, Tar), (Tar, Cancer), (Smoking, Cancer)]
)
cancer_graph = NxMixedGraph.from_edges(
    directed=[(Smoking, Tar), (Tar, Cancer), (Smoking, Cancer)], undirected=[(Smoking, Tar)]
)

fig, (lax, rax) = plt.subplots(1, 2, figsize=(10, 3))

fontdict = {"fontsize": 20, "fontweight": "bold"}
lax.text(-6.55, 0.56, "A", fontdict=fontdict)
rax.text(-6.55, 0.56, "B", fontdict=fontdict)

cancer_graph_initial.draw(ax=lax, layout=layout, font_size=15, node_size=5000)
cancer_graph.draw(ax=rax, layout=layout, font_size=15, node_size=5000, radius=-0.7)

fig.tight_layout(pad=0.0)
fig.savefig("cancer_tar.pdf")
