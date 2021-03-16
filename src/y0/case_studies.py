# -*- coding: utf-8 -*-

"""Graphs used in case studies.

.. seealso:: :mod:`y0.examples`.
"""

import networkx as nx

__all__ = [
    'igf_graph',
]

#: The IGF directed graph example from Sara
igf_graph = nx.DiGraph([
    ('EGF', 'SOS'),
    ('EGF', 'PI3K'),
    ('IGF', 'SOS'),
    ('IGF', 'PI3K'),
    ('SOS', 'Ras'),
    ('Ras', 'PI3K'),
    ('Ras', 'Raf'),
    ('PI3K', 'Akt'),
    ('Akt', 'Raf'),
    ('Raf', 'Mek'),
    ('Mek', 'Erk'),
])
