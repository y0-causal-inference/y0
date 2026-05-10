################
 Identification
################

.. automodule:: y0.algorithm.identify
    :members:

.. automodule:: y0.algorithm.do_calculus
    :members:

Generation-Facing Notes
=======================

The Dafny ID specification now isolates Line 4 component traversal behind a
dedicated wrapper so code generation can depend on a single abstraction point.

1. ``SetOfSetsToSeq`` remains available for proof compatibility.
2. ``Line4ComponentsSeq`` is the generation-facing entrypoint used by
    ``Identification.IDImpl``.
3. ``CComponentsWithout_Partition`` in ``semi_markovian.dfy`` collects the
    non-empty/subset obligations needed by Line 4 preconditions.

This keeps the current verified control flow stable while making it easier to
swap in a stronger deterministic ordering contract for generated artifacts.
