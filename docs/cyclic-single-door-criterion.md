# Efficient Adjustment Set Computation for the Single Door Criterion in Cyclic Causal Graphs

To efficiently compute an adjustment set for the single door criterion in a
cyclic causal graph, we must overcome the fact that standard d-separation fails
in the presence of feedback loops \cite{bongers2021foundations}. In cyclic
graphs, variables mutually determine one another, allowing spurious associations
to flow through feedback loops even when conditioned on classical d-separators
\cite{pmlr-v115-forre20a}.

To handle this, the algorithm must use $\sigma$-separation, which generalizes
d-separation for cyclic structural causal models \cite{pmlr-v115-forre20a}. To
ensure the algorithm remains highly efficient and avoids the $O(n^2 m)$
complexity of moralizing graphs \cite{pmlr-v115-van-der-zander20a}, we can
utilize a structural trick called a $\sigma$-extension
\cite{rantanen2020learning} combined with linear-time search primitives
\cite{pmlr-v115-van-der-zander20a}.

Here is the step-by-step algorithm to efficiently evaluate the single door
criterion for every edge:

### Phase 1: Global Preprocessing (The $\sigma$-Extension)

Instead of running a complex $\sigma$-separation check from scratch for every
edge, we can transform the cyclic graph into an extended graph where standard,
fast d-separation algorithms work perfectly \cite{rantanen2020learning}.

1. **Identify Strongly Connected Components (SCCs):** Find all SCCs in the
   cyclic causal graph $G$ \cite{bongers2021foundations}. A node $V_i$ and $V_j$
   are in the same SCC if there is a directed path from $V_i$ to $V_j$ and vice
   versa \cite{rantanen2020learning}.
2. **Build the $\sigma$-Extension Graph ($G^\sigma$):** Copy all nodes and edges
   from $G$ \cite{rantanen2020learning}. For every directed edge $V_i \to V_j$
   where $V_i$ and $V_j$ belong to the same SCC, add a bidirected edge
   $V_i \leftrightarrow V_j$ \cite{rantanen2020learning}.
   - _Why this works:_ Two nodes are $\sigma$-separated in $G$ if and only if
     they are d-separated in the $\sigma$-extension $G^\sigma$
     \cite{rantanen2020learning}.

### Phase 2: Edge-by-Edge Evaluation

For each directed edge $e = X \to Y$ in $G^\sigma$, perform the following
operations:

**1. Remove the Target Edge** Remove the specific edge $X \to Y$ from $G^\sigma$
to create a temporary test graph, $G_{-e}^\sigma$.

**2. Isolate Backdoor Paths** To strictly target backdoor paths (paths starting
with an arrow into $X$), construct the **proper back-door graph** $G^{pbd}_{XY}$
from $G_{-e}^\sigma$ \cite{pmlr-v115-van-der-zander20a}.

- Identify all nodes on proper possible causal paths from $X$ to $Y$ in
  $G_{-e}^\sigma$ \cite{pmlr-v115-van-der-zander20a}.
- Delete all directed edges out of $X$ that lie on these directed paths to
  mathematically isolate the non-causal (backdoor) pathways
  \cite{henckel2022graphical}.

**3. Compute the Adjustment Set Efficiently** Now, find a set $Z$ that
d-separates $X$ and $Y$ in the proper backdoor graph $G^{pbd}_{XY}$
\cite{pmlr-v115-van-der-zander20a}. You have two highly efficient options
depending on your goal:

- **Option A: The Globally Optimal Set ($O$-set)** If you want the adjustment
  set that yields the lowest asymptotic estimation variance (maximum statistical
  efficiency), compute the $O$-set \cite{rotnitzky2020efficient}.
  - Find the causal nodes $cn(X \to Y)$, which are all nodes on directed paths
    from $X$ to $Y$ \cite{henckel2022graphical}.
  - Find the forbidden nodes $forb(X \to Y)$, defined as the descendants of the
    causal nodes, plus $X$ itself \cite{henckel2022graphical}.
  - The optimal adjustment set is
    **$Z = pa(cn(X \to Y)) \setminus forb(X \to Y)$**, meaning you take all
    parents of the causal nodes and remove any forbidden nodes
    \cite{rotnitzky2020efficient}.

- **Option B: A Minimal Cardinality Set** If you simply want the smallest valid
  set of variables to measure, utilize the `FINDNEARESTSEP` algorithm or similar
  linear-time primitives \cite{pmlr-v115-van-der-zander20a}.
  - Restrict a standard graph traversal to only visit the ancestors of $X$ and
    $Y$ \cite{pmlr-v115-van-der-zander20a}.
  - Trace backward from $X$ and $Y$ using active path rules to find the
    "nearest" nodes that block all flow \cite{pmlr-v115-van-der-zander20a}.
    Because this directly traverses the graph without building intermediate
    moral graphs, it finds a minimal adjustment set in linear time
    \cite{pmlr-v115-van-der-zander20a}.

### Algorithm Complexity

By applying the $\sigma$-extension upfront, you avoid recalculating loop
dynamics \cite{pmlr-v115-van-der-zander20a}. For a graph with $V$ vertices and
$E$ edges, Phase 1 takes $O(V+E)$. Phase 2 runs $E$ times, and the proper
backdoor graph construction plus the adjustment set calculation ($O$-set or
Minimal) takes $O(V+E)$ per edge \cite{pmlr-v115-van-der-zander20a}. The total
time complexity for evaluating the single door criterion across the entire
cyclic graph is bounded by $O(E \times (V+E))$, which is highly scalable for
large causal networks \cite{pmlr-v115-van-der-zander20a}.

---

### References

```bibtex
@article{bongers2021foundations,
  title={Foundations of structural causal models with cycles and latent variables},
  author={Bongers, Stephan and Forr{\'e}, Patrick and Peters, Jonas and Mooij, Joris M},
  journal={The Annals of Statistics},
  volume={49},
  number={5},
  pages={2885--2915},
  year={2021},
  publisher={Institute of Mathematical Statistics}
}

@inproceedings{pmlr-v115-forre20a,
  title = {Causal Calculus in the Presence of Cycles, Latent Confounders and Selection Bias},
  author = {Forr{\'{e}}, Patrick and Mooij, Joris M.},
  booktitle = {Proceedings of The 35th Uncertainty in Artificial Intelligence Conference},
  pages = {71--80},
  year = {2020},
  volume = {115},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR}
}

@article{henckel2022graphical,
  title={Graphical criteria for efficient total effect estimation via adjustment in causal linear models},
  author={Henckel, Leonard and Perkovi{\'c}, Emilija and Maathuis, Marloes H},
  journal={Journal of the Royal Statistical Society: Series B (Statistical Methodology)},
  volume={84},
  number={2},
  pages={579--599},
  year={2022}
}

@inproceedings{rantanen2020learning,
  title={Learning Optimal Cyclic Causal Graphs from Interventional Data},
  author={Rantanen, Kari and Hyttinen, Antti and J{\"a}rvisalo, Matti},
  booktitle={Probabilistic Graphical Models},
  year={2020}
}

@article{rotnitzky2020efficient,
  title={Efficient Adjustment Sets for Population Average Causal Treatment Effect Estimation in Graphical Models},
  author={Rotnitzky, Andrea and Smucler, Ezequiel},
  journal={Journal of Machine Learning Research},
  volume={21},
  number={164},
  pages={1--86},
  year={2020}
}

@inproceedings{pmlr-v115-van-der-zander20a,
  title = {Finding Minimal d-separators in Linear Time and Applications},
  author = {van der Zander, Benito and Li\'{s}kiewicz, Maciej},
  booktitle = {Proceedings of The 35th Uncertainty in Artificial Intelligence Conference},
  pages = {637--647},
  year = {2020},
  volume = {115},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR}
}
```
