Identification of Joint Interventional Distributions in Recursive Semi-Markovian
Causal Models

**Ilya Shpitser, Judea Pearl** Cognitive Systems Laboratory Department of
Computer Science University of California, Los Angeles Los Angeles, CA. 90095
{ilyas, judea}@cs.ucla.edu

_This work was supported in part by AFOSR grant #F49620-01-1-0055, NSF grant
#IIS-0535223, and MURI grant #N00014-00-1-0617._ Copyright 2006, American
Association for Artificial Intelligence (www.aaai.org). All rights reserved.

Abstract

This paper is concerned with estimating the effects of actions from causal
assumptions, represented concisely as a directed graph, and statistical
knowledge, given as a probability distribution. We provide a necessary and
sufficient graphical condition for the cases when the causal effect of an
arbitrary set of variables on another arbitrary set can be determined uniquely
from the available information, as well as an algorithm which computes the
effect whenever this condition holds. Furthermore, we use our results to prove
completeness of do-calculus [Pearl, 1995], and a version of an identification
algorithm in [Tian, 2002] for the same identification problem. Finally, we
derive a complete characterization of semi-Markovian models in which all causal
effects are identifiable.

Introduction

This paper deals with computing effects of actions in domains specified as
causal diagrams, or graphs with directed and bidirected edges. Vertices in such
graphs correspond to variables of interest, directed edges correspond to
potential direct causal relationships between variables, and bidirected edges
correspond to 'hidden common causes,' or spurious dependencies between variables
[Pearl, 1995], [Pearl, 2000]. Aside from causal knowledge encoded by these
graphs, we also have statistical knowledge in the form of a joint probability
distribution over observable variables, which we will denote by $P$. An action
on a variable set $\mathbf{X}$ in a causal domain consists of forcing
$\mathbf{X}$ to particular values $\mathbf{x}$, regardless of the values
$\mathbf{X}$ would have taken prior to the intervention. This action, denoted
$do(\mathbf{x})$ in [Pearl, 2000], changes the original joint distribution $P$
over observables into a new interventional distribution denoted
$P_{\mathbf{x}}$. The marginal distribution $P_{\mathbf{x}}(\mathbf{Y})$ of a
variable set $\mathbf{Y}$ obtained from $P_{\mathbf{x}}$ will be our notion of
effect of action $do(\mathbf{x})$ on $\mathbf{Y}$.

Our task is to characterize cases when $P_{\mathbf{x}}(\mathbf{Y})$ can be
determined uniquely from $P$, or identified in a given graph $G$. It is well
known that in Markovian models, those causal domains whose graphs do not contain
bidirected edges, all effects are identifiable [Pearl, 2000]. If our model
contains 'hidden common causes', that is if the model is semi-Markovian, the
situation is less clear.

Consider the causal diagrams in Fig. 1 (a) and (b) which might represent a
situation in diagnostic medicine. For instance, nodes $W_{1}$, $W_{2}$ are
afflictions of a pregnant mother and her unborn child, respectively. $X$ is a
toxin produced in the mother's body as a result of the illness, which could be
artificially lowered by a treatment. $Y_{1}$, $Y_{2}$ stand for the survival of
mother and child. Bidirected arcs represent confounding factors for this
situation not explicitly named in the model, but affecting the outcome. We are
interested in computing the effect of lowering $X$ on $Y_{1}$, $Y_{2}$ without
actually performing the potentially dangerous treatment. In our framework this
corresponds to computing $P_{x}(Y_{1},Y_{2})$ from
$P(X,W_{1},W_{2},Y_{1},Y_{2})$. The subtlety of this problem can be illustrated
by noting that in Fig. 1 (a), the effect is identifiable, while in Fig. 1 (b),
it is not.

Multiple sufficient conditions for identifiability in the semi-Markovian case
are known [Spirtes, Glymour, & Scheines, 1993], [Pearl & Robins, 1995], [Pearl,
1995], [Kuroki & Miyakawa, 1999]. A summary of these results can be found in
[Pearl, 2000]. Most work in this area has generally taken advantage of the fact
that certain properties of the causal diagram reflect properties of $P$, and is
phrased in the language of graph theory. For example, the back-door criterion
[Pearl, 2000], states that if there exists a set $\mathbf{Z}$ of non-descendants
of $X$ that 'blocks' certain paths in the graph from $X$ to $\mathbf{Y}$, then
$P_{x}(\mathbf{Y})=\sum_{\mathbf{z}}P(\mathbf{Y}|\mathbf{z},x)P(\mathbf{z})$.

Results in [Pearl, 1995], [Halpern, 2000] take a different approach, and provide
sound rules which are used to manipulate the expression corresponding to the
effect algebraically. These rules are then applied until the resulting
expression can be computed from $P$. Though the axioms in [Halpern, 2000] were
shown to be complete, the practical applicability of the result to
identifiability is limited, since it does not provide a closed form criterion
for the cases when effects are not identifiable, nor a closed form algorithm for
expressing effects in terms of $P$ when they are identifiable. Instead, one must
rely on finding a good proof strategy and hope the effect expression is reduced
to something derivable from $P$.

Recently, a number of necessity results for identifiability have been proven.
One such result [Tian & Pearl, 2002] states that $P_{x}$ is identifiable if and
only if there is no path consisting entirely of bidirected arcs from $X$ to a
child of $X$. The authors have also been made aware of a paper currently in
review [Huang & Valtorta, 2006] which shows a modified version of an algorithm
found in [Tian, 2002] is complete for identifying $P_{\mathbf{x}}(\mathbf{y})$,
where $\mathbf{X}, \mathbf{Y}$ are sets. One of the contributions of this paper
is a simpler proof of the same result, using non-positive distributions. The
results in this paper were independently derived.

In this paper, we offer a complete solution to the problem of identifying
$P_{\mathbf{x}}(\mathbf{y})$ in semi-Markovian models. Using a graphical
structure called a hedge, we construct a sound and complete algorithm for
identifying $P_{\mathbf{x}}(\mathbf{y})$ from $P$. The algorithm returns either
an expression derivable from $P$ or a hedge which witnesses the
non-identifiability of the effect. We also show that steps of our algorithm
correspond to sequences of applications of rules of do-calculus [Pearl, 1995],
thus proving completeness of do-calculus for the same identification problem.
Furthermore, we show a version of Tian's algorithm [Tian, 2002] is also complete
and thus equivalent to ours. Finally, we derive a complete characterization of
models in which all effects are identifiable.

Notation and Definitions

In this section we reproduce the technical definitions needed for the rest of
the paper, and introduce common non-identifying graph structures. We will denote
variables by capital letters, and their values by small letters. Similarly, sets
of variables will be denoted by bold capital letters, and sets of values by bold
small letters. We will use some graph-theoretic abbreviations:
$Pa(\mathbf{Y})_{G}$, $An(\mathbf{Y})_{G}$, and $De(\mathbf{Y})_{G}$ will denote
the set of (observable) parents, ancestors, and descendants of the node set
$\mathbf{Y}$ in $G$, respectively. The lowercase versions of the above kinship
sets will denote corresponding sets of values. We will omit the graph subscript
if the graph in question is assumed or obvious. We will denote the set
$\{X \in G | [cite_start]De(X)_{G}=\emptyset\}$ as the root set of $G$. For a
given node $V$ in a graph $G$ and a topological ordering $\pi$ of nodes in $G$,
we denote $V_{\pi}^{(i-1)}$ to be the set of observable nodes preceding $V_{i}$
in $\pi$. A topological ordering of $G$ is a total order where no node can be
greater than its descendant in $G$.

A probabilistic causal model is a tuple
$M=\langle\mathbf{U},\mathbf{V},\mathbf{F},P(\mathbf{U})\rangle$, where
$\mathbf{V}$ is a set of observable variables, $\mathbf{U}$ is a set of
unobservable variables distributed according to $P(\mathbf{U})$, and
$\mathbf{F}$ is a set of functions. Each variable $V \in \mathbf{V}$ has a
corresponding function $f_{V} \in \mathbf{F}$ that determines the value of $V$
in terms of other variables in $\mathbf{V}$ and $\mathbf{U}$.

The induced graph $G$ of a causal model $M$ contains a node for every element in
$\mathbf{V}$, a directed edge between nodes $X$ and $Y$ if $f_{Y}$ possibly uses
the values of $X$ directly to determine the value of $Y$, and a bidirected edge
between nodes $X$ and $Y$ if $f_{X}$ and $f_{Y}$ both possibly use the value of
some variable in $\mathbf{U}$ to determine their respective values. In this
paper we consider recursive causal models, those models which induce acyclic
graphs.

For the purposes of this paper, we assume all variable domains are finite, and
$P(\mathbf{U})=\prod_{i}P(U_{i})$. The distribution on $\mathbf{V}$ induced by
$P(\mathbf{U})$ and $\mathbf{F}$ will be denoted $P(\mathbf{V})$. Sometimes it
is assumed $P(\mathbf{V})$ is a positive distribution. In this paper we do not
make this assumption. Thus, we must make sure that for every distribution
$P(\mathbf{W}|\mathbf{Z})$ that we consider, $P(\mathbf{Z})$ must be positive.
This can be achieved by making sure to sum over events with positive probability
only. Furthermore, for any action $do(\mathbf{x})$ that we consider, it must be
the case that $P(\mathbf{x}|Pa(\mathbf{X})_{G}\backslash \mathbf{X})>0$
otherwise the distribution $P_{\mathbf{x}}(\mathbf{V})$ is not well defined
[Pearl, 2000].

In any causal model there is a relationship between its induced graph $G$ and
$P$, where
$P(v_{1},...,v_{n},u_{1},...,u_{k}) = \prod_{i}P(v_{i}|pa^{*}(V_{i})_{G})\prod_{j}P(u_{j})$,
and $Pa^{*}(.)_{G}$ also includes unobservable parents [Pearl, 2000]. Whenever
this relationship holds, we say that $G$ is an I-map (independence map) of $P$.
The I-map relationship allows us to link independence properties of $P$ to $G$
by using the following well known notion of path separation [Pearl, 1988].

**Definition 1 (d-separation)** A path $p$ in $G$ is said to be d-separated by a
set $\mathbf{Z}$ if and only if either:

1.  $p$ contains a chain $I \rightarrow M \rightarrow J$ or fork
    $I \leftarrow M \rightarrow J$ such that $M \in \mathbf{Z}$ or

2.  $p$ contains an inverted fork $I \rightarrow M \leftarrow J$ such that
    $De(M)_{G} \cap \mathbf{Z} = \emptyset$.

Two sets $\mathbf{X}, \mathbf{Y}$ are said to be d-separated given $\mathbf{Z}$
in $G$ if all paths from $\mathbf{X}$ to $\mathbf{Y}$ in $G$ are d-separated by
$\mathbf{Z}$. The following well known theorem links d-separation of vertex sets
in an I-map $G$ with the independence of corresponding variable sets in $P$.

**Theorem 1** If sets $\mathbf{X}$ and $\mathbf{Y}$ are d-separated by
$\mathbf{Z}$ in $G$, then $\mathbf{X}$ is independent of $\mathbf{Y}$ given
$\mathbf{Z}$ in every $P$ for which $G$ is an I-map.

We will abbreviate this statement of d-separation, and corresponding
independence by $(\mathbf{X} \perp\perp \mathbf{Y}|\mathbf{Z})_{G}$, following
the notation in [Dawid, 1979]. A path that is not d-separated is said to be
d-connected. A path starting from a node $X$ with an arrow pointing to $X$ is
called a back door path from $X$. A path consisting entirely of bidirected arcs
is called a bidirected path.

In the framework of causal models, actions are modifications of functional
relationships. Each action $do(\mathbf{x})$ on a causal model $M$ produces a new
model
$M_{\mathbf{x}} = (\mathbf{U}, \mathbf{V}, \mathbf{F}_{\mathbf{x}}, P(\mathbf{U}))$,
where $\mathbf{F}_{\mathbf{x}}$ is obtained by replacing $f_{X} \in \mathbf{F}$
for every $X \in \mathbf{X}$ with a new function that outputs a constant value
$\mathbf{x}$ given by $do(\mathbf{x})$. Since subscripts are used to denote
submodels, we will use numeric superscripts to enumerate models (e.g. $M^{1}$).
For a model $M^{i}$, we will often denote its associated probability
distributions as $P^{i}$ rather than $P$.

**Definition 2 (Causal Effect Identifiability)** The causal effect of an action
$do(\mathbf{x})$ on a set of variables $\mathbf{Y}$ such that
$\mathbf{Y} \cap \mathbf{X} = \emptyset$ is said to be identifiable from $P$ in
$G$ if $P_{\mathbf{x}}(\mathbf{Y})$ is (uniquely) computable from
$P(\mathbf{V})$ in any causal model which induces $G$.

**Lemma 1** Let $\mathbf{X}, \mathbf{Y}$ be two sets of variables. Assume there
exist two causal models $M^{1}$ and $M^{2}$ with the same induced graph $G$ such
that $P^{1}(\mathbf{V})=P^{2}(\mathbf{V})$,
$P^{1}(\mathbf{x}|Pa(\mathbf{X})_{G}\backslash \mathbf{X})>0$, and
$P_{\mathbf{x}}^{1}(\mathbf{Y}) \ne P_{\mathbf{x}}^{2}(\mathbf{Y})$. Then
$P_{\mathbf{x}}(\mathbf{y})$ is not identifiable in $G$. _Proof:_ No function
from $P$ to $P_{\mathbf{x}}(\mathbf{y})$ can exist by assumption, let alone a
computable function. $\Box$

The simplest example of a non-identifiable graph structure is the so called 'bow
arc' graph, see Fig. 2 (a).

**Theorem 2** $P_{x}(Y)$ is not identifiable in the bow arc graph. _Proof:_ We
construct two causal models $M^{1}$ and $M^{2}$ such that
$P^{1}(X,Y)=P^{2}(X,Y)$ and $P_{x}^{1}(Y) \ne P_{x}^{2}(Y)$. The two models
agree on the following: all 3 variables are boolean, $U$ is a fair coin, and
$f_{X}(u)=u$. Let $\oplus$ denote the exclusive or (XOR) function. Then the
value of $Y$ is determined by the function $u \oplus x$ in $M^{1}$, while $Y$ is
set to 0 in $M^{2}$. Then $P^{1}(Y=0)=P^{2}(Y=0)=1$,
$P^{1}(X=0)=P^{2}(X=0)=0.5$. Therefore, $P^{1}(X,Y)=P^{2}(X,Y)$, while
$P_{x}^{2}(Y=0)=1 \ne P_{x}^{1}(Y=0)=0.5$. Note that while $P$ is non-positive,
it is straightforward to modify the proof for the positive case by letting
$f_{Y}$ functions in both models return 1 half the time, and the values outlined
above half the time. $\Box$

A number of other specific graphs have been shown to contain unidentifiable
effects. For instance, in all graphs in Fig. 2, taken from [Pearl, 2000],
$P_{x}(Y)$ is not identifiable.

Throughout the paper, we will make use of the 3 rules of do-calculus [Pearl,
1995].

- **Rule 1:**
  $P_{\mathbf{x}}(\mathbf{y}|\mathbf{z},\mathbf{w})=P_{\mathbf{x}}(\mathbf{y}|\mathbf{w})$
  if
  $(\mathbf{Y} \perp\perp \mathbf{Z}|\mathbf{X},\mathbf{W})_{G_{\overline{\mathbf{X}}}}$

- **Rule 2:**
  $P_{\mathbf{x},\mathbf{z}}(\mathbf{y}|\mathbf{w}) = P_{\mathbf{x}}(\mathbf{y}|\mathbf{z}, \mathbf{w})$
  if
  $(\mathbf{Y} \perp\perp \mathbf{Z}|\mathbf{X}, \mathbf{W})_{G_{\overline{\mathbf{X}}\underline{\mathbf{Z}}}}$

- **Rule 3:**
  $P_{\mathbf{x},\mathbf{z}}(\mathbf{y}|\mathbf{w})=P_{\mathbf{x}}(\mathbf{y}|\mathbf{w})$
  if
  $(\mathbf{Y} \perp\perp \mathbf{Z}|\mathbf{X},\mathbf{W})_{G_{\overline{\mathbf{X}},\overline{\mathbf{Z}(\mathbf{W})}}}$
  where
  $\mathbf{Z}(\mathbf{W})=\mathbf{Z}\backslash An(\mathbf{W})_{G_{\overline{\mathbf{X}}}}$

These rules allow insertion and deletion of interventions and observational
evidence into and from distributions, using probabilistic independencies implied
by the causal graph due to Theorem 1. Here
$G_{\overline{\mathbf{X}}\underline{\mathbf{Z}}}$ is taken to mean the graph
obtained from $G$ by removing arrows pointing to $\mathbf{X}$ and arrows leaving
$\mathbf{Z}$.

C-Trees and Direct Effects

Sets of nodes interconnected by bidirected paths turned out to be an important
notion for identifiability and have been studied at length in [Tian, 2002] under
the name of C-components.

**Definition 3 (C-component)** Let $G$ be a semi-Markovian graph such that a
subset of its bidirected arcs forms a spanning tree over all vertices in $G$.
Then $G$ is a C-component (confounded component).

If $G$ is not a C-component, it can be uniquely partitioned into a set $C(G)$ of
subgraphs, each a maximal C-component. An important result states that for any
set $\mathbf{C}$ which is a C-component, in a causal model $M$ with graph $G$,
$P_{\mathbf{v}\backslash \mathbf{c}}(\mathbf{C})$ is identifiable [Tian, 2002].
The quantity $P_{\mathbf{v}\backslash \mathbf{c}}(\mathbf{C})$ will also be
denoted as $Q[\mathbf{C}]$. For the purposes of this paper, C-components are
important because a distribution $P$ in a semi-Markovian graph $G$ factorizes
such that each product term corresponds to a C-component. For instance, the
graphs shown in Fig. 2 (b) and (c), both have 2 C-components: $\{X, Z\}$ and
$\{Y\}$. Thus, the corresponding distribution factorizes as
$P(x,z,y)=Q[\{x,z\}]Q[\{y\}]=P_{y}(x,z)P_{x,z}(y)$. It is this factorization
which will ultimately allow us to decompose the identification problem into
smaller subproblems, and thus construct an identification algorithm.

We now consider a special kind of C-component which generalizes the
unidentifiable bow arc graph from the previous section.

**Definition 4 (C-tree)** Let $G$ be a semi-Markovian graph such that $G$ is a
C-component, all observable nodes have at most one child, and there is a node
$Y$ such that $An(Y)_{G} = G$. Then $G$ is a $Y$-rooted C-tree (confounded
tree).

The graphs in Fig. 2 (a) (d) (e) (f) and (h) are $Y$-rooted C-trees. There is a
relationship between C-trees and interventional distributions of the form
$P_{pa(Y)}(Y)$. Such distributions are known as direct effects, and correspond
to the influence of a variable $X$ on its child $Y$ along some edge, where the
variables $Pa(Y)\backslash \{X\}$ are fixed to some reference values. Direct
effects are of great importance in the legal domain, where one is often
concerned with whether a given party was directly responsible for damages, as
well as medicine, where elucidating the direct effect of medication, or disease
on the human body in a given context is crucial. See [Pearl, 2000], [Pearl,
2001] for a more complete discussion of direct effects. The absence of
$Y$-rooted C-trees in $G$ means the direct effect on $Y$ is identifiable.

**Lemma 2** Let $M$ be a causal model with graph $G$. Then for any node $Y$, the
direct effect $P_{pa(Y)}(Y)$ is identifiable if there is no subgraph of $G$
which forms a $Y$-rooted C-tree. _Proof:_ From [Tian, 2002], we know that
whenever there is no subgraph $G'$ of $G$, such that all nodes in $G'$ are
ancestors of $Y$, and $G'$ is a C-component, $P_{pa(Y)}(Y)$ is identifiable.
This entails the lemma. $\Box$

Theorem 2 suggests that C-trees are troublesome structures for the purposes of
identification of direct effects. In fact, our investigation revealed that
$Y$-rooted C-trees are troublesome for any effect on $Y$, as the following
theorem shows.

**Theorem 3** Let $G$ be a $Y$-rooted C-tree. Then the effect of any set of
nodes $\mathbf{X}$ in $G$ on $Y$ is not identifiable if $Y \notin \mathbf{X}$.
_Proof:_ We generalize the proof for the bow arc graph. We construct two models
with binary nodes. In the first model, the value of all observable nodes is set
to the bit parity (sum modulo 2) of the parent values. In the second model, the
same is true for all nodes except $Y$, with the latter being set to 0
explicitly. All $U$ nodes in both models are fair coins. Since $G$ is a tree,
and since every $U \in \mathbf{U}$ has exactly two children in $G$, every
$U \in \mathbf{U}$ has exactly two distinct downward paths to $Y$ in $G$. It's
then easy to establish that $Y$ counts the bit parity of every node in
$\mathbf{U}$ twice in the first model. But this implies $P^{1}(Y=1)=0$. Because
bidirected arcs form a spanning tree over observable nodes in $G$, for any set
of nodes $\mathbf{X}$ such that $Y \notin \mathbf{X}$ there exists
$U \in \mathbf{U}$ with one child in $An(\mathbf{X})_{G}$ and one child in
$G\backslash An(\mathbf{X})_{G}$. Thus $P_{\mathbf{x}}^{1}(Y=1)>0$ but
$P_{\mathbf{x}}^{2}(Y=1)=0$. It is straightforward to generalize this proof for
the positive $P(\mathbf{V})$ in the same way as in Theorem 2. $\Box$

While this theorem closes the identification problem for direct effects, the
problem of identifying general effects on a single variable $Y$ is more subtle,
as the following corollary shows.

**Corollary 1** Let $G$ be a semi-Markovian graph, let $\mathbf{X}$ and $Y$ be
disjoint sets of variables. If there exists $W \in An(Y)_{G_{\mathbf{x}}}$ such
that there exists a $W$-rooted C-tree which contains any variables in
$\mathbf{X}$, then $P_{\mathbf{x}}(Y)$ is not identifiable. _Proof:_ Fix a
$W$-rooted C-tree $T$, and a path $p$ from $W$ to $Y \in \mathbf{Y}$, where
$W \in An(Y)_{G_{\mathbf{x}}}$. Consider the graph $p \cup T$. Note that in this
graph $P_{\mathbf{x}}(Y)=\sum_{w}P_{\mathbf{x}}(w)P(Y|w)$. It is now easy to
construct $P(Y|W)$ in such a way that the mapping from $P_{\mathbf{x}}(W)$ to
$P_{\mathbf{x}}(Y)$ is one to one, while making sure $P$ is positive. $\Box$

This corollary implies that the effect of $do(\mathbf{x})$ on a given singleton
$Y$ can be non-identifiable even if $Y$ is nowhere near a C-tree, as long as the
effect of $do(\mathbf{x})$ on a set of ancestors of $Y$ is non-identifiable.
Therefore identifying effects on a single variable is not really any easier than
the general problem of identifying effects on multiple variables. We consider
this general problem in the next section.

Finally, we note that the last two results relied on existence of a C-tree
without giving an explicit algorithm for constructing one. In the remainder of
the paper we will give an algorithm which, among other things, will construct
the necessary C-tree, if it exists.

C-Forests, Hedges, and Non-Identifiability

The previous section established a powerful necessary condition for the
identification of effects on a single variable. It is the natural next step to
ask whether a similar condition exists for effects on multiple variables. We
start by considering a multi-root generalization of a C-tree.

**Definition 5 (C-forest)** Let $G$ be a semi-Markovian graph, where
$\mathbf{Y}$ is the root set. Then $G$ is a $\mathbf{Y}$-rooted C-forest
(confounded forest) if $G$ is a C-component, and all observable nodes have at
most one child.

We will show that just as there is a close relationship between C-trees and
direct effects, there is a close connection between C-forests and general
effects of the form $P_{\mathbf{x}}(\mathbf{Y})$, where $\mathbf{X}$ and
$\mathbf{Y}$ are sets of variables. To explicate this connection, we introduce a
special structure formed by a pair of C-forests that will feature prominently in
the remainder of the paper.

**Definition 6 (hedge)** Let $\mathbf{X}, \mathbf{Y}$ be disjoint sets of
variables in $G$. Let $F, F'$ be $\mathbf{R}$-rooted C-forests such that
$F \cap \mathbf{X} \ne \emptyset$, $F' \cap \mathbf{X} = \emptyset$,
$F' \subseteq F$, and
$\mathbf{R} \subset An(\mathbf{Y})_{G_{\overline{\mathbf{X}}}}$. Then $F$ and
$F'$ form a hedge for $P_{\mathbf{x}}(\mathbf{y})$ in $G$.

The mental picture for a hedge is as follows. We start with a C-forest $F'$.
Then, $F'$ 'grows' new branches, while retaining the same root set, and becomes
$F$. Finally, we 'trim the hedge', by performing the action $do(\mathbf{x})$
which has the effect of removing some incoming arrows in $F\backslash F'$, the
'newly grown' portion of the hedge. It's easy to check that every graph in Fig.
2 contains a pair of C-forests that form a hedge for $P_{x}(Y)$. The graph in
Fig. 1 (a) does not contain C-forests forming a hedge for $P_{x}(Y_{1},Y_{2})$,
while the graph in Fig. 1 (b) does: if $e$ is the edge between $W_{1}$ and $X$,
then $F=G\backslash \{e\}$, and $F'=F\backslash \{X\}$. Note that for the
special case of C-trees, $F$ is the C-tree itself, and $F'$ is the singleton
root $Y$. This last observation suggests the next result as a generalization of
Theorem 3.

**Theorem 4** Assume there exist $\mathbf{R}$-rooted C-forests $F, F'$ that form
a hedge for $P_{\mathbf{x}}(\mathbf{y})$ in $G$. Then
$P_{\mathbf{x}}(\mathbf{y})$ is not identifiable in $G$. _Proof:_ We first show
$P_{\mathbf{x}}(\mathbf{r})$ is not identifiable in $F$. As before, we construct
two models with binary nodes. In $M^{1}$ every variable in $F$ is equal to the
bit parity of its parents. In $M^{2}$ the same is true, except all nodes in $F'$
disregard the parent values in $F\backslash F'$. All $\mathbf{U}$ are fair coins
in both models. As was the case with C-trees, for any C-forest $F$, every
$U \in \mathbf{U} \cap F$ has exactly two downward paths to $\mathbf{R}$. It is
now easy to establish that in $M^{1}$ $\mathbf{R}$ counts the bit parity of
every node in $\mathbf{U}^{1}$ twice, while in $M^{2}$ $\mathbf{R}$ counts the
bit parity of every node in $\mathbf{U}^{2} \cap F'$ twice. Thus, in both models
with no interventions, the bit parity of $\mathbf{R}$ is even. Next, fix two
distinct instantiations of $\mathbf{U}$ that differ by values of $U^{*}$.
Consider the topmost node $W \in F$ with an odd number of parents in $U^{*}$
(which exists because bidirected edges in $F$ form a spanning tree). Then
flipping the values of $U^{*}$ once will flip the value $W$ once. Thus the
function from $\mathbf{U}$ to $\mathbf{V}$ induced by a C-forest $F$ in $M^{1}$
and $M^{2}$ is one to one. The above results, coupled with the fact that in a
C-forest, $|\mathbf{U}|+1=|\mathbf{V}|$ implies that any assignment where
$\sum \mathbf{r} \pmod 2 = 0$ is equally likely, and all other node assignments
are impossible in both $F$ and $F'$. Since the two models agree on all functions
and distributions in $F \cap F'$,
$\sum_{f\backslash f'}P^{1} = \sum_{f\backslash f'}P^{2}$. It follows that the
observational distributions are the same in both models. Furthermore,
$\sum_{\mathbf{r}}P^{1}(\mathbf{V})$ is a positive distribution, thus
$P^{1}(\mathbf{x}|Pa(\mathbf{X})_{G} \backslash \mathbf{X})>0$ for any
$\mathbf{x}$. As before, we can find $U \in \mathbf{U}$ with one child in
$An(\mathbf{X})_{F}$, and one child in $F\backslash An(\mathbf{X})_{F}$, which
implies $P_{\mathbf{x}}^{1}(\sum \mathbf{r} \pmod 2 = 1)>0$ in $M^{1}$, but not
$M^{2}$. Since $P_{\mathbf{x}}(\mathbf{r})$ is not identifiable in $G$, and
$\mathbf{R} \subset An(\mathbf{Y})_{G_{\overline{\mathbf{X}}}}$, we can
construct $P(\mathbf{Y}|\mathbf{R})$ to be a one to one mapping between
$P_{\mathbf{x}}(\mathbf{r})$ and $P_{\mathbf{x}}(\mathbf{y})$, as we did in
Corollary 1. For instance, let $\mathbf{Y}'$ be the minimal subset of
$\mathbf{Y}$ such that
$\mathbf{R} \subseteq An(\mathbf{Y}')_{G_{\overline{\mathbf{X}}}}$. Then let all
nodes in
$G' = An(\mathbf{Y}')_{G_{\overline{\mathbf{X}}}} \backslash An(\mathbf{R})$ be
equal to the bit parity of the parents. Without loss of generality, assume every
node in $G'$ has at most one child. Then every $R \in \mathbf{R}$ has a unique
downward path to $\mathbf{Y}'$ which means the bit parities of $\mathbf{R}$ and
$\mathbf{Y}'$ are the same. This implies the result. $\Box$

Hedges generalize not only the C-tree condition, but also the complete condition
for identification of $P_{x}$ from $P$ in [Tian & Pearl, 2002] which states that
if $Y$ is a child of $X$ and there a bidirected path from $X$ to $Y$ then (and
only then) $P_{x}$ is not identifiable. Let $G$ consist of $X$, $Y$ and the
nodes $W_{1},...,W_{k}$ on the bidirected path from $X$ to $Y$. It is not
difficult to check that $G$ and $G\backslash \{X\}$ form a hedge for
$P_{x}(Y,W_{1},...,W_{k})$.

Since hedges generalize two complete conditions for special cases of the
identification problem, it might be reasonable to suppose that a complete
characterization of identifiability might involve hedges in some way. To prove
this supposition, we would need to construct an algorithm which identifies any
effect lacking a hedge. This algorithm is the subject of the next section.

A Complete Identification Algorithm

Given the characterization of unidentifiable effects in the previous section, we
can attempt to solve the identification problem in all other cases, and hope for
completeness. To do this we construct an algorithm that systematically takes
advantage of the properties of C-components to decompose the identification
problem into smaller subproblems until either the entire expression is
identified, or we run into the problematic hedge structure. This algorithm,
called **ID**, is shown in Fig. 3.

_(Algorithm details corresponding to Fig 3:)_ function
ID($\mathbf{y}, \mathbf{x}, P, G$) INPUT: $\mathbf{x},\mathbf{y}$ value
assignments, $P$ a probability distribution, $G$ a causal diagram (an I-map of
$P$). OUTPUT: Expression for $P_{\mathbf{x}}(\mathbf{y})$ in terms of $P$ or
FAIL($F, F'$).

1. if $\mathbf{x}=\emptyset$, return
   $\sum_{\mathbf{v}\backslash \mathbf{y}}P(\mathbf{v})$.

2. if $\mathbf{V} \ne An(\mathbf{Y})_{G}$, return
   ID($\mathbf{y}, \mathbf{x} \cap An(\mathbf{Y})_{G}, P(An(\mathbf{Y})), An(\mathbf{Y})_{G}$).

3. let
   $\mathbf{W}=(\mathbf{V}\backslash \mathbf{X})\backslash An(\mathbf{Y})_{G_{\overline{\mathbf{X}}}}$.
   if $\mathbf{W} \ne \emptyset$, return
   ID($\mathbf{y}, \mathbf{x} \cup \mathbf{w}, P, G$).

4. $C(G\backslash \mathbf{X}) = \{S_{1},...,S_{k}\}$, return
   $\sum_{\mathbf{v}\backslash(\mathbf{y}\cup\mathbf{x})}\prod_{i} ID(s_{i},\mathbf{v}\backslash s_{i},P,G)$.
   if $C(G\backslash \mathbf{X}) = \{S\}$,

5. if $C(G)=\{G\}$ throw FAIL($G, S$).

6. if $S \in C(G)$ return
   $\sum_{s\backslash \mathbf{y}}\prod_{V_{i} \in S}P(v_{i}|v_{\pi}^{(i-1)})$.

7. if $(\exists S') S \subset S' \in C(G)$, return
   ID($\mathbf{y}, \mathbf{x} \cap S', \prod_{V_{i} \in S'}P(V_{i}|V_{\pi}^{(i-1)} \cap S', v_{\pi}^{(i-1)}\backslash S'), S'$).
   _(Note: $\pi$ is some topological ordering of nodes in $G$. FAIL propagates
   through recursive calls like an exception, and returns $F, F'$ which form the
   hedge which witnesses non-identifiability of $P_{\mathbf{x}}(\mathbf{y})$.)\_

Before showing the soundness and completeness properties of **ID**, we give the
following example of the operation of the algorithm. Consider the graph $G$ in
Fig. 1 (a), where we want to identify $P_{x}(y_{1},y_{2})$ from $P(\mathbf{V})$.
We know that $G = An(\{Y_{1},Y_{2}\})_{G}$,
$C(G\backslash \{X\}) = \{G\backslash \{X\}\}$, and $\mathbf{W}=\{W_{1}\}$.
Thus, we invoke line 3 and attempt to identify $P_{x,w}(y_{1},y_{2})$. Now
$C(G\backslash \{X,W\}) = \{\{Y_{1}\},\{W_{2}\},\{Y_{2}\}\}$ so we invoke
line 4. Thus the original problem reduces to identifying
$\sum_{w_{2}}P_{x,w_{1},w_{2},y_{2}}(y_{1})P_{w_{1},x,y_{1},y_{2}}(w_{2})P_{x,w_{1},w_{2},y_{1}}(y_{2})$.
Solving for the second expression, we trigger line 2, noting that we can ignore
nodes which are not ancestors of $W_{2}$. Thus,
$P_{w_{1},x,y_{1},y_{2}}(w_{2})=P(w_{2})$. Similarly, we ignore non-ancestors of
$Y_{2}$ in the third expression to obtain
$P_{x,w_{1},w_{2},y_{1}}(y_{2})=P_{w_{2}}(y_{2})$. We conclude at line 6, to
obtain $P_{w_{2}}(y_{2})=P(y_{2}|w_{2})$. Solving for the first expression, we
first trigger line 2 also, obtaining
$P_{x,w_{1},w_{2},y_{2}}(y_{1})=P_{x,w_{1}}(y_{1})$. Next, we trigger line 7,
reducing the problem to computing $P_{w_{1}}(y_{1})$ from
$P(Y_{1}|X,W_{1})P(W_{1})$. Finally, we trigger line 2, obtaining
$P_{w_{1}}(y_{1})=\sum_{w_{1}}P(y_{1}|x,w_{1})P(w_{1})$. Putting everything
together, we obtain:
$P_{x}(y_{1},y_{2}) = \sum_{w_{2}}P(y_{2}|w_{2})P(w_{2})\sum_{w_{1}}P(y_{1}|x,w_{1})P(w_{1})$.

As we showed before, the very same effect $P_{x}(y_{1},y_{2})$ in a very similar
graph $G'$ shown in Fig. 1 (b) is not identifiable due to the presence of
C-forests forming a hedge. We now prove that **ID** terminates and is sound.

**Lemma 3** **ID** always terminates. _Proof:_ At any call on line 7,
$(\exists X \in \mathbf{X})X \notin S'$ else the failure condition on line 5
would have been triggered. Thus any recursive call to **ID** reduces the size of
either the set $\mathbf{X}$ or the set $\mathbf{V}\backslash \mathbf{X}$. Since
both of these sets are finite, and their union forms $\mathbf{V}$, **ID** must
terminate. $\Box$

To show soundness, we need a number of utility lemmas justifying various lines
of the algorithm. Though some of these results are already known, we will
reprove them here using do-calculus to entail the results in the next section.
When we refer to do-calculus we will just refer to rule numbers (e.g. 'by rule
2'). Throughout the proofs we will fix some topological ordering of observable
nodes in $G$. First, we must show that an effect of the form
$P_{\mathbf{x}}(\mathbf{y})$ decomposes according to the set of C-components of
the graph $G\backslash \mathbf{X}$.

**Lemma 4** Let $M$ be a causal model with graph $G$. Let
$\mathbf{y}, \mathbf{x}$ be value assignments. Let
$C(G\backslash \mathbf{X}) = \{S_{1},...,S_{k}\}$. Then
$P_{\mathbf{x}}(\mathbf{y}) = \sum_{\mathbf{v}\backslash(\mathbf{y}\cup\mathbf{x})}\prod_{i}P_{\mathbf{v}\backslash s_{i}}(s_{i})$.
_Proof:_ Assume $\mathbf{X}=\emptyset$, and let
$A_{i} = An(S_{i})_{G} \backslash S_{i}$. Then
$\prod_{i}P_{\mathbf{v}\backslash s_{i}}(s_{i}) = \prod_{i}P_{a_{i}}(s_{i}) = \prod_{i}\prod_{V_{j} \in S_{i}}P_{a_{i}}(v_{j}|v_{\pi}^{(j-1)}\backslash a_{i}) = \prod_{i}\prod_{V_{j} \in S_{i}}P(v_{j}|v_{\pi}^{(j-1)}) = \prod_{i}P(v_{i}|v_{\pi}^{(i-1)}) = P(\mathbf{v})$.
The first identity is by rule 3, the second is by chain rule of probability. To
prove the third identity, we consider two cases. If
$A \in A_{i} \backslash V_{\pi}^{(j-1)}$, we can eliminate the intervention on
$A$ from the expression $P_{a_{i}}(v_{j}|v_{\pi}^{(j-1)})$ by rule 3, since
$(V_{j} \perp\perp A|V_{\pi}^{(j-1)})_{G_{\overline{A}_{i}}}$. If
$A \in A_{i} \cap V_{\pi}^{(j-1)}$, consider any back-door path from $A$ to
$V_{j}$. Any such path with a node not in $V_{\pi}^{(j-1)}$ will be d-separated
because, due to recursiveness, it must contain a blocked collider. Further, this
path must contain bidirected arcs only, since all nodes on this path are
conditioned or fixed. Because $A_{i} \cap S_{i} = \emptyset$, all such paths are
d-separated. The identity now follows from rule 2. The last two identities are
just grouping of terms, and application of chain rule. The same factorization
applies to the submodel $M_{\mathbf{x}}$ which induces the graph
$G\backslash \mathbf{X}$ which implies the result. $\Box$

The next lemma shows that to identify the effect on $\mathbf{Y}$, it is
sufficient to restrict our attention to the ancestor set of $\mathbf{Y}$,
thereby ensuring the soundness of line 2.

**Lemma 5** Let $\mathbf{X}' = \mathbf{X} \cap An(\mathbf{Y})_{G}$. Then
$P_{\mathbf{x}}(\mathbf{y})$ obtained from $P$ in $G$ is equal to
$P_{\mathbf{x}'}'(\mathbf{y})$ obtained from $P'=P(An(\mathbf{Y}))$ in
$An(\mathbf{Y})_{G}$. _Proof:_ Let
$\mathbf{W} = \mathbf{V} \backslash An(\mathbf{Y})_{G}$. Then the submodel
$M_{\mathbf{w}}$ induces the graph $G_{\mathbf{W}} = An(\mathbf{Y})_{G}$, and
its distribution is $P' = P_{\mathbf{w}}(An(\mathbf{Y})) = P(An(\mathbf{Y}))$ by
rule 3. Now
$P_{\mathbf{x}}(\mathbf{y}) = P_{\mathbf{x}'}(\mathbf{y}) = P_{\mathbf{x}',\mathbf{w}}(\mathbf{y}) = P_{\mathbf{x}'}'(\mathbf{y})$
by rule 3. $\Box$

Next, we use do-calculus to show that introducing additional interventions in
line 3 is sound as well.

**Lemma 6** Let
$\mathbf{W} = (\mathbf{V} \backslash \mathbf{X}) \backslash An(\mathbf{Y})_{G_{\overline{\mathbf{X}}}}$.
Then $P_{\mathbf{x}}(\mathbf{y}) = P_{\mathbf{x},\mathbf{w}}(\mathbf{y})$, where
$\mathbf{w}$ are arbitrary values of $\mathbf{W}$. _Proof:_ Note that by
assumption, $\mathbf{Y} \perp\perp \mathbf{W} | [cite_start]\mathbf{X}$ in
$G_{\overline{\mathbf{X}},\overline{\mathbf{W}}}$. The conclusion follows by
rule 3. $\Box$

Next, we must ensure the validity of the positive base case on line 6.

**Lemma 7** When the conditions of line 6 are satisfied,
$P_{\mathbf{x}}(\mathbf{y}) = \sum_{s\backslash \mathbf{y}}\prod_{V_{i} \in S}P(v_{i}|v_{\pi}^{(i-1)})$.
_Proof:_ If line 6 preconditions are met, then $G$ local to that recursive call
is partitioned into $S$ and $\mathbf{X}$, and there are no bidirected arcs from
$\mathbf{X}$ to $S$. The conclusion now follows from the proof of Lemma 4.
$\Box$

Finally, we show the soundness of the last recursive call.

**Lemma 8** Whenever the conditions of the last recursive call of **ID** are
satisfied, $P_{\mathbf{x}}$ obtained from $P$ in the graph $G$ is equal to
$P_{\mathbf{x} \cap S'}'$ obtained from
$P' = \prod_{V_{i} \in S'}P(V_{i}|V_{\pi}^{(i-1)} \cap S', v_{\pi}^{(i-1)}\backslash S')$
in the graph $S'$. _Proof:_ It is easy to see that when the last recursive call
executes, $\mathbf{X}$ and $S$ partition $G$, and
$\mathbf{X} \subset An(S)_{G}$. This implies that the submodel
$M_{\mathbf{x}\backslash S'}$ induces the graph
$G\backslash(\mathbf{X}\backslash S')=S'$. The distribution
$P_{\mathbf{x}\backslash S'}$ of $M_{\mathbf{x}\backslash S'}$ is equal to $P'$
by the proof of Lemma 4. It now follows that
$P_{\mathbf{x}} = P_{\mathbf{x} \cap S',\mathbf{x}\backslash S'} = P_{\mathbf{x} \cap S'}'$.
$\Box$

We can now show the soundness of **ID**.

**Theorem 5 (soundness)** Whenever **ID** returns an expression for
$P_{\mathbf{x}}(\mathbf{y})$ it is correct. _Proof:_ If $\mathbf{x}=\emptyset$,
the desired effect can be obtained from $P$ by marginalization, thus this base
case is clearly correct. The soundness of all other lines except the failing
line 5 has already been established. $\Box$

Finally, we can characterize the relationship between hedges and the inability
of **ID** to identify an effect.

**Theorem 6** Assume **ID** fails to identify $P_{\mathbf{x}}(\mathbf{y})$
(executes line 5). Then there exist $\mathbf{X}' \subseteq \mathbf{X}$,
$\mathbf{Y}' \subseteq \mathbf{Y}$ such that the graph pair $G, S$ returned by
the fail condition of **ID** contain as edge subgraphs C-forests $F, F'$ that
form a hedge for $P_{\mathbf{x}'}(\mathbf{y}')$. _Proof:_ Consider line 5, and
$G$ and $\mathbf{y}$ local to that recursive call. Let $\mathbf{R}$ be the root
set of $G$. Since $G$ is a single C-component, it is possible to remove a set of
directed arrows from $G$ while preserving the root set $\mathbf{R}$ such that
the resulting graph $F$ is an $\mathbf{R}$-rooted C-forest. Moreover, since
$F' = F \cap S$ is closed under descendants, and since only single directed
arrows were removed from $S$ to obtain $F'$, $F'$ is also a C-forest.
$F' \cap \mathbf{X} = \emptyset$, and $F \cap \mathbf{X} \ne \emptyset$ by
construction. $\mathbf{R} \subseteq An(\mathbf{Y})_{G_{\overline{\mathbf{X}}}}$
by lines 2 and 3 of the algorithm. It's also clear that $\mathbf{y}, \mathbf{x}$
local to the recursive call in question are subsets of the original input.
$\Box$

**Corollary 2 (completeness)** **ID** is complete. _Proof:_ By the previous
theorem, if **ID** fails, then $P_{\mathbf{x}'}(\mathbf{y}')$ is not
identifiable in a subgraph $H = An(\mathbf{Y})_{G} \cap De(F)_{G}$ of $G$.
Moreover, $\mathbf{X} \cap H = \mathbf{X}'$, by construction of $H$. As such, it
is easy to extend the counterexamples in Theorem 6 with variables independent of
$H$, with the resulting models inducing $G$, and witnessing the
unidentifiability of $P_{\mathbf{x}}(\mathbf{y})$. $\Box$

The following is now immediate. **Corollary 3 (hedge criterion)**
$P_{\mathbf{x}}(\mathbf{y})$ is identifiable from $P$ in $G$ if and only if
there does not exist a hedge for $P_{\mathbf{x}'}(\mathbf{y}')$ in $G$, for any
$\mathbf{X}' \subset \mathbf{X}$ and $\mathbf{Y}' \subset \mathbf{Y}$.

So far we have not only established completeness, but also fully characterized
graphically all situations where distributions of the form
$P_{\mathbf{x}}(\mathbf{y})$ are identifiable. We can use these results to
derive a characterization of identifiable models, that is, causal models where
all effects are identifiable.

**Corollary 4 (model identification)** Let $G$ be a semi-Markovian causal
diagram. Then all causal effects are identifiable in $G$ if and only if $G$ does
not contain a node $X$ connected to its child $Y$ by a bidirected path. _Proof:_
If $F, F'$ are C-forests which form a hedge for some effect, there must be a
variable $X \in F$, which is an ancestor of another variable $Y \in F$. Thus, if
no $X$ exists with a child $Y$ in the same C-component, then no hedge can exist
in $G$, and **ID** never reaches the fail condition. Thus all effects are
identifiable. Otherwise, $P_{x}$ is not identifiable by [Tian & Pearl, 2002].
$\Box$

The complete algorithm presented in this section can be viewed as a marriage of
graphical and algebraic approaches to identifiability. **ID** manipulates the
first three arguments algebraically, in a manner similar to do-calculus - not a
coincidental similarity as the following section will show. At the same time, if
we ignore the third argument, **ID** can be viewed as a purely graphical
algorithm which, given an effect suspected of being non-identifiable, constructs
the problematic hedge structure witnessing this property.

Connections to Existing Identification Algorithms

In the previous section we established that **ID** is a sound and complete
algorithm for all effects of the form $P_{\mathbf{x}}(\mathbf{y})$. It is
natural to ask whether this result can be used to show completeness of earlier
algorithms conjectured to be complete.

First we consider do-calculus, which can be viewed as a declarative
identification algorithm, with its completeness remaining an open question. We
show that the steps of the algorithm **ID** correspond to sequences of standard
probabilistic manipulations, and applications of rules of do-calculus, which
entails completeness of do-calculus for identifying unconditional effects.

**Theorem 7** The rules of do-calculus, together with standard probability
manipulations are complete for determining identifiability of all effects of the
form $P_{\mathbf{x}}(\mathbf{y})$. _Proof:_ We must show that all operations
corresponding to lines of **ID** correspond to sequences of standard probability
manipulations and applications of the rules of do-calculus. These manipulations
are done either on the effect expression $P_{\mathbf{x}}(\mathbf{y})$, or the
observational distribution $P$, until the algorithm either fails, or the two
expressions 'meet' by producing a single chain of manipulations. Line 1 is just
standard probability operations. Line 5 is a fail condition. The proof that
lines 2, 3, 4, 6, and 7 correspond to sequences of do-calculus manipulations
follows from Lemmas 5, 6, 4, 7, and 8 respectively. $\Box$

Next, we consider a version of an identification algorithm due to Tian, shown in
Fig. 5. The soundness of this algorithm has already been addressed elsewhere, so
we turn to the matter of completeness.

_(Note: A few minor corrections have been applied to apparent OCR artifacts in
the source text to align with the standard mathematical notation used throughout
the rest of the paper, such as correcting $D_2$ to $D_i$ and standardizing the
set difference and graph surgery notations)._

### Figure 5: An identification algorithm modified from [Tian, 2002]

**function** `c-identify`($C$, $T$, $Q[T]$)

**INPUT:** $C \subseteq T$, both are C-components, $Q[T]$ a probability
distribution

**OUTPUT:** Expression for $Q[C]$ in terms of $Q[T]$ or FAIL

Let $A = An(C)_T$.

1. If $A = C$, return $\sum_{T \setminus C} P$
2. If $A = T$, return FAIL
3. If $C \subset A \subset T$, there exists a C-component $T'$ such that
   $C \subset T' \subset A$ return `c-identify`($C$, $T'$, $Q[T']$) _(where
   $Q[T']$ is known to be computable from $\sum_{T \setminus A} Q[T]$)\_

---

**function** `identify`($\mathbf{y}$, $\mathbf{x}$, $P$, $G$)

**INPUT:** $\mathbf{x}, \mathbf{y}$ value assignments, $P$ a probability
distribution, $G$ a causal diagram.

**OUTPUT:** Expression for $P_{\mathbf{x}}(\mathbf{y})$ in terms of $P$ or FAIL.

1. Let $D = An(\mathbf{Y})_{G_{\overline{\mathbf{X}}}}$
2. Assume $C(D) = \{D_1, ..., D_k\}$, $C(G) = \{C_1, ..., C_m\}$
3. return $\sum_{D \setminus \mathbf{Y}} \prod_i$ `c-identify`($D_i$, $C_{D_i}$,
   $Q[C_{D_i}]$) where $(\forall i) D_i \subseteq C_{D_i}$

**Theorem 8** Assume **identify** fails to identify
$P_{\mathbf{x}}(\mathbf{y})$. Then there exist C-forests $F, F'$ forming a hedge
for $P_{\mathbf{x}'}(\mathbf{y}')$ where $\mathbf{X}' \subseteq \mathbf{X}$,
$\mathbf{Y}' \subseteq \mathbf{Y}$. _Proof:_ Assume **c-identify** fails.
Consider C-components $C, T$ local to the failed recursive call. Let
$\mathbf{R}$ be the root set of $C$. Because $T = An(C)_{T}$, $\mathbf{R}$ is
also a root set of $T$. As in the proof of Theorem 6, we can remove a set of
directed arrows from $C$ and $T$ while preserving $\mathbf{R}$ as the root set
such that the resulting edge subgraphs are C-forests. By line 1 of **identify**,
$C, T \subseteq An(\mathbf{Y})_{G_{\overline{\mathbf{X}}}}$. Finally, because
**c-identify** will always succeed if $D_{i} = C_{D_{i}}$, it must be the case
that $D_{i} \subset C_{D_{i}}$. But this implies
$\mathbf{X} \cap C = \emptyset$, $\mathbf{X} \cap T \ne \emptyset$. Thus, edge
subgraphs of $C$, and $T$ are C-forests forming a hedge for
$P_{\mathbf{x}'}(\mathbf{y}')$, where $\mathbf{X}' \subseteq \mathbf{X}$,
$\mathbf{Y}' \subset \mathbf{Y}$. $\Box$

**Corollary 5** **identify** is complete. _Proof:_ This is implied by Theorem 8,
and Corollary 3. $\Box$

Acknowledgments

We would like to thank Gunez Ercal, Manabu Kuroki, and Jin Tian for helpful
discussions on earlier versions of this paper. We also thank anonymous reviewers
whose comments helped improve this paper.

Conclusions

We have provided a complete characterization of cases when joint interventional
distributions are identifiable in semi-Markovian models. Using a graphical
structure called the hedge, we were able to construct a sound and complete
algorithm for this identification problem, prove completeness of two existing
algorithms, and derive a complete description of semi-Markovian models in which
all effects are identifiable.

The natural open question stemming from this work is whether the algorithm
presented can lead to the identification of conditional interventional
distributions of the form $P_{\mathbf{x}}(\mathbf{y}|\mathbf{z})$. Another
remaining question is whether the results in this paper could prove helpful for
identifying general counterfactual expressions such as those invoked in natural
direct and indirect effects [Pearl, 2001], and path-specific effects [Avin,
Shpitser, & Pearl, 2005].

---

References

```bibtex
@inproceedings{avin2005identifiability,
  title={Identifiability of path-specific effects},
  author={Avin, C. and Shpitser, I. and Pearl, J.},
  booktitle={International Joint Conference on Artificial Intelligence},
  volume={19},
  pages={357--363},
  year={2005}
}

@article{dawid1979conditional,
  title={Conditional independence in statistical theory},
  author={Dawid, A. P.},
  journal={Journal of the Royal Statistical Society},
  volume={41},
  pages={1--31},
  year={1979}
}

@article{halpern2000axiomatizing,
  title={Axiomatizing causal reasoning},
  author={Halpern, J.},
  journal={Journal of A.I. Research},
  pages={317--337},
  year={2000}
}

@techreport{huang2006completeness,
  title={On the completeness of an identifiability algorithm for semi-markovian models},
  author={Huang, Y. and Valtorta, M.},
  year={2006},
  institution={Computer Science and Engineering Department, University of South Carolina},
  number={TR-2006-01}
}

@article{kuroki1999identifiability,
  title={Identifiability criteria for causal effects of joint interventions},
  author={Kuroki, M. and Miyakawa, M.},
  journal={Journal of Japan Statistical Society},
  volume={29},
  pages={105--117},
  year={1999}
}

@inproceedings{pearl1995probabilistic,
  title={Probabilistic evaluation of sequential plans from causal models with hidden variables},
  author={Pearl, J. and Robins, J. M.},
  booktitle={Uncertainty in Artificial Intelligence},
  volume={11},
  pages={444--453},
  year={1995}
}

@book{pearl1988probabilistic,
  title={Probabilistic Reasoning in Intelligent Systems},
  author={Pearl, J.},
  year={1988},
  publisher={Morgan and Kaufmann},
  address={San Mateo}
}

@article{pearl1995causal,
  title={Causal diagrams for empirical research},
  author={Pearl, J.},
  journal={Biometrika},
  volume={82},
  number={4},
  pages={669--709},
  year={1995}
}

@book{pearl2000causality,
  title={Causality: Models, Reasoning, and Inference},
  author={Pearl, J.},
  year={2000},
  publisher={Cambridge University Press}
}

@inproceedings{pearl2001direct,
  title={Direct and indirect effects},
  author={Pearl, J.},
  booktitle={Proceedings of UAI-01},
  pages={411--420},
  year={2001}
}

@book{spirtes1993causation,
  title={Causation, Prediction, and Search},
  author={Spirtes, P. and Glymour, C. and Scheines, R.},
  year={1993},
  publisher={Springer Verlag},
  address={New York}
}

@inproceedings{tian2002general,
  title={A general identification condition for causal effects},
  author={Tian, J. and Pearl, J.},
  booktitle={Eighteenth National Conference on Artificial Intelligence},
  pages={567--573},
  year={2002}
}

@phdthesis{tian2002studies,
  title={Studies in Causal Reasoning and Learning},
  author={Tian, J.},
  year={2002},
  school={Department of Computer Science, University of California, Los Angeles}
}

```
