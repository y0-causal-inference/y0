<p align="center">
  <img src="docs/source/logo.png" height="120">
</p>

<h1 align="center">
  y0
</h1>

<p align="center">
    <a href="https://github.com/y0-causal-inference/y0/actions?query=workflow%3ATests">
        <img alt="Tests" src="https://github.com/y0-causal-inference/y0/workflows/Tests/badge.svg" />
    </a>
   <a href="https://github.com/cthoyt/cookiecutter-python-package">
      <img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-snekpack-blue" /> 
   </a>
    <a href="https://pypi.org/project/y0">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/y0" />
    </a>
    <a href="https://pypi.org/project/y0">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/y0" />
    </a>
    <a href="https://github.com/y0-causal-inference/y0/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/y0" />
    </a>
    <a href='https://y0.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/y0/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://zenodo.org/badge/latestdoi/328745468">
        <img src="https://zenodo.org/badge/328745468.svg" alt="DOI">
    </a>
    <a href="https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" />
    </a>
</p>

`y0` (pronounced "why not?") is Python code for causal inference.

## 💪 Getting Started

### Representing Probability Expressions

`y0` has a fully featured internal domain specific language for representing
probability expressions:

```python
from y0.dsl import P, A, B

# The probability of A given B
expr_1 = P(A | B)

# The probability of A given not B
expr_2 = P(A | ~B)

# The joint probability of A and B
expr_3 = P(A, B)
```

It can also be used to manipulate expressions:

```python
from y0.dsl import P, A, B, Sum

P(A, B).marginalize(A) == Sum[A](P(A, B))
P(A, B).conditional(A) == P(A, B) / Sum[A](P(A, B))
```

DSL objects can be converted into strings with `str()` and parsed back
using `y0.parser.parse_y0()`.

A full demo of the DSL can be found in this
[Jupyter Notebook](https://github.com/y0-causal-inference/y0/blob/main/notebooks/DSL%20Demo.ipynb)

### Representing Causality

`y0` has a notion of acyclic directed mixed graphs built on top of
`networkx` that can be used to model causality:

```python
from y0.graph import NxMixedGraph
from y0.dsl import X, Y, Z1, Z2

# Example from:
#   J. Pearl and D. Mackenzie (2018)
#   The Book of Why: The New Science of Cause and Effect.
#   Basic Books, p. 240.
napkin = NxMixedGraph.from_edges(
    directed=[
        (Z2, Z1),
        (Z1, X),
        (X, Y),
    ],
    undirected=[
        (Z2, X),
        (Z2, Y),
    ],
)
```

`y0` has many pre-written examples in `y0.examples` from Pearl, Shpitser,
Bareinboim, and others.

### do Calculus

`y0` provides _actual_ implementations of many algorithms that have remained
unimplemented for the last 15 years of publications including:

| Algorithm          | Reference                                                                   |
|--------------------|-----------------------------------------------------------------------------|
| ID                 | [Shpitser and Pearl, 2006](https://dl.acm.org/doi/10.5555/1597348.1597382)  |
| IDC                | [Shpitser and Pearl, 2008](https://www.jmlr.org/papers/v9/shpitser08a.html) |
| ID*                | [Shpitser and Pearl, 2012](https://arxiv.org/abs/1206.5294)                 |
| IDC*               | [Shpitser and Pearl, 2012](https://arxiv.org/abs/1206.5294)                 |
| Surrogate Outcomes | [Tikka and Karvanen, 2018](https://arxiv.org/abs/1806.07172)                |

Apply an algorithm to an ADMG and a causal query to generate an estimand
represented in the DSL like:

```python
from y0.dsl import P, X, Y
from y0.examples import napkin
from y0.algorithm.identify import Identification, identify

# TODO after ID* and IDC* are done, we'll update this interface
query = Identification.from_expression(graph=napkin, query=P(Y @ X))
estimand = identify(query)
assert estimand == P(Y @ X)
```

## 🚀 Installation

The most recent release can be installed from
[PyPI](https://pypi.org/project/y0/) with:

```bash
$ pip install y0
```

The most recent code and data can be installed directly from GitHub with:

```bash
$ pip install git+https://github.com/y0-causal-inference/y0.git
```

## 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.md](https://github.com/y0-causal-inference/y0/blob/master/.github/CONTRIBUTING.md) for more information on getting
involved.

## 👋 Attribution

### ⚖️ License

The code in this package is licensed under the [BSD-3-Clause
license](https://github.com/y0-causal-inference/y0/blob/master/LICENSE).

### 📖 Citation

Before we publish an application note on `y0`, you can cite this software
via our Zenodo record (also see the badge above):

```bibtex
@software{y0,
  author       = {Charles Tapley Hoyt and
                  Jeremy Zucker and
                  Marc-Antoine Parent},
  title        = {y0-causal-inference/y0},
  month        = jun,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.1.0},
  doi          = {10.5281/zenodo.4950768},
  url          = {https://doi.org/10.5281/zenodo.4950768}
}
```

### 🙏 Supporters

This project has been supported by several organizations (in alphabetical order):

- [Harvard Program in Therapeutic Science - Laboratory of Systems Pharmacology](https://hits.harvard.edu/the-program/laboratory-of-systems-pharmacology/)
- [Pacific Northwest National Laboratory](https://www.pnnl.org/)

### 💰 Funding

The development of the Y0 Causal Inference Engine has been funded by the following grants:

| Funding Body                                             | Program                                                                                                                       | Grant           |
|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|-----------------|
| DARPA                                                    | [Automating Scientific Knowledge Extraction (ASKE)](https://www.darpa.mil/program/automating-scientific-knowledge-extraction) | HR00111990009   |
| PNNL Data Model Convergence Initiative    | [Causal Inference and Machine Learning Methods for Analysis of Security Constrained Unit Commitment (SCY0)](https://www.pnnl.gov/projects/dmc/converged-applications-projects) | 90001   |
| DARPA                                                    |  [Automating Scientific Knowledge Extraction and Modeling (ASKEM)](https://www.darpa.mil/program/automating-scientific-knowledge-extraction-and-modeling) |  HR00112220036  |

### 🍪 Cookiecutter

This package was created with [@audreyfeldroy](https://github.com/audreyfeldroy)'s
[cookiecutter](https://github.com/cookiecutter/cookiecutter) package using [@cthoyt](https://github.com/cthoyt)'s
[cookiecutter-python-package](https://github.com/cthoyt/cookiecutter-python-package) template.

## 🛠️ Development

<details>
  <summary>See developer instructions</summary>

The final section of the README is for if you want to get involved by making a code contribution.

### Developer Installation

To install in development mode, use the following:

```bash
$ git clone git+https://github.com/y0-causal-inference/y0.git
$ cd y0
$ pip install -e .
```

### ❓ Testing

After cloning the repository and installing `tox` with `pip install tox`, the unit tests in the `tests/` folder can be
run reproducibly with:

```shell
$ tox
```

Additionally, these tests are automatically re-run with each commit in a [GitHub Action](https://github.com/y0-causal-inference/y0/actions?query=workflow%3ATests).

### 📦 Making a Release

After installing the package in development mode and installing
`tox` with `pip install tox`, the commands for making a new release are contained within the `finish` environment
in `tox.ini`. Run the following from the shell:

```shell
$ tox -e finish
```

This script does the following:

1. Uses BumpVersion to switch the version number in the `setup.cfg` and
   `src/y0/version.py` to not have the `-dev` suffix
2. Packages the code in both a tar archive and a wheel
3. Uploads to PyPI using `twine`. Be sure to have a `.pypirc` file configured to avoid the need for manual input at this
   step
4. Push to GitHub. You'll need to make a release going with the commit where the version was bumped.
5. Bump the version to the next patch. If you made big changes and want to bump the version by minor, you can
   use `tox -e bumpversion minor` after.

</details>
