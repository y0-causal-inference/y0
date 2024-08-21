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
[cookiecutter-snekpack](https://github.com/cthoyt/cookiecutter-snekpack) template.

## 🛠️ For Developers

<details>
  <summary>See developer instructions</summary>

The final section of the README is for if you want to get involved by making a code contribution.

### Development Installation

To install in development mode, use the following:

```bash
git clone git+https://github.com/y0-causal-inference/y0.git
cd y0
pip install -e .
```

### Updating Package Boilerplate

This project uses `cruft` to keep boilerplate (i.e., configuration, contribution guidelines, documentation
configuration)
up-to-date with the upstream cookiecutter package. Update with the following:

```shell
pip install cruft
cruft update
```

More info on Cruft's update command is
available [here](https://github.com/cruft/cruft?tab=readme-ov-file#updating-a-project).

### 🥼 Testing

After cloning the repository and installing `tox` with `pip install tox tox-uv`, 
the unit tests in the `tests/` folder can be run reproducibly with:

```shell
tox -e py
```

Additionally, these tests are automatically re-run with each commit in a
[GitHub Action](https://github.com/y0-causal-inference/y0/actions?query=workflow%3ATests).

### 📖 Building the Documentation

The documentation can be built locally using the following:

```shell
git clone git+https://github.com/y0-causal-inference/y0.git
cd y0
tox -e docs
open docs/build/html/index.html
``` 

The documentation automatically installs the package as well as the `docs`
extra specified in the [`pyproject.toml`](pyproject.toml). `sphinx` plugins
like `texext` can be added there. Additionally, they need to be added to the
`extensions` list in [`docs/source/conf.py`](docs/source/conf.py).

The documentation can be deployed to [ReadTheDocs](https://readthedocs.io) using
[this guide](https://docs.readthedocs.io/en/stable/intro/import-guide.html).
The [`.readthedocs.yml`](../../dev/y0/.readthedocs.yml) YAML file contains all the configuration you'll need.
You can also set up continuous integration on GitHub to check not only that
Sphinx can build the documentation in an isolated environment (i.e., with ``tox -e docs-test``)
but also that [ReadTheDocs can build it too](https://docs.readthedocs.io/en/stable/pull-requests.html).

#### Configuring ReadTheDocs

1. Log in to ReadTheDocs with your GitHub account to install the integration
   at https://readthedocs.org/accounts/login/?next=/dashboard/
2. Import your project by navigating to https://readthedocs.org/dashboard/import then clicking the plus icon next to
   your repository
3. You can rename the repository on the next screen using a more stylized name (i.e., with spaces and capital letters)
4. Click next, and you're good to go!

### 📦 Making a Release

#### Configuring Zenodo

[Zenodo](https://zenodo.org) is a long-term archival system that assigns a DOI to each release of your package.

1. Log in to Zenodo via GitHub with this link: https://zenodo.org/oauth/login/github/?next=%2F. This brings you to a
   page that lists all of your organizations and asks you to approve installing the Zenodo app on GitHub. Click "grant"
   next to any organizations you want to enable the integration for, then click the big green "approve" button. This
   step only needs to be done once.
2. Navigate to https://zenodo.org/account/settings/github/, which lists all of your GitHub repositories (both in your
   username and any organizations you enabled). Click the on/off toggle for any relevant repositories. When you make
   a new repository, you'll have to come back to this

After these steps, you're ready to go! After you make "release" on GitHub (steps for this are below), you can navigate
to https://zenodo.org/account/settings/github/repository/y0-causal-inference/y0
to see the DOI for the release and link to the Zenodo record for it.

#### Registering with the Python Package Index (PyPI)

You only have to do the following steps once.

1. Register for an account on the [Python Package Index (PyPI)](https://pypi.org/account/register)
2. Navigate to https://pypi.org/manage/account and make sure you have verified your email address. A verification email
   might not have been sent by default, so you might have to click the "options" dropdown next to your address to get to
   the "re-send verification email" button
3. 2-Factor authentication is required for PyPI since the end of 2023 (see
   this [blog post from PyPI](https://blog.pypi.org/posts/2023-05-25-securing-pypi-with-2fa/)). This means
   you have to first issue account recovery codes, then set up 2-factor authentication
4. Issue an API token from https://pypi.org/manage/account/token

#### Configuring your machine's connection to PyPI

You have to do the following steps once per machine. Create a file in your home directory called
`.pypirc` and include the following:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = <the API token you just got>

# This block is optional in case you want to be able to make test releases to the Test PyPI server
[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = <an API token from test PyPI>
```

Note that since PyPI is requiring token-based authentication, we use `__token__` as the user, verbatim.
If you already have a `.pypirc` file with a `[distutils]` section, just make sure that there is an `index-servers`
key and that `pypi` is in its associated list. More information on configuring the `.pypirc` file can
be found [here](https://packaging.python.org/en/latest/specifications/pypirc).

#### Uploading to PyPI

After installing the package in development mode and installing
`tox` with `pip install tox tox-uv`,
run the following from the shell:

```shell
tox -e finish
```

This script does the following:

1. Uses [Bump2Version](https://github.com/c4urself/bump2version) to switch the version number in
   the `pyproject.toml`, `CITATION.cff`, `src/y0/version.py`,
   and [`docs/source/conf.py`](docs/source/conf.py) to not have the `-dev` suffix
2. Packages the code in both a tar archive and a wheel using [`build`](https://github.com/pypa/build)
3. Uploads to PyPI using [`twine`](https://github.com/pypa/twine).
4. Push to GitHub. You'll need to make a release going with the commit where the version was bumped.
5. Bump the version to the next patch. If you made big changes and want to bump the version by minor, you can
   use `tox -e bumpversion -- minor` after.

#### Releasing on GitHub

1. Navigate
   to https://github.com/y0-causal-inference/y0/releases/new
   to draft a new release
2. Click the "Choose a Tag" dropdown and select the tag corresponding to the release you just made
3. Click the "Generate Release Notes" button to get a quick outline of recent changes. Modify the title and description
   as you see fit
4. Click the big green "Publish Release" button

This will trigger Zenodo to assign a DOI to your release as well.

</details>
