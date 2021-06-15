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
      <img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-python--package-yellow" /> 
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

`y0` (pronounced "why not?") is Python code for causal inferencing.

## 💪 Getting Started

> TODO show in a very small amount of space the **MOST** useful thing your package can do.
Make it as short as possible! You have an entire set of docs for later.

## ⬇️ Installation

The most recent release can be installed from
[PyPI](https://pypi.org/project/y0/) with:

```bash
$ pip install y0
```

The most recent code and data can be installed directly from GitHub with:

```bash
$ pip install git+https://github.com/y0-causal-inference/y0.git
```

To install in development mode, use the following:

```bash
$ git clone git+https://github.com/y0-causal-inference/y0.git
$ cd y0
$ pip install -e .
```

## ⚖️ License

The code in this package is licensed under the [BSD-3-Clause
license](https://github.com/y0-causal-inference/y0/blob/master/LICENSE).

## 🙏 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.rst](https://github.com/y0-causal-inference/y0/blob/master/CONTRIBUTING.rst) for more information on getting
involved.

## Acknowledgements

### Supporters

This project has been supported by several organizations (in alphabetical order):

- [Harvard Program in Therapeutic Science - Laboratory of Systems Pharmacology](https://hits.harvard.edu/the-program/laboratory-of-systems-pharmacology/)
- [Pacific Northwest National Laboratory](https://www.pnnl.org/)

### 💰 Funding

The development of the Y0 Causal Inference Engine has been funded by the following grants:

| Funding Body                                             | Program                                                                                                                       | Grant           |
|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|-----------------|
| DARPA                                                    | [Automating Scientific Knowledge Extraction (ASKE)](https://www.darpa.mil/program/automating-scientific-knowledge-extraction) | HR00111990009   |
| PNNL Data Model Convergence Initiative    | [Causal Inference and Machine Learning Methods for Analysis of Security Constrained Unit Commitment (SCY0)](https://www.pnnl.gov/projects/dmc/converged-applications-projects) | 90001   |

### 🍪 Cookiecutter

This package was created with [@audreyfeldroy](https://github.com/audreyfeldroy)'s
[cookiecutter](https://github.com/cookiecutter/cookiecutter) package using [@cthoyt](https://github.com/cthoyt)'s
[cookiecutter-python-package](https://github.com/cthoyt/cookiecutter-python-package) template.

## 🛠️ Development

The final section of the README is for if you want to get involved by making a code contribution.

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
