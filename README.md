<!--
<p align="center">
  <img src="docs/source/logo.png" height="150">
</p>
-->

<h1 align="center">
  y0
</h1>

<p align="center">
    <a href="https://github.com/y0-causal-inference/y0/actions?query=workflow%3ATests">
        <img alt="Tests" src="https://github.com/y0-causal-inference/y0/workflows/Check%20mappings/badge.svg" />
    </a>
    <!-- Uncomment after deploying to PyPI
    <a href="https://pypi.org/project/y0">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/y0" />
    </a>
    <a href="https://pypi.org/project/y0">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/y0" />
    </a>
    <a href="https://github.com/y0-causal-inference/y0/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/y0" />
    </a>
    -->
</p>

Python code for causal modeling

## 💪 Getting Started

> TODO show in a very small amount of space the **MOST** useful thing your package can do.
Make it as short as possible! You have an entire set of docs for later.

### Command Line Interface

The y0 command line tool is automatically installed. It can
be used from the shell with the `--help` flag to show all subcommands:

```shell
$ y0 --help
```

> TODO show the most useful thing the CLI does! The CLI will have document auto-generated
by sphinx.

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

The code in this package is licensed under the MIT License.

## 🙏 Contributing
Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.rst](https://github.com/y0-causal-inference/y0/blob/master/CONTRIBUTING.rst) for more information on getting
involved.

## 🍪 Cookiecutter Acknowledgement

This package was created with [@audreyr](https://github.com/audreyr)'s
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
