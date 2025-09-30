# How to Contribute
We'd love to get patches from you!

#### Workflow
We follow the [GitHub Flow Workflow](https://guides.github.com/introduction/flow/):

1.  Fork the project 
1.  Check out the `main` branch 
1.  Create a feature branch
1.  Write code and tests for your change 
1.  From your branch, make a pull request against `https://github.com/oreoruuser/jakube` 
1.  Work with repo maintainers to get your change reviewed 
1.  Wait for your change to be pulled into `https://github.com/oreoruuser/jakube/tree/main` 
1.  Delete your feature branch

## Getting Started
### Prerequisites
To compile Voyager from scratch, the following packages will need to be installed:

- [Python 3.7](https://www.python.org/downloads/) or higher.
- A C++ compiler, e.g. `gcc`, `clang`, etc.

### Building Voyager
#### Building Python
There are some nuances to building the Voyager python code.  Please read on for more information.
Here python3.xx refers to the version of python you are using, e.g. python3.9, python3.10, etc.
For basic building, you should be able to simply run the following commands:
```shell
cd python
python3.xx -m pip install -r dev-requirements.txt
python3.xx -m pip install .
```

> If you're on macOS or Linux, you can try to compile a debug build _faster_ by using [Ccache](https://ccache.dev/):
> ## macOS
> ```shell
> brew install ccache
> rm -rf build && CC="ccache clang" CXX="ccache clang++" DEBUG=1 MAX_JOBS=8 python -m pip install -e .
> ```
> ## Linux
> e.g.
> ```shell
> sudo yum install ccache  # or apt, if on a Debian
> 
> # If using GCC:
> rm -rf build && CC="ccache gcc" CXX="ccache g++" DEBUG=1 MAX_JOBS=8 python -m pip install -e .
> 
> # ...or if using Clang:
> rm -rf build && CC="ccache clang" CXX="ccache clang++" DEBUG=1 MAX_JOBS=8 python -m pip install -e .
> ```

#### Building C++
To build the C++ library with `cmake`, use the following commands:
```shell
cd cpp
git submodule update --init --recursive
make build
```

## Testing
### Python Tests
We use `tox` for testing - running tests from end-to-end should be as simple as:

```shell
cd python
python3.xx -m pip install tox
tox
```

### C++ Tests
To run the C++ tests, use the following commands:
```shell
cd cpp
git submodule update --init --recursive
make test
```

## Style
Use [`clang-format`](https://clang.llvm.org/docs/ClangFormat.html) for C++ code, and `black` with defaults for Python code.

### Python
In order to check and run formatting within the python module, you can use tox to facilitate this.
```bash
cd python
# Check formatting only (don't change files)
tox -e check-formatting
# Run formatter for python bindings and native python code
tox -e format
```

### C++
If you are working on any C++ code throughout the repo, ensure you have `clang-format` (version 16) installed, and then use clang-format to handle C++ formatting:
```bash
cd cpp
cmake .
# Check formatting only (don't change files)
make check-formatting
# Run formatter 
make format
```

## Issues
When creating an issue please try to adhere to the following format:

    One line summary of the issue (less than 72 characters)

    ### Expected behaviour

    As concisely as possible, describe the expected behaviour.

    ### Actual behaviour

    As concisely as possible, describe the observed behaviour.

    ### Steps to reproduce the behaviour

    List all relevant steps to reproduce the observed behaviour.

## First Contributions
If you are a first time contributor to `voyager`,  familiarize yourself with the:
* [Code of Conduct](CODE_OF_CONDUCT.md)
* [GitHub Flow Workflow](https://guides.github.com/introduction/flow/)

# License 
By contributing your code, you agree to license your contribution under the 
terms of the [LICENSE](https://github.com/oreoruuser/jakube/tree/main/LICENSE).

# Troubleshooting
## Building the project
### `ModuleNotFoundError: No module named 'nanobind'`
Try updating your version of `pip`:
```shell
python3.xx -m pip install --upgrade pip
```

### `Failed to establish a new connection: [Errno -2] Name or service not known'`
You may have networking issues. Check to make sure you do not have the `PIP_INDEX_URL` environment variable set (or that it points to a valid index).

### `fatal error: Python.h: No such file or directory`
Ensure you have the Python development packages installed.
You will need to find correct package for your operating system. (i.e.: `python-dev`, `python-devel`, etc.)

### `AttributeError: 'NoneType' object has no attribute 'group'`
- Ensure that you have Tox version 4 or greater installed
- _or_ set `ignore_basepython_conflict=true` in `tox.ini`
- _or_ install Tox using `pip` and not your system package manager