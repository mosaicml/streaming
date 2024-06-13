# 1. Style and Conventions

## 1.1 Style Guide

Streaming generally follows Google's
[Python Style Guide](https://google.github.io/styleguide/pyguide.html) for how to format and structure code.

## 1.2. Pre-Commit Hooks

Streaming uses [Pre Commit](https://pre-commit.com/) to enforce style checks. To configure, run
```
pip install '.[dev]'  # if not already installed
pre-commit install
```

The pre-commit hooks will now be run before each commit. You can also run the hooks manually via:

```
pre-commit run  # run all hooks on changed files
pre-commit run --all-files  # or, run all hooks on all files
```

## 1.3. Code Formatting

Streaming uses the [yapf](https://github.com/google/yapf) formatter for general formatting
[isort](https://github.com/PyCQA/isort) to sort imports. These checks run through pre-commit
(see section 2.2). These checks can also be run manually via:

```
pre-commit run yapf --all-files  # for yapf
pre-commit run isort --all-files  # for isort
```

The configuration is stored in [pyproject.toml](pyproject.toml).

## 1.4. Code Structure

As a general rule of thumb,

-   Don't: Default to using inheritance for code reuse

    Do: prefer [composition over inheritance](https://en.wikipedia.org/wiki/Composition_over_inheritance)
-   Don't: strive to implement all logic using classes

    Do: strive to implement logic as pure functions when possible, and classes when there is good reason
-   Don't: Have a function accept falsy values that would then result in a no-op.

    Example of the anti-pattern:

    ```python
    from typing import Optional

    def custom_configuration(config: Optional[dict]):
        if config is None:
            # Don't do this check in the callee, which results in a no-op
            return
        ...
    ```

    Do: Require the caller, instead of the callee, check for and handle falsy values. It's ok to accept falsy values
    for individual arguments of a caller function, so long as the entire function would not be a no-op.

    Example:
    ```python
    from typing import Optional

    def custom_configuration(config: dict):
        ...

    def trainer(config: Optional[dict]):
        if config is not None:
            # Do this check in the caller function
            custom_configuration(config)
        ...
    ```

# 2. Type Annotations and Typechecking

Streaming aims to annotate all functions with type annotations (introduced in
[PEP 526](https://www.python.org/dev/peps/pep-0526/)). Type annotations help statically catch `TypeError` and
`AttributeError` bugs, in addition to other benefits, as outlined in the PEP.

For documentation on typing annotations, see:
* [PEP 483](https://peps.python.org/pep-0483/) for a simplified introduction
* [PEP 484](https://peps.python.org/pep-0484/) for the full specification
* [Python docs for `typing`](https://docs.python.org/3/library/typing.html) for the API reference

Streaming uses [pyright](https://github.com/microsoft/pyright)
to validate type annotations. PyRight is automatically run as part of the pre-commit hooks, but you can also
run PyRight specifically via:

```
pre-commit run pyright --all-files
```

The pyright configuration is stored in [pyproject.toml](pyproject.toml).


## 2.1 Debugging

Here are some suggestions to deal with pyright errors:

1. Suppose a variable could be one of multiple types, like the following:

    ```python
    from typing import Union

    def foo(x: Union[int, None]):
        return x + 5  # type error -- None + 5 is not allowed!
    ```

    PyRight will complain since `None + 5` is not a valid operation.
    Instead, add a check to ensure that `x is not None`:

    ```python
    from typing import Union

    def foo(x: Union[int, None]):
        if x is None:
            raise TypeError("x must be an integer, not None!")
        return x + 5  # valid
    ```

    Assert statements also work. However, assert statements should not be used for data validation
    (see the assert statement section below).
    ```python
    from typing import Union

    def foo(x: Union[int, None]):
        assert x is not None, "x should never be None"
        return x + 5  # valid
    ```

1. For variables where it is impossible for pyright to infer the correct type, use
[cast](https://docs.python.org/3/library/typing.html#typing.cast).
1. As a last resort, add a `# type: ignore` comment to the line where pyright emits an error.
Immediately following this statement, paste in the error emitted by pyright,
so other contributors will know why this error was silenced.


# 3. Public APIs
A public API, generally speaking, can be invoked by a user without a leading underscore in any portion of the path.
The following are examples of public APIs:

* Standalone functions in public modules (e.g. `streaming.base.distributed.get_world_size`)
* Classes in public modules (e.g. `streaming.base.format.MDSWriter`)
* Public methods in public classes (e.g. `streaming.base.format.MDSWriter.write`)
* Public modules (e.g. `streaming.base.dataset`)

The following rules apply to public APIs:
1. All public APIs must have a docstring (see the Documentation section below)
1. All parameters must have type annotations.
1. To minimize user imports, parameters should should use native PyTorch or Python types whenever possible.

    It is acceptable to use a union of types, so long as one of the options is a primitive.

1. Parameters that could take a sequence of elements should also allow `None` or a singleton.
    This simplifies the user API by not having to construct a list (or tuple) to hold a single element
    (or no element). For example, use `Optional[Union[torch.Tensor, Sequence[torch.Tensor]]`.


# 4. Use of `assert`

`assert` should be used only in test cases and for verifying invariants (likely required for type checking),
not for data validation. As asserts can be disabled in python by using the `-O` flag
(e.g. `python -O path/to/script.py`), they are not guaranteed to run. For data validation, instead use a style like
the following:

<!--pytest.mark.xfail-->
<!--
```python
parameter = None
```
-->
<!--pytest-codeblocks:cont-->
```python
if parameter is None:
    raise ValueError("parameter must be specified and cannot be None")
```


# 5. Imports and `__init__.py`

All imports in Streaming should be absolute -- that is, they do not begin with a period.

## 5.1 External Dependencies
1.  All external dependencies must be specified in both [setup.py](setup.py) for pip.

1.  If a dependency is not core to Streaming (e.g. it is for a model, dataset, or some callbacks):
    1.  It must be specified in a entry of the `extra_deps` dictionary of [setup.py](setup.py).
        This dictionary groups dependencies that can be conditionally installed. An entry named `foo`
        can be installed with `pip install 'mosaicml-streaming[foo]'`. For example, running `pip install 'mosaicml-streaming[docs]'`
        will install everything in `install_requires`, along with `docs`.
    1.  It must also be specified in the `run_constrained` and the `test.requires` section.
    1.  If the dependency is core to Streaming, add the dependency to the `install_requires` section of
        [setup.py](./setup.py).

## 5.2 Use of `__all__`

All public modules must define `__all__` to be the list of members that should be re-exported.
The variable is necessary to 1) limit what `from XXX import *` imports, and 2) ensure that the documentation only
includes exported members, not unrelated re-imports.

For example, from [streaming/base/dataset.py](streaming/base/dataset.py)

```python
"""The :class:`Dataset` class, used for building streaming iterable datasets."""
from torch.utils.data import IterableDataset

from streaming.base.format import reader_from_json
from streaming.base.spanner import Spanner

__all__ = ["Dataset"]  # export only the Dataset, not other imports like `Spanner` or `reader_from_json`


class Dataset(IterableDataset):
    ...
```


## 5.3 `__init__.py`

All public classes and functions should be added to the module's `__init__.py`.

<!--pytest.mark.skip-->
```python
from streaming.path.to.module.file import MyClass as MyClass
from streaming.path.to.module.file import my_func as my_func
```

If a file only contains public functions, then the following is also acceptable:

<!--pytest.mark.skip-->
```python
from streaming.path.to.module import my_file as my_file
```


# 6. Documentation

## 6.1 Docstrings

Streaming uses [Google Style Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
All public APIs require documentation.

### 6.1.1 What to include in Docstrings?

Docstrings, at a minimum, should include a summary of what the function or class does, along with the arguments it takes. See [below](#612-formatting-docstrings) for how to format docstrings. The [Google Style Guide](https://google.github.io/styleguide/pyguide.html) also includes some guidelines on how to write docstrings.

### 6.1.2 Formatting Docstrings

The following guidelines apply to documentation.
1.  Each function that needs a docstring must have its input arguments, return statement (if not None), and any custom
    exceptions annotated.
1.  The arguments for the `__init__` signature of classes should be documented under the class-level docstring. There
    should not be any `__init__`-level docstring.
1.  Each argument annotation should include the type. If the argument has a default value, the type annotation should
    specify "optional", and the docstring should say the default value. Some examples:

    ```python
    from typing import Optional, Union

    def foo(bar: int):
        """Foo.

        Args:
            bar (int): Required bar.
        """
        ...

    def foo2(bar: int = 42):
        """Foo2.

        Args:
            bar (int, optional): The first Argument. Default: ``42``.
        """
        ...

    def foo3(bar: Optional[int] = None):
        """Foo3.

        Args:
            bar (int, optional): The first Argument. Default: ``None``.
        """
        ...

    def foo4(bar: Union[int, str] = 42):
        """Foo4.

        Args:
            bar (int | str, optional): The first Argument. Default: ``42``.
        """
        ...

    def foo5(bar: int) -> int:
        """Foo5.

        Args:
            bar (int): Required bar.

        Returns:
            int: Description of return statement.
        """
        ...

    def foo6(bar: int) -> tuple[int, str]:
        """Foo6.

        Args:
            bar (int): Required bar.

        Returns:
            a (int): Returned value.
            b (str): Returned value.
        """
        ...
    ```

### 6.1.3 Building and Viewing Docs Locally

Assuming you already have a development install of Streaming (see these [instructions](CONTRIBUTING.md#prerequisites)), here’s how to build and previous the docs locally.

**️️ ⚠ Warning:** CI treats all sphinx warnings as errors, so they must be addressed before a PR can be merged. Building docs locally can help debug any warnings showing up on Jenkins!

In a terminal, run:

<!--pytest.mark.skip-->
```bash
source path/to/streaming_venv/bin/activate  # activate your streaming virtual env
cd streaming/docs  # cd to the docs folder inside your streaming clone
make clean  # Cleans the artifacts and remove source/api_reference folder
make html   # build the docs
make host   # Run the docs locally
```

Then, navigate to [http://localhost:8000](http://localhost:8000) in your browser.

## 6.2 Doctests

Most docstrings should also include a `.. doctest` or `.. testcode` example to clearly illustrate how one would interact with the class or function. As part of the CI/CD process, all `.. doctest` blocks are executed to ensure the example in the documentation actually works.

### 6.2.1 Writing Doctests

See the [Sphinx Doctest Extension](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html) for all of the available directives. Do not use `.. code-block::` for Python examples, as they are untested.

Any test fixtures for doctests should go in [docs/source/doctest_fixtures.py](docs/source/doctest_fixtures.py) or in a `.. testsetup::` block.

For example:
```python
import torch
from typing import Optional

def my_function(x: Optional[torch.Tensor]) -> torch.Tensor:
    """blah function

    Args:
        input (torch.Tensor): Your guess.

    Returns:
        torch.Tensor: How good your input is.

    Raises:
        ValueError: If your input is negative.

    Example:
        .. testsetup::

            # optional setup section, not shown in docs
            import torch
            x = torch.randn(42)


        .. testcode::

            # shown in docs; runs after testsetup
            my_function(x)
    """
    ...
```

All doctests load the [docs/source/doctest_fixtures.py](docs/source/doctest_fixtures.py) file *before* tests run. If there are any variables that would be helpful have defined for all tests, feel free to add them into this file. However, if a variable is more specific to an individual doctest, then it would be best to include it in a `.. testsetup::` block, as not to pollute the global fixture namespace. (Unlike pytest fixtures, all doctest fixtures are given to every doctest; they cannot be specifically requested)

### 6.2.2 Running Doctests Locally

Assuming you already have a development install of Streaming (see these [instructions](CONTRIBUTING.md#prerequisites)), here’s how to run the doctests.

<!--pytest.mark.skip-->
```bash
source path/to/streaming_venv/bin/activate  # activate your streaming virtual env
cd streaming/docs  # cd to the docs folder inside your streaming clone
make clean  # Cleans the artifacts and remove source/api_reference folder
make html  # the html build must be completed first to ensure all doctests are identified
make doctest 2>/dev/null # For more verbosity, do not direct stderr to /dev/null
```
