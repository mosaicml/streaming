# Contributing to Streaming

Thanks for considering contributing to Streaming!

Issues tagged with [good first issue](https://github.com/mosaicml/streaming/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) are great options to start contributing.

If you have questions, join us on [Slack](https://dub.sh/mcomm) -- we'll be happy to help you!

We welcome contributions for bug fixes, new features you'd like to contribute to the community, or improve test suite!


## Prerequisites

To set up the development environment in your local box, run the commands below.

1\. Install the dependencies needed for testing and linting the code:

<!--pytest.mark.skip-->
```bash
pip install -e '.[dev]'

# Optional: If you would like to install all the dependencies
pip install -e '.[all]'
```

2\. Configure [pre-commit](https://pre-commit.com/), which automatically formats code before
each commit:

<!--pytest.mark.skip-->
```bash
pre-commit install
```

## Submitting a Contribution

To submit a contribution:

1\. Fork a copy of the [Streaming](https://github.com/mosaicml/streaming) library to your own account.

2\. Clone your fork locally and add the mosaicml repo as a remote repository:

<!--pytest.mark.skip-->
```bash
git clone git@github.com:<github_id>/streaming.git
cd streaming
git remote add upstream https://github.com/mosaicml/streaming.git
```

3\. Create a branch and make your proposed changes.

<!--pytest.mark.skip-->
```bash
cd streaming
git checkout -b cool-new-feature
```

4\. Run linting as part of `pre-commit`.

<!--pytest.mark.skip-->
```bash
git add <file1> <file2>
pre-commit run

# Optional: Run pre-commit for all files
pre-commit run --all-files
```

5\. Run the unit test to ensure it passes locally.

<!--pytest.mark.skip-->
```bash
ulimit -n unlimited # Workaround: To overcome 'Too many open files' issues since streaming uses atexit handler to close file descriptor at the end.

pytest -vv -s . # run all the unittests
cd docs && make clean && make doctest # run doctests
```

6\. [Optional] Compile and visualize the documentation locally. If you have a documentation changes, running the below commands is mandatory.

<!--pytest.mark.skip-->
```bash
cd docs
pip install -e '.[docs]'
make clean && make html
make host   # open the output link in a browser.
```

See the [Makefile](/Makefile) for more information.


7\. When you are ready, submit a pull request into the streaming repository!
<!--pytest.mark.skip-->
```bash
git commit -m "cool feature"    # Add relevant commit message
git push origin cool-new-feature
```

Create a pull request to propose changes you've made to a fork of an upstream repository by following this [guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

## Configuring README Code Snippets

Streaming uses [pytest-codeblocks](https://github.com/nschloe/pytest-codeblocks) to test all example code snippets. The pytest-codeblocks repository explains how to annotate code snippets, which supports most `pytest` configurations. For example, if a test requires model training, the GPU mark (`<!--pytest.mark.skip-->`) should be applied.
