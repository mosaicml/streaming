# Contributing to Streaming

Thanks for considering contributing to Streaming!

Issues tagged with [good first issue](https://github.com/mosaicml/streaming/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) are great options to start contributing.

If you have questions, join us on [Slack](https://mosaicml.me/slack) -- we'll be happy to help you!

We welcome contributions for bug fixes, new features you'd like to contribute to the community, or improve test suite!


## Prerequisites

To set up the development environment in your local box, run the commands below.

1\. Install the dependencies needed for testing and linting the code:

<!--pytest.mark.skip-->
```bash
pip install -e '.[dev]'
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
git checkout -b cool-new-feature
```

4\. When you are ready, submit a pull request into the streaming repository! If merged, we'll reach out to send you some free swag :)

## Configuring README Code Snippets

Streaming uses [pytest-codeblocks](https://github.com/nschloe/pytest-codeblocks) to test all example code snippets. The pytest-codeblocks repository explains how to annotate code snippets, which supports most `pytest` configurations. For example, if a test requires model training, the GPU mark (`<!--pytest.mark.skip-->`) should be applied.

## Running Tests

To test your changes locally, run:

1. `pytest .`  # run all the unittests
1. `cd docs && make doctest`  # run doctests

See the [Makefile](/Makefile) for more information.

If you want to run pre-commit hooks manually, which check for code formatting and type annotations, run `pre-commit run --all-files`
