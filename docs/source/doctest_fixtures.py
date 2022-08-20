# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# disabling general type issues because of monkeypatching
# pyright: reportGeneralTypeIssues=none

"""Fixtures available in doctests.

The script is run before any doctests are executed,
so all imports and variables are available in any doctest.
The output of this setup script does not show up in the documentation.
"""
import os
import sys
import tempfile

# Need to insert the repo root at the beginning of the path, since there may be other modules named `tests`
# Assuming that docs generation is running from the `docs` directory
_docs_dir = os.path.abspath('.')
_repo_root = os.path.dirname(_docs_dir)
if sys.path[0] != _repo_root:
    sys.path.insert(0, _repo_root)

# Change the cwd to be the tempfile, so we don't pollute the documentation source folder
tmpdir = tempfile.mkdtemp()
cwd = os.path.abspath('.')
os.chdir(tmpdir)
