# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming package setup."""

import os

import setuptools
from setuptools import setup

# Read the streaming version
# Cannot import from `streaming.__version__` since that will not be available when building or installing the package
with open(os.path.join(os.path.dirname(__file__), 'streaming', '_version.py')) as f:
    version_globals = {}
    version_locals = {}
    exec(f.read(), version_globals, version_locals)
    streaming_version = version_locals['__version__']

# Use repo README for PyPi description
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

# Hide the content between <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN --> and
# <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END --> tags in the README
while True:
    start_tag = '<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN -->'
    end_tag = '<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END -->'
    start = long_description.find(start_tag)
    end = long_description.find(end_tag)
    if start == -1:
        assert end == -1, 'there should be a balanced number of start and ends'
        break
    else:
        assert end != -1, 'there should be a balanced number of start and ends'
        long_description = long_description[:start] + long_description[end + len(end_tag):]

classifiers = [
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
]

install_requires = [
    'boto3>=1.21.45,<2',
    'Brotli>=1.0.9',
    'datasets>=2.4.0,<3',
    'matplotlib>=3.5.2,<4',
    'paramiko>=2.11.0,<3',
    'python-snappy>=0.6.1,<1',
    'torch>=1.10,<2',
    'torchtext>=0.10',
    'torchvision>=0.10',
    'tqdm>=4.64.0,<5',
    'transformers>=4.21.3,<5',
    'xxhash>=3.0.0,<4',
    'zstd>=1.5.2.5,<2',
    'oci>=2.88,<3',
]

extra_deps = {}

extra_deps['dev'] = [
    'docformatter>=1.4',
    'jupyter==1.0.0',
    'pre-commit>=2.18.1,<3',
    'pytest==7.2.0',
    'pytest_codeblocks==0.16.1',
    'pytest-cov>=4,<5',
    'toml==0.10.2',
    'yamllint==1.28.0',
    'moto>=4.0,<5',
]

extra_deps['docs'] = [
    'GitPython==3.1.29',
    'docutils==0.17.1',
    'furo==2022.9.29',
    'myst-parser==0.18.1',
    'nbsphinx==0.8.10',
    'pandoc==2.3',
    'pypandoc==1.10',
    'sphinx-argparse==0.4.0',
    'sphinx-copybutton==0.5.1',
    'sphinx==4.4.0',
    'sphinx_panels==0.6.0',
    'sphinxcontrib-images==0.9.4',
    'sphinxcontrib.katex==0.9.3',
    'sphinxemoji==0.2.0',
    'sphinxext.opengraph==0.7.3',
]

extra_deps['all'] = set(dep for deps in extra_deps.values() for dep in deps)

package_name = os.environ.get('MOSAIC_PACKAGE_NAME', 'mosaicml-streaming')

if package_name != 'mosaicml-streaming':
    print(f'Building mosaicml-streaming as {package_name}')

setup(
    name=package_name,
    version=streaming_version,
    author='MosaicML',
    author_email='team@mosaicml.com',
    description=
    'Streaming lets users create PyTorch compatible datasets that can be streamed from cloud-based object stores',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mosaicml/streaming/',
    packages=setuptools.find_packages(exclude=['tests*']),
    classifiers=classifiers,
    install_requires=install_requires,
    extras_require=extra_deps,
    python_requires='>=3.7',
)
