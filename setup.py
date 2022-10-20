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

classifiers = [
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
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
]

extra_deps = {}

extra_deps['dev'] = [
    'docformatter>=1.4',
    'jupyter==1.0.0',
    'pre-commit>=2.18.1,<3',
    'pytest==7.1.3',
    'pytest_codeblocks==0.16.1',
    'pytest-xdist>=2',
    'toml==0.10.2',
    'yamllint==1.28.0',
]

extra_deps['docs'] = [
    'GitPython==3.1.29',
    'docutils==0.17.1',
    'furo==2022.9.29',
    'myst-parser==0.18.1',
    'nbsphinx==0.8.9',
    'pandoc==2.2',
    'pypandoc==1.9',
    'sphinx-argparse==0.3.2',
    'sphinx-copybutton==0.5.0',
    'sphinx==4.4.0',
    'sphinx_panels==0.6.0',
    'sphinxcontrib-images==0.9.4',
    'sphinxcontrib.katex==0.9.0',
    'sphinxemoji==0.2.0',
    'sphinxext.opengraph==0.6.3',
]

extra_deps['all'] = set(dep for deps in extra_deps.values() for dep in deps)

package_name = os.environ.get('MOSAIC_PACKAGE_NAME', 'mosaicml-streaming')

if package_name != 'mosaicml-streaming':
    print(f'Building mosaicml-streaming as {package_name}')

# Use repo README for PyPi description
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

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
