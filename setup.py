# Copyright 2022 MosaicML. All Rights Reserved.

import os

import setuptools
from setuptools import setup

classifiers = [
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
]

install_requires = [
    'boto3==1.24.37',
    'Brotli==1.0.9',
    'datasets==2.4.0',
    'matplotlib==3.5.2',
    'paramiko==2.11.0',
    'python-snappy==0.6.1',
    'torch>=1.10,<2',
    'torchtext>=0.10',
    'torchvision>=0.10',
    'tqdm==4.64.0',
    'transformers==4.21.3',
    'xxhash==3.0.0',
    'zstd==1.5.2.5',
]

extra_deps = {}

extra_deps['dev'] = [
    'docformatter==1.4',
    'pytest==7.1.2',
    'toml==0.10.2',
    'yamllint==1.26.3',
    'pre-commit>=2.18.1,<3',
]

extra_deps['docs'] = [
    'GitPython==3.1.27',
    'docutils==0.17.1',
    'furo==2022.3.4',
    'myst-parser==0.16.1',
    'nbsphinx==0.8.8',
    'pandoc==2.2',
    'pypandoc==1.8.1',
    'sphinx-argparse==0.3.1',
    'sphinx-copybutton==0.5.0',
    'sphinx==4.4.0',
    'sphinx_panels==0.6.0',
    'sphinxcontrib-images==0.9.4',
    'sphinxcontrib.katex==0.8.6',
    'sphinxemoji==0.2.0',
    'sphinxext.opengraph==0.6.1',
]

extra_deps['all'] = set(dep for deps in extra_deps.values() for dep in deps)

setup(
    name='streaming',
    version='0.1.0',
    author='MosaicML',
    author_email='team@mosaicml.com',
    description='Streaming datasets',
    url='https://github.com/mosaicml/streaming/',
    packages=setuptools.find_packages(exclude=['tests*']),
    classifiers=classifiers,
    install_requires=install_requires,
    extras_require=extra_deps,
    python_requires='>=3.7',
)
