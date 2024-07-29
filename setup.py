# Copyright 2022-2024 MosaicML Streaming authors
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
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
]

install_requires = [
    'boto3>=1.21.45,<2',
    'Brotli>=1.0.9',
    'google-cloud-storage>=2.9.0,<2.11.0',
    'matplotlib>=3.5.2,<4',
    'paramiko>=2.11.0,<4',
    'python-snappy>=0.6.1,<1',
    'torch>=1.10,<3',
    'torchvision>=0.10',
    'tqdm>=4.64.0,<5',
    'transformers>=4.21.3,<5',
    'xxhash>=3.0.0,<4',
    'zstd>=1.5.2.5,<2',
    'oci>=2.88,<3',
    'azure-storage-blob>=12.0.0,<13',
    'azure-storage-file-datalake>=12.11.0,<13',
    'azure-identity>=1.13.0',
]

extra_deps = {}

extra_deps['dev'] = [
    'datasets>=2.4.0,<3',
    'pyarrow>14.0.0',
    'docformatter>=1.4',
    'jupyter==1.0.0',
    'pre-commit>=2.18.1,<4',
    'pytest==8.3.2',
    'pytest_codeblocks==0.17.0',
    'pytest-cov>=4,<6',
    'toml==0.10.2',
    'yamllint==1.35.1',
    'moto>=4.0,<6',
    'fastapi==0.111.1',
    'pydantic==2.8.2',
    'uvicorn==0.30.3',
    'pytest-split==0.9.0',
]

extra_deps['docs'] = [
    'GitPython==3.1.42',
    'docutils==0.17.1',
    'furo==2022.9.29',
    'myst-parser==0.16.1',
    'nbsphinx==0.9.1',
    'pandoc==2.3',
    'pypandoc==1.13',
    'sphinx-argparse==0.4.0',
    'sphinx-copybutton==0.5.2',
    'sphinx==4.4.0',
    'sphinx-tabs==3.4.5',
    'sphinxcontrib.katex==0.9.6',
    'sphinxcontrib-applehelp==1.0.0',
    'sphinxcontrib-devhelp==1.0.0',
    'sphinxcontrib-htmlhelp==2.0.0',
    'sphinxcontrib-qthelp==1.0.0',
    'sphinxcontrib-serializinghtml==1.1.5',
]

extra_deps['simulator'] = [
    'sortedcollections>=2.1.0,<3',
    'streamlit>=1.26.0,<2',
    'altair>=5.1.1,<6',
    'omegaconf>=2.3.0,<3',
    'PyYAML>=6.0,<7',
    'pandas>=2.0.3,<3',
    'wandb>=0.15.5,<1',
    'humanize>=4.7.0,<5',
]

extra_deps['spark'] = [
    'pyspark>=3,<4',
]

extra_deps['databricks'] = [
    'databricks-sdk==0.29.0',
]

extra_deps['alipan'] = [
    'AliPCS-Py>=0.8,<1',
]

extra_deps['hf'] = [
    'huggingface_hub>=0.23.4,<0.24',
]

extra_deps['testing'] = [
    'mosaicml-cli>=0.5.25,<0.7',
]

extra_deps['all'] = sorted({dep for deps in extra_deps.values() for dep in deps})

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
    include_package_data=True,
    package_data={
        'streaming': ['py.typed'],
    },
    packages=setuptools.find_packages(exclude=['tests*']),
    entry_points={
        'console_scripts': ['simulator = simulation.launcher:launch_simulation_ui',],
    },
    classifiers=classifiers,
    install_requires=install_requires,
    extras_require=extra_deps,
    python_requires='>=3.9',
)
