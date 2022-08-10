# Copyright 2022 MosaicML. All Rights Reserved.

import os

import setuptools
from setuptools import setup


classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
]

install_requires = [
    "boto3==1.24.37",
    "Brotli==1.0.9",
    "matplotlib==3.5.2",
    "paramiko==2.11.0",
    "python-snappy==0.6.1",
    "torch>=1.10,<2",
    "torchtext>=0.10",
    "torchvision>=0.10",
    "tqdm==4.64.0",
    "xxhash==3.0.0",
    "zstd==1.5.2.5",
]

extra_deps = {
    'dev': [
        "docformatter==1.4",
        "isort==5.10.1",
        "pytest==7.1.2",
        "toml==0.10.2",
        "yamllint==1.26.3",
        "yapf==0.32.0",
    ]
}

setup(
    name="streaming",
    version="0.1.0",
    author="MosaicML",
    author_email="team@mosaicml.com",
    description='Streaming datasets',
    url="https://github.com/mosaicml/streaming/",
    packages=setuptools.find_packages(exclude=["tests*"]),
    classifiers=classifiers,
    install_requires=install_requires,
    extras_require=extra_deps,
    python_requires=">=3.7",
)
