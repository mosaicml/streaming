# Copyright 2022 MosaicML. All Rights Reserved.

import os

import setuptools
from setuptools import setup


def package_files(prefix: str, directory: str, extension: str):
    # from https://stackoverflow.com/a/36693250
    paths = []
    for (path, _, filenames) in os.walk(os.path.join(prefix, directory)):
        for filename in filenames:
            if filename.endswith(extension):
                paths.append(os.path.relpath(os.path.join(path, filename), prefix))
    return paths


data_files = package_files("mcontrib", "algorithms", ".json")

install_requires = [
    "boto3==1.24.37",
    "Brotli==1.0.9",
    "matplotlib==3.5.2",
    "paramiko==2.11.0",
    "python-snappy==0.6.1",
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
    version="0.0.0",
    author="MosaicML",
    author_email="team@mosaicml.com",
    description='Streaming datasets',
    url="https://github.com/mosaicml/streaming/",
    include_package_data=True,
    package_data={
        "mcontrib": data_files,
    },
    packages=setuptools.find_packages(exclude=["tests*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=install_requires,
    extras_require=extra_deps,
    python_requires=">=3.7",
)
