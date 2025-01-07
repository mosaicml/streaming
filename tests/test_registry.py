# Copyright 2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
import pathlib
from importlib.metadata import EntryPoint
from typing import Any, Callable, Union

import catalogue
import pytest

from streaming.base import registry_utils
from streaming.base.stream import Stream, streams_registry


def test_streams_registry_setup():
    assert isinstance(streams_registry, registry_utils.TypedRegistry)
    assert streams_registry.namespace == ('streaming', 'streams_registry')

    stream = streams_registry.get('stream')
    assert stream == Stream


# The tests below are adapted with minimal changes from llm-foundry
# to guarantee registry_utils works as expected


def test_registry_create(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(catalogue, 'Registry', {})

    new_registry = registry_utils.create_registry(
        'streaming',
        'test_registry',
        generic_type=str,
        entry_points=False,
    )

    assert new_registry.namespace == ('streaming', 'test_registry')
    assert isinstance(new_registry, registry_utils.TypedRegistry)


def test_registry_typing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(catalogue, 'Registry', {})
    new_registry = registry_utils.create_registry(
        'streaming',
        'test_registry',
        generic_type=str,
        entry_points=False,
    )
    new_registry.register('test_name', func='test')

    # This would fail type checking without the type ignore
    # It is here to show that the TypedRegistry is working (gives a type error without the ignore),
    # although this would not catch a regression in this regard
    new_registry.register('test_name', func=1)  # type: ignore


def test_registry_add(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(catalogue, 'Registry', {})
    new_registry = registry_utils.create_registry(
        'streaming',
        'test_registry',
        generic_type=str,
        entry_points=False,
    )
    new_registry.register('test_name', func='test')

    assert new_registry.get('test_name') == 'test'


def test_registry_overwrite(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(catalogue, 'Registry', {})
    new_registry = registry_utils.create_registry(
        'streaming',
        'test_registry',
        generic_type=str,
        entry_points=False,
    )
    new_registry.register('test_name', func='test')
    new_registry.register('test_name', func='test2')

    assert new_registry.get('test_name') == 'test2'


def test_registry_init_code(tmp_path: pathlib.Path):
    register_code = """
from streaming.base.stream import Stream, streams_registry

@streams_registry.register('test_stream')
class TestStream(Stream):
    pass
"""

    with open(tmp_path / 'init_code.py', 'w') as _f:
        _f.write(register_code)

    registry_utils.import_file(tmp_path / 'init_code.py')

    assert issubclass(streams_registry.get('test_stream'), Stream)

    del catalogue.REGISTRY[('streaming', 'streams_registry', 'test_stream')]

    assert 'test_stream' not in streams_registry


def test_registry_entrypoint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(catalogue, 'Registry', {})

    monkeypatch.setattr(
        importlib.metadata,
        'entry_points',
        lambda: {
            'streaming_test_registry': [
                EntryPoint(
                    name='test_entry',
                    value='streaming.base.stream:Stream',
                    group='streaming_test_registry',
                ),
            ],
        },
    )

    monkeypatch.setattr(
        catalogue,
        'AVAILABLE_ENTRY_POINTS',
        importlib.metadata.entry_points(),
    )
    new_registry = registry_utils.create_registry(
        'streaming',
        'test_registry',
        generic_type=str,
        entry_points=True,
    )
    assert new_registry.get('test_entry') == Stream


def test_registry_builder(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(catalogue, 'Registry', {})

    new_registry = registry_utils.create_registry(
        'streaming',
        'test_registry',
        entry_points=False,
        generic_type=Union[type[Stream], Callable[..., Stream]],
    )

    class TestStream(Stream):

        def __init__(self):
            pass

    new_registry.register('test_stream', func=TestStream)

    # Valid, no validation
    valid_class = registry_utils.construct_from_registry(
        'test_stream',
        new_registry,
        pre_validation_function=TestStream,
    )
    assert isinstance(valid_class, TestStream)

    class NotStream:
        pass

    # Invalid, class validation
    with pytest.raises(
            ValueError,
            match='Expected test_stream to be of type',
    ):
        registry_utils.construct_from_registry(
            'test_stream',
            new_registry,
            pre_validation_function=NotStream,
        )

    # Invalid, function pre-validation
    with pytest.raises(ValueError, match='Invalid'):

        def pre_validation_function(x: Any):
            raise ValueError('Invalid')

        registry_utils.construct_from_registry(
            'test_stream',
            new_registry,
            pre_validation_function=pre_validation_function,
        )

    # Invalid, function post-validation
    with pytest.raises(ValueError, match='Invalid'):

        def post_validation_function(x: Any):
            raise ValueError('Invalid')

        registry_utils.construct_from_registry(
            'test_stream',
            new_registry,
            post_validation_function=post_validation_function,
        )

    # Invalid, not a class or function
    new_registry.register('non_callable', func=1)  # type: ignore
    with pytest.raises(
            ValueError,
            match='Expected non_callable to be a class or function',
    ):
        registry_utils.construct_from_registry('non_callable', new_registry)

    # Valid, partial function
    new_registry.register(
        'partial_func',
        func=lambda x, y: x * y,
    )  # type: ignore
    partial_func = registry_utils.construct_from_registry(
        'partial_func',
        new_registry,
        partial_function=True,
        kwargs={'x': 2},
    )
    assert partial_func(y=3) == 6

    # Valid, builder function
    new_registry.register('builder_func', func=lambda: TestStream())
    valid_built_class = registry_utils.construct_from_registry(
        'builder_func',
        new_registry,
        partial_function=False,
    )
    assert isinstance(valid_built_class, TestStream)


def test_registry_init_code_fails(tmp_path: pathlib.Path):
    register_code = """
asdf
"""

    with open(tmp_path / 'init_code.py', 'w') as _f:
        _f.write(register_code)

    with pytest.raises(RuntimeError, match='Error executing .*init_code.py'):
        registry_utils.import_file(tmp_path / 'init_code.py')


def test_registry_init_code_dne(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError, match='File .* does not exist'):
        registry_utils.import_file(tmp_path / 'init_code.py')
