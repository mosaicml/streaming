# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Streamlit widgets for simulation web UI."""

import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import altair as alt
from streaming.base.util import bytes_to_int
from core.sim_time import TimeUnit, ensure_time
from core.utils import get_simulation_stats
from numpy.typing import NDArray
import streamlit as st
import pandas as pd
from typing import Optional

def get_line_chart(data, throughput_window, throughput=True):
        hover = alt.selection_point(
            fields=["step"],
            nearest=True,
            on="mouseover",
            empty=False,
        )

        lines = (
            alt.Chart(data, title="Throughput (" + str(throughput_window) + "-step rolling average)")
            .mark_line()
            .encode(
                x="step",
                y="throughput (batches/s)",
            )
        ) if throughput else (
            alt.Chart(data, title="Cumulative Network Usage, all nodes")
            .mark_line()
            .encode(
                x="step",
                y="cumulative network usage (bytes)"
            )
        )

        # Draw points on the line, and highlight based on selection
        points = lines.transform_filter(hover).mark_circle(size=65)

        # Draw a rule at the location of the selection
        tooltips = (
            alt.Chart(data)
            .mark_rule()
            .encode(
                x="step",
                y="throughput (batches/s)" if throughput else "cumulative network usage (bytes)",
                opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
                tooltip=[
                    alt.Tooltip("step", title="Step"),
                    alt.Tooltip("throughput (batches/s)" if throughput else "cumulative network usage (bytes)", title="Throughput" if throughput else "Network Usage"),
                ],
            )
            .add_params(hover)
        )
        return (lines + points + tooltips).interactive()

def stream_entry(col, streams, key, add_stream: bool = True, defaults: dict = None):
    stream_entries = {}
    col.write(f"*Stream {key+1}*")
    on = col.toggle("use `index.json`", key=str(key)+"toggle") if add_stream else None
    if on or not add_stream:
        path = col.text_input("path to `index.json`", 
                              value="/absolute/path/to/index.json" 
                              if defaults is None else defaults["path"], 
                              help="path to the `index.json` file for this stream. \
                                the `index.json` file contains information about the shards in \
                                your dataset.", key=str(key)+"path", disabled=(not add_stream))
        if add_stream:
            path_type = col.selectbox('path type', ["local", "remote"], key=str(key)+"path_type")
            stream_entries["path_type"] = path_type
        stream_entries["path"] = path
    else:
        shards = col.number_input('number of shards', step=1, value=20850, help="number of total \
                                  shards across your whole dataset.", key=str(key)+"shards")
        samples_per_shard = col.number_input('samples per shard', step=1, value=4093,
                                              help="average number of samples contained \
                                                in each shard.", key=str(key)+"samples")
        avg_raw_shard_size = col.text_input('avg raw shard size (bytes)', value="67MB", 
                                        help="average raw size, in bytes, \
                                            of a single shard.", key=str(key)+"rawsize")
        avg_raw_shard_size = bytes_to_int(avg_raw_shard_size)
        avg_zip_shard_size = col.text_input('avg compressed shard size (bytes)', value="None", 
                                        help="average compressed size, in bytes, \
                                            of a single shard.", key=str(key)+"zipsize")
        avg_zip_shard_size = None if avg_zip_shard_size == "None" \
            else bytes_to_int(avg_zip_shard_size)
        stream_entries["shards"] = shards
        stream_entries["samples_per_shard"] = samples_per_shard
        stream_entries["avg_raw_shard_size"] = avg_raw_shard_size
        stream_entries["avg_zip_shard_size"] = avg_zip_shard_size
    proportion = col.text_input('proportion',
                                  value="None" if defaults is None else defaults["proportion"], 
                                  help="proportion of the full training dataset that this stream \
                                    represents.", key=str(key)+"proportion",
                                    disabled=(not add_stream))
    proportion = float(proportion) if proportion != "None" else None
    repeat = col.text_input('repeat',
                              value="None" if defaults is None else defaults["repeat"], 
                              help="number of times to repeat the samples in this \
                                stream.", key=str(key)+"repeat",
                                disabled=(not add_stream))
    repeat = float(repeat) if repeat != "None" else None
    choose = col.text_input('choose',
                              value="None" if defaults is None else defaults["choose"], 
                              help="number of samples to choose from this \
                                stream.", key=str(key)+"choose",
                                disabled=(not add_stream))
    choose = int(choose) if choose != "None" else None
    stream_entries["proportion"] = proportion
    stream_entries["repeat"] = repeat
    stream_entries["choose"] = choose
    
    streams[key] = stream_entries
    if add_stream and col.checkbox(label="add stream", key=str(key)+"checkbox"):
        stream_entry(col, streams, key+1)

def param_inputs(col, input_params: dict, defaults: dict = {}):
    """Define parameter input area."""
    col3, col4, col5 = col.columns(3)

    # dataset
    streams = {}
    col3.write("**Dataset Parameters**")
    if "streams" in defaults:
        key = 0
        for _, stream in defaults["streams"].items():
            # Case is only possible when reading in streams from yaml file. Stream will have path.
            stream_entry(col3, streams, key, add_stream=False, defaults=stream)
            key += 1
        streams = defaults["streams"]
    else:
        stream_entry(col3, streams, 0, add_stream=True)
    col3.text("")
    input_params["streams"] = streams

    # training
    col4.write("**Training Parameters**")
    if "max_duration" in defaults:
        default_max_duration = defaults["max_duration"]
        default_value = int(default_max_duration.value)
        default_unit_index = 0 if default_max_duration.unit == TimeUnit.BATCH else 1
        time_value = col4.number_input('training duration', step=1,
                                       value = default_value,
                                       help="training duration value, in specified units.")
        time_units = col4.selectbox('units', ["batches", "epochs"],
                                       index = default_unit_index,
                                       help="units of training duration.")
    else:
        time_value = col4.number_input('training duration', step=1,
                                       value=1000,
                                       help="training duration value, in specified units.")
        time_units = col4.selectbox('units', ["batches", "epochs"],
                                       help="units of training duration.")
    # Get Time object from inputs
    time_string = str(time_value)
    time_string += "ba" if time_units == "batches" else "ep"
    max_duration = ensure_time(time_string, TimeUnit.EPOCH)
    epoch_size = col4.text_input('epoch size (samples)', value="",
                                 help="epoch size for this run, in samples.")
    epoch_size = None if epoch_size == "" or epoch_size == "None" else int(epoch_size)
    device_batch_size = col4.number_input('device batch size', step=1,
                                          value=16 if "device_batch_size" not in defaults
                                            else defaults["device_batch_size"],
                                          help="number of samples per device (GPU) per batch. \
                                            the global batch size is `device_batch_size * \
                                            devices_per_node * physical_nodes`")
    col4.text("")
    input_params["max_duration"] = max_duration
    input_params["epoch_size"] = epoch_size
    input_params["device_batch_size"] = device_batch_size

    # hardware and network
    col4.write("**Hardware and Network Parameters**")
    physical_nodes = col4.number_input('number of physical nodes', step=1,
                                       value=1 if "physical_nodes" not in defaults
                                            else defaults["physical_nodes"],
                                       help="number of physical nodes for this run. \
                                        a node typically consists of 8 devices (GPUs).")
    devices = col4.number_input('devices per node', step=1,
                                value=8 if "devices" not in defaults else defaults["devices"],
                                help="number of devices (GPUs) per node for this run. \
                                    there are typically 8 devices per node.")
    time_per_sample = col4.number_input('process time per sample (s)', step = 0.0005,
                                        value=0.0175 if "time_per_sample" not in defaults
                                            else defaults["time_per_sample"],
                                         format="%.4f", help="time for one device to process one \
                                        sample from your dataset.")
    node_network_bandwidth = col4.text_input('network bandwidth per node (bytes/s)', 
                                            value="500MB" if "node_network_bandwidth" not in defaults
                                                else defaults["node_network_bandwidth"], 
                                            help="network bandwidth available to \
                                            each node. in practice, network bandwidth is \
                                            variable and is affected by many factors, \
                                            including cluster demand.")
    col4.text("")
    input_params["physical_nodes"] = physical_nodes
    input_params["devices"] = devices
    input_params["time_per_sample"] = time_per_sample
    input_params["node_network_bandwidth"] = node_network_bandwidth

    # streaming
    col5.write("**Streaming Parameters**")
    workers = col5.number_input('workers per device', step=1,
                                value=8 if "workers" not in defaults else defaults["workers"],
                                help="number of dataloader \workers per device (GPU).")
    canonical_nodes = col5.number_input('number of canonical nodes', step=1,
                                        value=2 if "canonical_nodes" not in defaults
                                            else defaults["canonical_nodes"],
                                        help="number of canonical nodes to split your dataset \
                                            into. a canonical node is a bucket of shards that is \
                                            assigned to a particular physical node.")
    predownload = col5.text_input('predownload per worker (samples)',
                                  value="None" if "predownload" not in defaults
                                        else defaults["predownload"],
                                  help="number of samples ahead each worker should download. \
                                    predownload does not occur before the first batch; \
                                    rather, it occurs while training is ongoing.")
    predownload = None if predownload == "" or predownload == "None" else int(predownload)
    shuffle = col5.checkbox(label="shuffle", value=True if "shuffle" not in defaults
                                            else defaults["shuffle"],
                            help="whether or not to shuffle the samples for this run.")
    shuffle_algo="py1e" if defaults is None or "shuffle_algo" not in defaults \
        else defaults["shuffle_algo"]
    shuffle_block_size="1M" if defaults is None or "shuffle_block_size" not in defaults \
        else defaults["shuffle_block_size"]
    seed=42 if defaults is None or "seed" not in defaults else defaults["seed"]
    if shuffle:
        algos = ["py1e", "py1br", "py1b", "py1s", "py2s", "naive"]
        default_index = 0
        if "shuffle_algo" in defaults:
            default_index = algos.index(defaults["shuffle_algo"])
        shuffle_algo = col5.selectbox('shuffling algorithm', algos, index=default_index, 
                                    help="shuffling algorithm to use for this run. your shuffle \
                                        parameters may affect model training.")
        shuffle_block_size = col5.text_input('shuffle block size (samples)',
                                            value="2M" if "shuffle_block_size" not in defaults
                                                else defaults["shuffle_block_size"], 
                                            help="shuffle block size for this run. \
                                                used in the `py1b`, `py1br`, and `py1e` \
                                                shuffling algorithms, samples in blocks of \
                                                `shuffle_block_size` are randomly shuffled \
                                                inside each bucket of shards (aka canonical node).")
        seed = col5.number_input('shuffle seed', step=1,
                                value=42 if "seed" not in defaults else defaults["seed"],
                                help="random seed for shuffling.")
    cache_limit = col5.text_input('cache limit (bytes)',
                                  value="None" if "cache_limit" not in defaults
                                        else defaults["cache_limit"], 
                                  help="cache limit per node for this run. \
                                    setting cache limit too low will impact throughput.")
    cache_limit = None if cache_limit=="" or cache_limit=="None" else bytes_to_int(cache_limit)
    sampling_methods = ["balanced", "fixed"]
    sampling_method = col5.selectbox('sampling method', sampling_methods,
                                      index=0 if "sampling_method" not in defaults
                                        else sampling_methods.index(defaults["sampling_method"]),
                                      help="sampling method for this run. controls how samples are\
                                        chosen each epoch. can be either 'balanced' or 'fixed'.")
    sampling_granularity = col5.number_input('sampling granularity', step=1,
                                      value=1 if "sampling_granularity" not in defaults
                                        else defaults["sampling_granularity"],
                                      help="sampling granularity for this run. controls how\
                                        samples are balanced across shards. higher values will\
                                        cause more samples to be drawn from each shard at a time.")
    batching_methods = ["random", "per_stream", "stratified"]
    batching_method = col5.selectbox('batching method', batching_methods,
                                      index=0 if "batching_method" not in defaults
                                        else batching_methods.index(defaults["batching_method"]),
                                      help="batching method for this run. controls how batches\
                                        are constructed.")
    col5.text("")
    input_params["workers"] = workers
    input_params["canonical_nodes"] = canonical_nodes
    input_params["predownload"] = predownload
    input_params["cache_limit"] = cache_limit
    input_params["shuffle"] = shuffle
    input_params["shuffle_algo"] = shuffle_algo
    input_params["shuffle_block_size"] = shuffle_block_size
    input_params["seed"] = seed
    input_params["sampling_method"] = sampling_method
    input_params["sampling_granularity"] = sampling_granularity
    input_params["batching_method"] = batching_method

def display_simulation_stats(component, total_batches: int, step_times: NDArray, 
                             time_per_sample: float, device_batch_size: int, 
                             time_to_first_batch: float, min_cache_limit: int,
                             cache_limit: Optional[int]):
    all_throughput_drops, warmup_time, warmup_step, post_warmup_throughput_drops = \
        get_simulation_stats(step_times, time_per_sample, device_batch_size)
    with component.container():
        st.write(f"Minimum cache limit needed: **{min_cache_limit:,} bytes**")
        if cache_limit is not None and cache_limit < min_cache_limit:
            # Cache limit is too low, and will cause shard redownloads / throughput drops.
            st.warning('The provided cache limit is lower than the minimum cache limit needed to \
                     prevent shard re-downloads. This can cause throughput issues.',
                     icon="⚠️")
        if warmup_step == total_batches:
            # display error if the warmup phase is the whole run, 
            # meaning that we never hit peak throughput.
            st.error('This configuration is severely bottlenecked by downloading. \
                     The run will not be performant.', icon="🚨")
        elif post_warmup_throughput_drops:
            # display warning if post-warmup throughput drops are more than 10% of the run.
            st.warning('This configuration experiences some downloading-related slowdowns \
                       even after warmup.', icon="⚠️")
        st.write("**{0} steps**, or **{1:.1f}%** of all steps, waited for \
                 shard downloads.".format(all_throughput_drops, 
                                          100*all_throughput_drops/(total_batches)))
        if warmup_step != total_batches:
            # only display post-warmup throughput drop info if we actually ended the warmup period
            # (i.e. we hit peak throughput at some point)
            st.write("There were **{} steps** that waited for shard downloads after the warmup \
                     period.".format(post_warmup_throughput_drops))
        st.write("Estimated time to first batch: **{0:.2f} s**".format(time_to_first_batch))
        st.write("Estimated warmup time: **{0:.2f} s**".format(warmup_time))

def get_shuffle_quality_chart(data):
    bars = (
        alt.Chart(data, title="Shuffle Quality")
        .mark_bar()
        .encode(
            x="algo",
            y="quality",
            tooltip="quality"
        )
        .properties(
            width=550,
        )
    )

    return bars.interactive()


def display_shuffle_quality_graph(futures, component):
    # Retrieve shuffle quality result since it is available
    shuffle_algos_qualities = list(zip(*[f.result() for f in futures]))
    shuffle_algos = list(shuffle_algos_qualities[0])
    shuffle_qualities = list(shuffle_algos_qualities[1])
    shuffle_quality_df = pd.DataFrame({"algo": shuffle_algos,
                                "quality": shuffle_qualities})
    component.altair_chart(get_shuffle_quality_chart(shuffle_quality_df),
                                use_container_width=True)