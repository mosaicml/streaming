# simulator ui using streamlit

import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
from simulation_funcs import simulate
from streaming.base.util import number_abbrev_to_int, bytes_to_int


# set up page
st.set_page_config(layout="wide") 
col1, space, col2 = st.columns((8, 1, 8))
col2.title("Streaming Simulator")
col2.write("Enter run parameters in the left panel.")
col2.text("")
progress_bar = col1.progress(0)
status_text = col1.empty()
col1.text("")
throughput_plot = col2.empty()
network_plot = col2.empty()
throughput_window = 10

def get_chart(data, throughput=True):
        hover = alt.selection_single(
            fields=["step"],
            nearest=True,
            on="mouseover",
            empty="none",
        )

        lines = (
            alt.Chart(data, title="Throughput" if throughput else "Network Usage")
            .mark_line()
            .encode(
                x="step",
                y="throughput (batches/s)" if throughput else "cumulative network usage (bytes)"
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
            .add_selection(hover)
        )
        return (lines + points + tooltips).interactive()

def submit_simulation(shards, samples_per_shard, avg_shard_size, epochs, batches_per_epoch, device_batch_size,
                      workers, canonical_nodes, predownload, shuffle_algo, cache_limit, shuffle_block_size,
                      seed, physical_nodes, devices, time_per_sample, node_network_bandwidth):
    gen_sim = simulate(shards, samples_per_shard, avg_shard_size,
                                        device_batch_size, time_per_sample, batches_per_epoch,
                                        epochs, physical_nodes, devices, node_network_bandwidth,
                                        workers, canonical_nodes, predownload, cache_limit, 
                                        shuffle_algo, shuffle_block_size, seed, True)

    gen_step_times = []
    gen_shard_downloads = []
    throughput_data = []
    network_data = []
    throughput_steps = []
    network_steps = []
    new_throughput_data = []
    new_network_data = []
    for i, (step_time, shard_download) in enumerate(gen_sim):
        if step_time is not None and shard_download is not None:
            gen_step_times.append(step_time)
            gen_shard_downloads.append(shard_download)
            # plot throughput once we have enough samples for the window 
            if i >= throughput_window - 1:
                step_time_window = np.array(gen_step_times[-throughput_window:])
                throughput = 1/np.mean((step_time_window))
                throughput_steps.append(i+1)
                throughput_data.append(throughput)
                new_throughput_data.append(throughput)
            # plot network usage
            cumulative_shard_download = np.sum(np.array(gen_shard_downloads))
            network_steps.append(i+1)
            network_data.append(cumulative_shard_download)
            new_network_data.append(cumulative_shard_download)
        
        # update plots and percentages once every 500 batches
        if i == 1 or i % 500 == 0 or i == batches_per_epoch * epochs - 1:
            throughput_df = pd.DataFrame({"step": throughput_steps, "throughput (batches/s)": throughput_data})
            network_df = pd.DataFrame({"step": network_steps, "cumulative network usage (bytes)": network_data})
            throughput_plot.altair_chart(get_chart(throughput_df, True), use_container_width=True)
            network_plot.altair_chart(get_chart(network_df, False), use_container_width=True)
            # update progress bar and text
            percentage = int(100*(i+1) / (batches_per_epoch * epochs))
            status_text.text("%i%% Complete" % percentage)
            progress_bar.progress(percentage)

with col1.form("my_form"):

    submitted = st.form_submit_button("Simulate Run", use_container_width=True)
    st.text("")

    col3, col4 = st.columns(2)

    # dataset
    col3.write("**Dataset Parameters**")
    shards = col3.number_input('number of shards', step=1, value=20850, help="number of total shards across your whole dataset.")
    samples_per_shard = col3.number_input('samples per shard', step=1, value=4093, help="average number of samples contained in each shard. the `index.json` file can help estimate this.")
    avg_shard_size = col3.text_input('average shard size (bytes)', value="67MB", help="average size, in bytes, of a single shard. the `index.json` file can help estimate this.")
    col3.text("")

    # training
    col4.write("**Training Parameters**")
    epochs = col4.number_input('number of epochs', step=1, value=1, help="number of epochs for this run.")
    batches_per_epoch = col4.text_input('batches per epoch', value="3k", help="number of batches per epoch for this run.")
    batches_per_epoch = number_abbrev_to_int(batches_per_epoch)
    device_batch_size = col4.number_input('device batch size', step=1, value=16, help="number of samples per device (GPU) per batch. the global batch size is `device_batch_size * devices_per_node * physical_nodes`")
    col4.text("")

    # hardware and network
    col3.write("**Hardware and Network Parameters**")
    physical_nodes = col3.number_input('number of physical nodes', step=1, value=2, help="number of physical nodes for this run. a node typically consists of 8 devices (GPUs).")
    devices = col3.number_input('devices per node', step=1, value=8, help="number of devices (GPUs) per node for this run. there are typically 8 devices per node.")
    time_per_sample = col3.number_input('process time per sample (s)', step = 0.0005, value=0.0175, format="%.4f", help="time for one device to process one sample from your dataset.")
    node_network_bandwidth = col3.text_input('network bandwidth per node (bytes/s)', value="2GB", help="network bandwidth available to each node. in practice, network bandwidth is variable and is affected by many factors, including cluster demand.")
    col3.text("")

    # streaming
    col4.write("**Streaming Parameters**")
    workers = col4.number_input('workers per device', step=1, value=8, help="number of dataloader workers per device (GPU).")
    canonical_nodes = col4.number_input('number of canonical nodes', step=1, value=8, help="number of canonical nodes to split your dataset into. a canonical node is a bucket of shards that is assigned to a particular physical node.")
    predownload = col4.text_input('predownload per worker (samples)', value=64, help="number of samples ahead each worker should download. predownload does not occur before the first batch; rather, it occurs while training is ongoing.")
    #shuffle_algo = col4.text_input('shuffling algorithm', value="py1b", help="shuffling algorithm to use for this run. your shuffle parameters may affect model training.")
    shuffle_algo = col4.selectbox('shuffling algorithm', ["py1b", "py1br", "py1e", "py1s", "py2s", "naive", "None"], help="shuffling algorithm to use for this run. your shuffle parameters may affect model training.")
    if shuffle_algo == "None":
        shuffle_algo = None
    cache_limit = col4.text_input('cache limit (bytes)', value="None", help="cache limit per node for this run. setting cache limit too low will impact throughput.")
    if cache_limit == "None":
        cache_limit = None
    else:
        cache_limit = bytes_to_int(cache_limit)
    shuffle_block_size = col4.text_input('shuffle block size (samples)', value="16M", help="shuffle block size for this run. used in the `py1b`, `py1br`, and `py1e` shuffling algorithms, samples in blocks of `shuffle_block_size` are randomly shuffled inside each bucket of shards (aka canonical node).")
    seed = col4.number_input('random seed', step=1, value=42, help="random seed for shuffling.")
    col4.text("")

    if submitted:
        submit_simulation(shards, samples_per_shard, avg_shard_size, epochs, batches_per_epoch, device_batch_size,
                        workers, canonical_nodes, predownload, shuffle_algo, cache_limit, shuffle_block_size,
                        seed, physical_nodes, devices, time_per_sample, node_network_bandwidth)

    