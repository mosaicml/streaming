# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Launch simulator UI from command line when this package is installed."""

import os
import subprocess


def launch_simulation_ui():
    """Launch the simulation UI."""
    absolute_cwd_path = os.path.dirname(os.path.abspath(__file__))
    absolute_sim_path = os.path.join(absolute_cwd_path, 'interfaces', 'sim_ui.py')
    subprocess.run(['streamlit', 'run', absolute_sim_path])
