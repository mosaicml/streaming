# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Launch simulator UI from command line when this package is installed."""

import os
import subprocess


def launch_simulation_ui():
    """Launch the simulation UI."""
    subprocess.run(['streamlit', 'run', os.path.abspath('simulation/interfaces/sim_ui.py')])
