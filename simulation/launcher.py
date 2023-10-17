# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Launch simulator UI from command line when this package is installed."""

import subprocess
import os

def launch_simulation_ui():
    subprocess.run(["streamlit", "run", os.path.abspath("simulation/interfaces/sim_ui.py")])