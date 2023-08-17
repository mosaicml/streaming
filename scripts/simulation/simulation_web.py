# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Web app to simulate streaming with different input params.

Install:

    pip3 install fastapi pydantic uvicorn

Run:

    uvicorn scripts.simulation:simulation_web:app --port 2000 --reload
"""

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import base64
from simulation.simulation_funcs import simulate, plot_simulation

INDEX = '''
<!doctype html>
<html>
<head>
  <title>Streaming Simulator</title>
  <style type="text/css">
body {
    font-family: monospace;
    margin: 0px;
    margin-bottom: 400px;
}
table {
    padding: 2px;
}
td {
    padding: 2px;
    text-align: center;
}
.all {
    background: #91ca45;
    border-radius: 4px;
    margin: 10px;
    padding-right: 16px;
}
.keyvalue_outer {
    background: #91ca45;
    padding: 6px;
    border-radius: 4px;
    margin: 10px;
}
.keyvalue_inner {
    background: white;
    border-radius: 4px;
    width: 550px;
    padding: 0px;
    padding-top: 6px;
    padding-bottom: 6px;
    border-spacing: 0px 10px;
}
.key {
    text-align: left;
    font-size: 125%;
    padding: 0px;
    font-family: sans-serif;
}
.value {
    text-align: right;
    font-size: 150%;
    padding: 0px;
    width: 150px;
    font-family: monospace;
    border: none;
    border-bottom: 2px solid #91ca45;
}
#button {
    transition: 0.5s;
    font-size: 125%;
    width: 300px;
    margin: 0 auto;
    position: relative;
    font-family: sans-serif;
    font-weight: bold;
    color: black;
    border-color: #91ca45;
    box-shadow: 0px 0px 20px #91ca45;
}
#button:hover {
    color: #91ca45;
    background: white;
    transition: 0.3s;
    cursor: pointer;
    box-shadow: 0px 0px 20px white;
}
  </style>
</head>
<body>
  <table>
    <tr>
      <td style="vertical-align: top; padding-right: 0px">
        <div class="keyvalue_outer">
          <table class="keyvalue_inner">
            <tr>
              <td><input id="shards" class="value" type="text" value="20000"></input></td>
              <td class="key"># shards</td>
            </tr>
            <tr>
              <td><input id="samples_per_shard" class="value" type="text" value="4000"></input></td>
              <td class="key"># samples per shard</td>
            </tr>
            <tr>
              <td><input id="avg_shard_size" class="value" type="text" value="1.6e7"></input></td>
              <td class="key">average shard size (bytes)</td>
            </tr>
          </table>
        </div>
        <div class="keyvalue_outer">
          <table class="keyvalue_inner">
            <tr>
              <td><input id="epochs" class="value" type="text" value="1"></input></td>
              <td class="key"># epochs</td>
            </tr>
            <tr>
              <td><input id="batches_per_epoch" class="value" type="text" value="200"></input></td>
              <td class="key">batches per epoch</td>
            </tr>
            <tr>
              <td><input id="device_batch_size" class="value" type="text" value="16"></input></td>
              <td class="key">device batch size</td>
            </tr>
            <tr>
              <td><input id="avg_batch_time" class="value" type="text" value="0.27"></input></td>
              <td class="key">average batch time (seconds)</td>
            </tr>
          </table>
        </div>
        <div class="keyvalue_outer">
          <table class="keyvalue_inner">
            <tr>
              <td><input id="physical_nodes" class="value" type="text" value="2"></input></td>
              <td class="key"># physical nodes</td>
            </tr>
            <tr>
              <td><input id="devices" class="value" type="text" value="8"></input></td>
              <td class="key"># devices per node</td>
            </tr>
            <tr>
              <td><input id="node_network_bandwidth" class="value" type="text" value="5e8"></input></td>
              <td class="key">internet bandwidth per node (bytes/sec)</td>
            </tr>
          </table>
        </div>
        <div class="keyvalue_outer">
          <table class="keyvalue_inner">
            <tr>
              <td><input id="canonical_nodes" class="value" type="text" value="2"></input></td>
              <td class="key"># canonical nodes</td>
            </tr>
            <tr>
              <td><input id="workers" class="value" type="text" value="8"></input></td>
              <td class="key"># workers per device</td>
            </tr>
            <tr>
              <td><input id="predownload" class="value" type="text" value="3800"></input></td>
              <td class="key">predownload per worker</td>
            </tr>
            <tr>
              <td><input id="cache_limit" class="value" type="text" value=""></input></td>
              <td class="key">cache limit per node</td>
            </tr>
            <tr>
              <td><input id="shuffle_algo" class="value" type="text" value="py1b"></input></td>
              <td class="key">shuffle algorithm</td>
            </tr>
            <tr>
              <td><input id="shuffle_block_size" class="value" type="text" value="16000000"></input></td>
              <td class="key">shuffle block size</td>
            </tr>
            <tr>
              <td><input id="seed" class="value" type="text" value="17"></input></td>
              <td class="key">shuffle seed</td>
            </tr>
          </table>
        </div>
        <div class="keyvalue_outer">
          <div class="keyvalue_inner" id="button" onclick="clicked_simulate()">
            <center>Simulate Streaming</center>
          </div>
        </div>
      </td>
      <td id="result" style="padding-left: 0px">
      </td>
    </tr>
  </table>
  <script type="text/javascript">

var post = function(req, url, on_done) {
    var http = new XMLHttpRequest();
    http.open("POST", url, true);
    http.setRequestHeader("Content-Type", "application/json");
    http.onreadystatechange = function() {
        if (http.readyState == 4 && http.status == 200) {
            var obj = JSON.parse(http.responseText);
            on_done(obj);
        }
    };
    var h1 = '<img src="static/loading.gif">';
    document.getElementById('result').innerHTML = h1;
    var text = JSON.stringify(req);
    http.send(text);
};

var display_image = function(obj) {
    var image = obj.image;
    var h = '<img src="data:image/png;base64,' + image + '">';
    document.getElementById('result').innerHTML = h;
}

var get_int = function(id) {
    var text = document.getElementById(id).value;
    if (text.length == 0){
        return null;
    }
    return parseInt(+text);
};

var get_float = function(id) {
    var text = document.getElementById(id).value;
    if (text.length == 0){
        return null;
    }
    return parseFloat(+text);
};

var get_str = function(id) {
    var text = document.getElementById(id).value;
    if (text.length == 0){
        return null;
    }
    return text;
};

var clicked_simulate = function() {
    var req = {
        shards: get_int("shards"),
        samples_per_shard: get_int("samples_per_shard"),
        avg_shard_size: get_float("avg_shard_size"),
        device_batch_size: get_int("device_batch_size"),
        avg_batch_time: get_float("avg_batch_time"),
        batches_per_epoch: get_int("batches_per_epoch"),
        epochs: get_int("epochs"),
        physical_nodes: get_int("physical_nodes"),
        devices: get_int("devices"),
        node_network_bandwidth: get_float("node_network_bandwidth"),
        workers: get_int("workers"),
        canonical_nodes: get_int("canonical_nodes"),
        predownload: get_int("predownload"),
        cache_limit: get_int("cache_limit"),
        shuffle_algo: get_str("shuffle_algo"),
        shuffle_block_size: get_int("shuffle_block_size"),
        seed: get_int("seed"),
    };
    post(req, "/api/simulate", display_image);
};

clicked_simulate();
  </script>
</body>
</html>
'''

from pathlib import Path

current_file = Path(__file__)
current_file_dir = current_file.parent
project_root = current_file_dir.parent
project_root_absolute = project_root.resolve()
static_root_absolute = project_root_absolute / "simulation/static"

app = FastAPI()

# mount static file directory for the nice loading gif :)
app.mount("/static", StaticFiles(directory=static_root_absolute), name="static")

@app.get('/')
def get_root() -> HTMLResponse:
    """Get the index HTML file."""
    return HTMLResponse(INDEX)

class GetSimulationRequest(BaseModel):
    """simulation input parameters."""
    shards: int
    samples_per_shard: int
    avg_shard_size: float
    device_batch_size: int
    avg_batch_time: float
    batches_per_epoch: int
    epochs: int
    physical_nodes: int
    devices: int
    node_network_bandwidth: float
    workers: int
    canonical_nodes: int
    predownload: int
    cache_limit: Optional[int] = None
    shuffle_algo: Optional[str] = None
    shuffle_block_size: int = 1 << 18
    seed: int = 42

@app.post('/api/simulate')
def post_api_simulate(req: GetSimulationRequest) -> dict:
    """Serve a POST request to simulate a run.

    Args:
        req (GetSimulationRequest): The simulation input params.

    Returns:
        dict: JSON object containing the base64 image string for the simulation plots.
    """
    step_times, shard_downloads = simulate(req.shards, req.samples_per_shard, req.avg_shard_size, req.device_batch_size,
                        req.avg_batch_time, req.batches_per_epoch, req.epochs, req.physical_nodes, req.devices,
                        req.node_network_bandwidth, req.workers, req.canonical_nodes, req.predownload,
                        req.cache_limit, req.shuffle_algo, req.shuffle_block_size, req.seed)
    
    plots_buffer = plot_simulation(step_times, shard_downloads)

    if plots_buffer is not None:
      base64_encoded_image = base64.b64encode(plots_buffer).decode("utf-8")
      return {"image": base64_encoded_image}
    else:
        raise ValueError("plot_simulation returned None. Set web=True to return bytes.")
