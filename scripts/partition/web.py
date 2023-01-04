# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Web app to visualize StreamingDataset sample space partitioning.

Install:

    pip3 install fastapi pydantic uvicorn

Run:

    uvicorn scripts.partition.web:app --port 1337 --reload
"""

import os

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from streaming.base.partitioning import get_partitions

INDEX = '''
<!doctype html>
<html>
<head>
  <title>Partitioner</title>
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
    background: radial-gradient(black, #08f);
    border-radius: 4px;
    margin: 10px;
    padding-right: 16px;
}
.node {
    background: radial-gradient(#bdf, #4af);
    border-radius: 4px;
}
.rank {
    background: radial-gradient(#def, #8cf);
    border-radius: 4px;
}
.worker {
    background: radial-gradient(white, #bdf);
    border-radius: 4px;
    padding-top: 0px;
    padding-bottom: 0px;
}
.batch {
    padding-left: 5px;
    padding-right: 5px;
}
.sample {
    background: rgba(255, 255, 255, 0.5);
    border-radius: 4px;
    padding-top: 0px;
    padding-bottom: 0px;
}
.keyvalue_outer {
    background: radial-gradient(black, #08f);
    padding: 6px;
    padding-right: 20px;
    border-radius: 4px;
    margin: 10px;
}
.keyvalue_inner {
    background: white;
    border-radius: 4px;
    width: 275px;
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
    width: 60px;
    font-family: monospace;
    border: none;
    border-bottom: 2px solid #08f;
}
#button {
    transition: 0.5s;
    font-size: 125%;
    font-family: sans-serif;
    width: 275px;
    font-weight: bold;
    color: #048;
    border-color: #048;
    box-shadow: 0px 0px 20px #048;
}
#button:hover {
    color: #8ac;
    background: white;
    transition: 0.5s;
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
              <td><input id="dataset_size" class="value" type="text" value="678"></input></td>
              <td class="key">dataset <u>s</u>amples</td>
            </tr>
            <tr>
              <td><input id="device_batch_size" class="value" type="text" value="7"></input></td>
              <td class="key">device <u>b</u>atch size</td>
            </tr>
            <tr>
              <td><input id="offset_in_epoch" class="value" type="text" value="0"></input></td>
              <td class="key"><u>o</u>ffset in epoch</td>
            </tr>
          </table>
        </div>
        <div class="keyvalue_outer">
          <table class="keyvalue_inner">
            <tr>
              <td><input id="canonical_nodes" class="value" type="text" value="6"></input></td>
              <td class="key"><u>c</u>anonical nodes</td>
            </tr>
            <tr>
              <td><input id="physical_nodes" class="value" type="text" value="3"></input></td>
              <td class="key">physical <u>n</u>odes</td>
            </tr>
            <tr>
              <td><input id="node_devices" class="value" type="text" value="4"></input></td>
              <td class="key">node <u>d</u>evices</td>
            </tr>
            <tr>
              <td><input id="device_workers" class="value" type="text" value="5"></input></td>
              <td class="key">device <u>w</u>orkers</td>
            </tr>
          </table>
        </div>
        <div class="keyvalue_outer">
          <div class="keyvalue_inner" id="button" onclick="clicked_get_partitions()">
            <center>Partition</center>
          </div>
        </div>
      </td>
      <td id="result" style="padding-left: 0px">
      </td>
    </tr>
  </table>
  <script type="text/javascript">
var event_code2field = {
    KeyS: "dataset_size",
    KeyB: "device_batch_size",
    KeyO: "offset_in_epoch",
    KeyC: "canonical_nodes",
    KeyN: "physical_nodes",
    KeyD: "node_devices",
    KeyW: "device_workers",
};

var get_int = function(id) {
    var text = document.getElementById(id).value;
    return parseInt(text);
};

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
    var text = JSON.stringify(req);
    http.send(text);
};

var draw_sample_id = function(id, max_id_digits) {
    var is_neg = id < 0;
    id = Math.abs(id);

    var digits = [];
    for (var i = 0; i < max_id_digits; ++i) {
        digits.push(Math.floor(id % 10));
        id /= 10;
    }
    digits.reverse();

    for (var i = 0; i < digits.length - 1; ++i) {
        if (!digits[i]) {
            digits[i] = '<span style="visibility: hidden">0</span>';
        } else {
            break;
        }
    }

    if (is_neg) {
        digits[digits.length - 1] = '<span style="visibility: hidden">0</span>';
    }
    var text = digits.join('');
    return '<td class="sample">' + text + '</td>';
};

var draw_partitions = function(obj) {
    var ids = obj.ids;

    var num_nodes = ids.length;
    var ranks_per_node = ids[0].length;
    var workers_per_rank = ids[0][0].length;
    var batches_per_worker = ids[0][0][0].length;
    var batch_size = ids[0][0][0][0].length;

    var num_ids = 0;
    for (var node = 0; node < ids.length; ++node) {
        for (var rank = 0; rank < ids[node].length; ++rank) {
            for (var worker = 0; worker < ids[node][rank].length; ++worker) {
                for (var batch = 0; batch < ids[node][rank][worker].length; ++batch) {
                    for (var sample = 0; sample < ids[node][rank][worker][batch].length; ++sample) {
                        var id = ids[node][rank][worker][batch][sample];
                        if (0 <= id) {
                            ++num_ids;
                        }
                    }
                }
            }
        }
    }
    var max_id_digits = Math.ceil(Math.log10(num_ids - 1));

    var h = '';
    for (var node = 0; node < ids.length; ++node) {
        h += '<table class="all">';
        h += '<tr>';
        h += '<td>';
        h += '<table class="node">';
        for (var rank = 0; rank < ids[node].length; ++rank) {
            h += '<tr>';
            if (!rank) {
                h += '<td rowspan="' + ranks_per_node + '">';
                h += 'Node&nbsp;' + node + '&nbsp;';
                h += '</td>';
            }
            h += '<td>';
            h += '<table class="rank">';
            for (var worker = 0; worker < ids[node][rank].length; ++worker) {
                h += '<tr>';
                if (!worker) {
                    h += '<td rowspan="' + workers_per_rank + '">';
                    h += 'Device&nbsp;' + rank + '&nbsp;';
                    h += '</td>';
                }
                h += '<td>';
                h += '<table class="worker">';
                h += '<tr>';
                h += '<td>';
                h += 'Worker&nbsp;' + worker + '&nbsp;';
                h += '</td>';
                for (var batch = 0; batch < ids[node][rank][worker].length; ++batch) {
                    for (var sample = 0; sample < ids[node][rank][worker][batch].length; ++sample) {
                        var id = ids[node][rank][worker][batch][sample];
                        h += draw_sample_id(id, max_id_digits);
                    }
                }
                h += '</tr>';
                h += '</table>';
                h += '</td>';
                h += '</tr>';
            }
            h += '</table>';
            h += '</td>';
            h += '</tr>';
        }
        h += '</table>';
        h += '</td>';
        h += '</tr>';
        h += '</table>';
    }

    document.getElementById('result').innerHTML = h;

    for (var key in event_code2field) {
        var field = event_code2field[key];
        document.getElementById(field).blur();
    }
};

var clicked_get_partitions = function() {
    var req = {
        dataset_size: get_int("dataset_size"),
        device_batch_size: get_int("device_batch_size"),
        offset_in_epoch: get_int("offset_in_epoch"),
        canonical_nodes: get_int("canonical_nodes"),
        physical_nodes: get_int("physical_nodes"),
        node_devices: get_int("node_devices"),
        device_workers: get_int("device_workers"),
    };
    post(req, "/api/get_partitions", draw_partitions);
};

var focus = function(field) {
    var elem = document.getElementById(field);
    elem.value = '';
    elem.focus();
};

document.onkeypress = function(event) {
    if (event.code == "Enter") {
        clicked_get_partitions();
        return;
    }

    var chr = String.fromCharCode(event.keyCode);

    if (chr < '0' || '9' < chr) {
        var field = event_code2field[event.code];
        if (field !== undefined) {
            focus(field);
        }
        event.preventDefault();
    }
};

clicked_get_partitions();
  </script>
</body>
</html>
'''

app = FastAPI()


@app.get('/')
def get_root() -> str:
    """Get the index HTML file."""
    return HTMLResponse(INDEX)


class GetPartitionsRequest(BaseModel):
    """Partitioning configuration."""
    dataset_size: int
    device_batch_size: int
    offset_in_epoch: int
    canonical_nodes: int
    physical_nodes: int
    node_devices: int
    device_workers: int


@app.post('/api/get_partitions')
def post_api_get_partitions(req: GetPartitionsRequest) -> dict:
    """Serve a POST request to get partitions.

    Args:
        req (GetPartitionsRequest): The partitioning configuration.

    Returns:
        dict: JSON object containing the sample IDs, of shape (nodes, ranks per node, workers per
            rank, batches per worker, batch size).
    """
    ids = get_partitions(req.dataset_size, req.canonical_nodes, req.physical_nodes,
                         req.node_devices, req.device_workers, req.device_batch_size,
                         req.offset_in_epoch)
    ids = ids.reshape(req.physical_nodes, req.node_devices, req.device_workers, -1,
                      req.device_batch_size)
    return {
        'ids': ids.tolist(),
    }
