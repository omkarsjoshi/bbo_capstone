"""
bbo_data.py: Loads data into the notebook

The folder consists of initial dataset, along with the latest queries and results
obtained from the Capstoen portal. 
"""
import numpy as np
import pandas as pd
from numpy import array  # needed for eval() inside _read_records


def load_function(fn_num, data_dir="initial_data"):
    """Load function's original initial data and returns it as dataframe"""
    X_read = np.load(f'{data_dir}/function_{fn_num}/initial_inputs.npy')
    y_read = np.load(f'{data_dir}/function_{fn_num}/initial_outputs.npy')
    cols = [f'x{i+1}' for i in range(X_read.shape[1])]
    return pd.DataFrame(X_read, columns=cols), pd.Series(y_read, name='y')


def _read_records(path):
    """Prase the inputs.txt and outputs.txt files by accumulating the lines from
    these files and returns it as a list"""
    with open(path) as f:
        text = f.read()
    records, buf, depth = [], "", 0
    for line in text.split("\n"):
        if not line.strip():
            continue
        buf += line
        depth += line.count("[") - line.count("]")
        if depth == 0:
            records.append(eval(buf, {"array": array, "np": np}))
            buf = ""
    return records


def append_from_txt(fn_num, X, y, inputs_path="inputs_W12.txt", outputs_path="outputs_W12.txt"):
    """Append the data from inputs_WXX.txt and outputs_WXX to the initla data.txt"""
    inputs = _read_records(inputs_path)
    outputs = _read_records(outputs_path)
    idx = fn_num - 1
    X_new = np.array([row[idx] for row in inputs])
    y_new = np.array([row[idx] for row in outputs])
    cols = [f'x{i+1}' for i in range(X_new.shape[1])]
    X_out = pd.concat([X, pd.DataFrame(X_new, columns=cols)], ignore_index=True)
    y_out = pd.concat([y, pd.Series(y_new, name='y')], ignore_index=True)
    return X_out, y_out



def load_all_individual(data_dir="initial_data", inputs_path="inputs_W12.txt",
                         outputs_path="outputs_W12.txt", n_functions=8):
    """Build all_X and all_y for all functions, along with X and y for easy reference
    through variable explorer"""
    all_X, all_y = [], []
    for fn in range(1, n_functions + 1):
        X, y = load_function(fn, data_dir)
        X, y = append_from_txt(fn, X, y, inputs_path, outputs_path)
        all_X.append(X); all_y.append(y)

    flat = [v for pair in zip(all_X, all_y) for v in pair]
    return (all_X, all_y, *flat)
