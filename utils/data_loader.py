import pandas as pd
import psycopg2

def load_notes(input_cfg):
    mode = input_cfg["mode"]

    if mode == "csv":
        path = input_cfg["csv"]["path"]
        column = input_cfg["csv"]["column"]
        return pd.read_csv(path)[column].dropna().tolist()

    else:
        raise ValueError("Unsupported input mode")
