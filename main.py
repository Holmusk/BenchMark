import yaml
import torch
import pandas as pd
import os
import sys
from utils.system_info import get_system_info
from utils.data_loader import load_notes
from benchmark import run_benchmarks

def main():
    # Load config from current working directory (where user runs the command)
    config_path = os.path.join(os.getcwd(), "config.yml")

    if not os.path.exists(config_path):
        print(f"config.yml not found in current directory: {config_path}")
        print("Please make sure config.yml is in the directory where you run 'benchtool'")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Loads system info
    system_config = config.get("system", {})
    torch.set_num_threads(system_config.get("num_threads", 1))

    # Loads data from CSV
    input_config = config.get("input", {})
    if input_config.get("mode") == "csv":
        csv_path = input_config["csv"]["path"]
        # If the path is not absolute, resolve it relative to the current working directory
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(os.getcwd(), csv_path)
        input_config["csv"]["path"] = csv_path
    note_texts = load_notes(input_config)  # This loads the notes based on config (CSV mode)

    # To run benchmarking
    print("Running benchmarks...")
    run_benchmarks(config)

if __name__ == "__main__":
    main()
