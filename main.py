import yaml
import torch
import pandas as pd
from utils.system_info import get_system_info
from utils.data_loader import load_notes
from benchmark import run_benchmarks

def main():
    #Loads the configuration
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    #Loads system info
    system_config = config.get("system", {})
    torch.set_num_threads(system_config.get("num_threads", 1))

    #Loads data from CSV 
    input_config = config.get("input", {})
    note_texts = load_notes(input_config)  # This loads the notes based on config (CSV mode)

    #Runs benchmarking
    print("Running benchmarks...")
    run_benchmarks(config)

if __name__ == "__main__":
    main()
