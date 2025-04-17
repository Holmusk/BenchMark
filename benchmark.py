import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from utils.logger import save_results_csv
from utils.model_loader import load_model_and_tokenizer
from utils.system_info import get_system_info
from utils.data_loader import load_notes

def run_benchmarks(config):

    batch_size = config["benchmark"].get("batch_size")
    runs = config["benchmark"].get("runs")
    warmup_runs = config["benchmark"].get("warmup_runs")
    model_config = config["model"]
    device_str = config["system"].get("device")


    if None in [batch_size, runs, warmup_runs]:
        raise ValueError("Missing required configuration values (batch_size, runs, warmup_runs)")
    
    #loads model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(model_config, device_str)

    #loads notes from source(csv/db)
    note_texts= load_notes(config["input"])

    durations = []

    #For batching notes
    dataloader = DataLoader(note_texts, batch_size=batch_size, shuffle=False)

    #Warmup runs
    for i in range(warmup_runs):
        for batch in dataloader:
            run_inference(batch, model, tokenizer, device)

    # Benchmarking runs (timed)
    for _ in range(runs):
        start_time = time.time()
        for batch in tqdm(dataloader, desc="Benchmarking"):
            run_inference(batch, model, tokenizer, device)
        end_time = time.time()
        durations.append(end_time - start_time)

    avg_duration = sum(durations) / len(durations)
    total_notes = len(note_texts)
    speed = total_notes / avg_duration

    result = {
        "average_inference_time_sec": avg_duration,
        "inference_speed_notes_per_sec": speed,
        "total_notes": total_notes,
        "batch_size": batch_size,
        "runs": runs,
        "warmup_runs": warmup_runs,
        "durations": durations,
        "system_info": get_system_info(device)  # Adding system info
    }
 
    if config["benchmark"].get("save_output", False):
        save_results_csv(result, config["benchmark"]["output_path"])
    else:
        print("No path specified")


#for batch tokenization and inference
def run_inference(batch, model, tokenizer, device):
    
    #Tokenizing texts in batches
    inputs = tokenizer(batch, return_tensors="pt", padding = True, truncation=True, max_length=512)

    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference 
    with torch.no_grad():
        _ = model(**inputs)






