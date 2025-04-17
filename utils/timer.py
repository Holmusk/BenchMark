import torch
import time
from tqdm import tqdm

def benchmark_model(model, tokenizer, notes, batch_size, runs, warmup_runs, device):
    all_durations = []

    inputs = [notes[i:i+batch_size] for i in range(0, len(notes), batch_size)]

    #For warmup runs without saving anything(no timing)
    for _ in range(warmup_runs):
        for batch in inputs:
            with torch.no_grad():
                _ = model(**tokenizer(batch, return_tensors = "pt", padding = True, truncation = True).to(device))

    #For timed runs
    for _ in range(runs):
        
        start = time.time()

        for batch in tqdm(inputs, desc="Benchmarking"):
            with torch.no_grad():
                _ = model(**tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device))
            duration = time.time() - start
            all_durations.append(duration)

    avg_time = sum(all_durations) / runs
    notes_per_sec = len(notes) / avg_time

    return {
        "average_inference_time_sec": avg_time,
        "inference_speed_notes_per_sec": notes_per_sec,
        "total_notes": len(notes),
        "batch_size": batch_size,
        "runs": runs,
        "warmup_runs": warmup_runs,
        "durations": all_durations
    }


