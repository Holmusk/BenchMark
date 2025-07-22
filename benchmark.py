import time
from tqdm import tqdm
import torch
import os
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from utils.logger import save_results
from utils.model_loader import load_model_and_tokenizer
from utils.system_info import get_system_info
from utils.data_loader import load_notes
from utils.dataset import TokenizedDataset
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTModelForSequenceClassification

def run_benchmarks(config):
    # Set tokenizer parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Get configuration
    batch_size = config["benchmark"].get("batch_size", 32)
    runs = config["benchmark"].get("runs")
    warmup_runs = config["benchmark"].get("warmup_runs")
    model_config = config["model"]
    device_str = "cpu"  # Force CPU
    
    # Optimize CPU threads
    n_cores = os.cpu_count() or 1  # Default to 1 if cpu_count returns None
    torch.set_num_threads(n_cores)
    torch.set_num_interop_threads(n_cores)
    
    if None in [batch_size, runs, warmup_runs]:
        raise ValueError("Missing required configuration values")
    
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(model_config, device_str)
    
    # Load notes and create dataset with batched tokenization
    note_texts = load_notes(config["input"])
    print(f"Processing {len(note_texts)} notes...")
    
    # Create dataset with batched tokenization
    dataset = TokenizedDataset(
        texts=note_texts,
        tokenizer=tokenizer,
        max_length=512,
        batch_size=1000  # Process 1000 texts at a time during tokenization
    )
    
    # Create dataloader with single worker to avoid tokenizer warnings
    data_collator = DataCollatorWithPadding(tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,  # Use single worker to avoid tokenizer parallelism issues
        pin_memory=False  # Disable for CPU
    )

    durations = []
    batch_times = []

    # Set model to eval mode if it's a PyTorch model
    if not isinstance(model, (ORTModelForFeatureExtraction, ORTModelForSequenceClassification)):
        model.eval()

    # Warmup runs
    print("Running warmup...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                _ = model(**batch)

    # Benchmarking runs
    print(f"Running {runs} benchmark iterations...")
    for run in range(runs):
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Run {run + 1}/{runs}"):
                batch_start = time.perf_counter()
                batch = {k: v.to(device) for k, v in batch.items()}
                _ = model(**batch)
                batch_times.append(time.perf_counter() - batch_start)
        
        durations.append(time.perf_counter() - start_time)

    # Calculate statistics
    avg_duration = sum(durations) / len(durations)
    total_notes = len(note_texts)
    speed = total_notes / avg_duration
    
    # Calculate batch statistics
    avg_batch_time = sum(batch_times) / len(batch_times)
    max_batch_time = max(batch_times)
    min_batch_time = min(batch_times)

    result = {
        "average_inference_time_sec": avg_duration,
        "inference_speed_notes_per_sec": speed,
        "total_notes": total_notes,
        "batch_size": batch_size,
        "runs": runs,
        "warmup_runs": warmup_runs,
        "durations": durations,
        "avg_batch_time": avg_batch_time,
        "max_batch_time": max_batch_time,
        "min_batch_time": min_batch_time,
        "num_cpu_threads": n_cores,
        "system_info": get_system_info(device)
    }

    if config["benchmark"].get("save_output", False):
        save_results(result, config["benchmark"]["output_path"])
    else:
        print("\nBenchmark Results:")
        print(f"Average Processing Time: {avg_duration:.2f} seconds")
        print(f"Processing Speed: {speed:.2f} notes/second")
        print(f"Average Batch Time: {avg_batch_time:.4f} seconds")






