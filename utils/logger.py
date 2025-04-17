import csv
import os

def save_results_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    fieldnames = list(results.keys())
    with open(path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(results)
