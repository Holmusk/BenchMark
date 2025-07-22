import csv
import os
import psycopg2
import json
from utils.data_loader import load_notes

def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    fieldnames = list(results.keys())
    with open(path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(results)
