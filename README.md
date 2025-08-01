# BenchTool

A lightweight tool to benchmark the performance of NLP models based on medical text datasets. Supports HuggingFace, ONNX, and ORT models, and works with input data from CSV files

## Features
- Compare multiple NLP models
- Supports HuggingFace, ONNX, and ORT model types
- Input from CSV
- Outputs performance metrics like inference time and throughput
- Configurable via a single YAML file

## Installation

 **Create a virtual environment:**
   ```sh
   python3 -m venv env
   source env/bin/activate  
   ```

Download the latest release wheel from the [GitHub Releases page](https://github.com/Holmusk/BenchMark/releases) and install with pip:

```sh
pip install https://github.com/Holmusk/BenchMark/releases/download/v0.1.0/benchtool-0.1.0-py3-none-any.whl
```

## Usage

Prepare your `config.yml` and input CSV in your working directory. Then run:

```sh
benchtool
```

## Example `config.yml`

```yaml
model:
  name: "dbmdz/bert-large-cased-finetuned-conll03-english"
  task: ""

input:
  mode: "csv"
  csv:
    path: "100notes.csv"
    column: "note_text"

benchmark:
  batch_size: 16
  runs: 2
  warmup_runs: 1
  save_output: true
  output_path: "./benchmark_results.csv"

system:
  device: "cpu"
  num_threads: 8
```

## Notes
- Make sure your `config.yml` and input CSV are in the directory where you run `benchtool`.
- For more details, see the code and comments in the repository.
