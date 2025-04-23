# Benchmarking Tool for NLP Models

A lightweight tool to benchmark the performance of NLP models based on medical text datasets. It supports HuggingFace, ONNX, and ORT models and works with input data from CSV files or a PostgreSQL database.

---

- Compare multiple NLP models
- Supports HuggingFace, ONNX, and ORT model types
- Input from CSV or PostgreSQL database
- Outputs performance metrics like inference time and throughput
- Configurable via a single YAML file

---

##  Installation

```git clone https://github.com/Holmusk/BenchMark.git```
```cd your-directory```

create a virtual environment: ```python3 -m venv Enviornment```\
activate:  ```source Enviornment/bin/activate```

To install dependenices: ```pip install -r requirements.txt```
