# traffic-benchmark

A benchmarking suite for traffic data generation and analysis.

## Project Structure

```
traffic-benchmark/
├── domain/             # Domain models and business logic
│   └── __init__.py
├── generators/         # Traffic data generators
│   └── __init__.py
├── dataset/            # Dataset handling
│   ├── __init__.py
│   ├── core/           # Core dataset loading and processing
│   │   └── __init__.py
│   └── stats/          # Statistical analysis of datasets
│       └── __init__.py
├── scripts/            # Standalone scripts (data prep, experiments)
└── tests/              # Unit and integration tests
    └── __init__.py
```

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt
```
