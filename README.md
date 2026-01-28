# MLBenchmark
MLBenchmark is a toolkit, that allows to conviniently benchmark various machine learning methods on different data modalities. It can be utilized by ML engineers and scientists, developing their own method, as well as the common users to test different machine learning scenarios.

### Project status
This project is under active development and new tasks and ML methods to be added soon.
Currently the only supported task is tabular classification.
Also, currently it only supports ML algorithms, that are part of AutoML tools. Specifically, [AutoGluon](https://github.com/autogluon/autogluon) and [H2O-AutoML](https://github.com/h2oai/h2o-3).

### Installation and usage

#### Installation
1. Clone the project.
2. Initialize project with `uv init` and create a virtual environment with `uv venv -p 3.10`.
3. Install dependencies with `uv sync`. For CPU-only installation type `uv sync --extra cpu`. 

#### Usage
```python
from core.api import MLBenchmark
from data.repository import OpenMLDatasetRepository


def main():
    bench = MLBenchmark(
        repository=BinaryImbalancedDatasetRepository(),
        automl='ag',
        preset='best',
        metric='f1'
    )
    bench.run()


if __name__ == '__main__':
    main()
```

### Contribution
Contribution is welcome! Feel free to open issues and submit pull requests.
