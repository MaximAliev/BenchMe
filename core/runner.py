import itertools
import logging
import pprint
import sys
import time
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Union, Optional, List, cast, final
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from loguru import logger

from core.automl import AutoML, Imbaml, AutoGluon
from core.domain import TabularDataset, MLTask
from core.preprocessing import TabularDatasetPreprocessor
from benchmark.repository import FittedModel, OpenMLRepository, TabularDatasetRepository, ZenodoRepository


# TODO: support presets and leaderboard.
class MLBench:
    def __init__(
        self,
        automl = 'ag',
        validation_metric = 'f1',
        repository = 'zenodo',
        log_to_file = True,
        test_metrics: Optional[List[str]] = None,
        *args,
        **kwargs
    ):
        self._validation_metric: str
        self._automl: AutoML
        self._fitted_model: Optional[FittedModel]

        self.validation_metric = validation_metric
        self.automl = (automl, args, kwargs)
        self.repository = repository
        self._log_to_file = log_to_file
        self._test_metrics = test_metrics

        self._configure_environment()

    @logger.catch
    def run(self) -> None:
        for dataset in self._repository.datasets:
            self._run_on_dataset(dataset)

    def _configure_environment(self) -> None:
        # if self._log_to_file:
        #     log_filepath = 'logs/'
        #     Path(log_filepath).mkdir(parents=True, exist_ok=True)
        #     log_filepath += datetime.now().strftime(f'{self._automl} {",".join(self.validation_metric)} %Y.%m.%d %H:%M:%S')
        #     log_filepath += '.log'
        #     logging_handlers.append(logging.FileHandler(filename=log_filepath, encoding='utf-8', mode='w'))

        # logger.add(sys.stdout, colorize=True, format='{level} {message}', level='INFO')

        logger.info(f"Validation metric is {self.validation_metric}.")

    @final
    def _run_on_dataset(self, dataset: TabularDataset) -> None:
        if dataset is None:
            logger.error("Run failed. Reason: dataset is undefined.")
            return

        if isinstance(dataset.X, np.ndarray) or isinstance(dataset.X, pd.DataFrame):
            preprocessor = TabularDatasetPreprocessor()
            preprocessed_data = preprocessor.preprocess_data(dataset.X, dataset.y.squeeze())

            assert preprocessed_data is not None

            X, y = preprocessed_data
            X_train, X_test, y_train, y_test = preprocessor.split_data_on_train_and_test(X, y.squeeze())
        else:
            raise TypeError(f"pd.DataFrame or np.ndarray was expected. Got: {type(dataset.X)}")

        logger.info(f"{dataset.id}...Loaded dataset name: {dataset.name}.")
        logger.info(f'Rows: {X_train.shape[0]}. Columns: {X_train.shape[1]}')
        
        class_belongings = Counter(y_train)
        logger.info(class_belongings)

        if len(class_belongings) > 2:
            raise ValueError("Multiclass problems currently not supported.")

        iterator_of_class_belongings = iter(sorted(class_belongings))
        *_, positive_class_label = iterator_of_class_belongings
        logger.debug(f"Inferred positive class label: {positive_class_label}.")

        number_of_positives = class_belongings.get(positive_class_label)
        if number_of_positives is None:
            raise ValueError("Unknown positive class label.")

        training_dataset = TabularDataset(
            id=dataset.id,
            name=dataset.name,
            X=X_train,
            y=y_train,
            y_label=dataset.y_label
        )

        training_dataset_size = int(pd.DataFrame(X_train).memory_usage(deep=True).sum() / (1024 ** 2))
        training_dataset.size = training_dataset_size
        logger.debug(f"Train sample size is approximately {training_dataset.size} mb.")

        id = itertools.count(start=1)
        task  = MLTask(
            id=next(id),
            dataset=training_dataset,
            metric=self.validation_metric
        )

        start_time = time.time()
        self._automl.fit(task)

        time_passed = time.time() - start_time
        logger.info(f"Training took {time_passed // 60} min.")

        y_predicted = self._automl.predict(X_test)

        metrics = {self.validation_metric}
        if self._test_metrics is not None:
            for metric in self._test_metrics:
                metrics.add(metric)
        self._automl.score(metrics, y_test, y_predicted, positive_class_label)

    @property
    def repository(self) -> TabularDatasetRepository:
        return self._repository
    
    @repository.setter
    def repository(self, value: str):
        if value == 'zenodo':
            self._repository = ZenodoRepository()
        elif value == 'openml':
            self._repository = OpenMLRepository()
        else:
            raise ValueError(
                f"""
                Invalid value of repository parameter:{value}.
                Options available: ['openml', 'zenodo'].
                """
            )
    
    @property
    def validation_metric(self) -> str:
        return self._validation_metric
    
    @validation_metric.setter
    def validation_metric(self, value: str):
        if value not in [
            'f1',
            'precision',
            'recall',
            'roc_auc',
            'average_precision',
            'balanced_accuracy',
            'mcc',
            'accuracy'
        ]:
            raise ValueError(
                f"""
                Invalid value of metric parameter: {value}.
                Options available: [
                    'f1',
                    'precision',
                    'recall',
                    'roc_auc',
                    'average_precision',
                    'balanced_accuracy',
                    'mcc',
                    'accuracy'].
                """)
        self._validation_metric = value
    
    @property
    def automl(self) -> AutoML:
        return self._automl

    @automl.setter
    def automl(self, value: Tuple[str, Tuple[Any, ...], Dict[str, Any]]):
        if value[0] == 'ag':
            self._automl = AutoGluon(*value[1], **value[2])
        elif value[0] == 'imbaml':
            self._automl = Imbaml(*value[1], **value[2])
        else:
            raise ValueError(
                f"""
                Invalid value of automl parameter: {value[0]}.
                Options available: ['ag', 'imbaml'].
                """)
