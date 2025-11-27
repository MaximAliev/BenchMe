from core.runner import MLBench


def main():
    automl_hub = MLBench(validation_metric='f1', test_metrics=['average_precision', 'mcc'])
    dataset_repo = automl_hub.repository
    dataset_repo.load_datasets()

    automl_hub.run()


if __name__ == '__main__':
    main()