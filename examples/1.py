from core.runner import BAML


def main():
    # bench = BAML(
    #     automl='ag',
    #     validation_metric='f1',
    #     test_metrics=['average_precision', 'mcc']
    # )
    bench = BAML(
        automl='h2o',
        validation_metric='f1',
    )

    bench.run()


if __name__ == '__main__':
    main()