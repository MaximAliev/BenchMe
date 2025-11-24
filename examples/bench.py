from core.runner import AutoMLHub


def main():
    automl_hub = AutoMLHub(repository='zenodo')
    dataset_repo = automl_hub.repository
    dataset_repo.load_datasets()

    automl_hub.run()


if __name__ == '__main__':
    main()