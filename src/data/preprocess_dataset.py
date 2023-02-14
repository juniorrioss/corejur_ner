"""
    Main file for Named Entity Recognition Utils
    Contains Dataset Stats, KFOLD splits, Fold Stratifed Balance
    All settings must be changed in config/settings.yaml folder

"""
import os

import hydra
from omegaconf import DictConfig, OmegaConf

import dataset_preprocessing as preprocessing
import utils


@hydra.main(config_path="../../config", config_name="settings")
def main(config: DictConfig):
    """Run the entire pipe
    Contains Stats, Kfold splits and fold balancing


    Args:
        config (DictConfig): All settings in settings.yaml file
    """
    FILENAME = config["DATASET"]["filename"]
    path_file = os.path.join(config["DATASET"]["folder_raw"], FILENAME)

    SAVE_FOLDER = config["DATASET"].get("folder_preprocessed", "preprocess")

    save_path = os.path.join(SAVE_FOLDER, FILENAME)
    # assert the version do not exists
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    assert (
        os.path.exists(os.path.join(SAVE_FOLDER, FILENAME)) is False
    ), "The version already exists"

    print("Loading Dataset")
    df = utils.conll2pandas(path_file)  # LOAD THE DATASET FROM CONLL FILE
    print("Dataset loaded")

    PreProcessor = preprocessing.PreProcessDataset(config["PREPROCESSING"])
    df_preprocessed = PreProcessor.run(df)

    # SAVE IN CONLL
    utils.pandas2conll(df_preprocessed, save_path)

    print("Done!")


if __name__ == "__main__":
    main()
