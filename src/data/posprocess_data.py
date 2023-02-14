"""
    Main file for Named Entity Recognition Utils
    Contains Dataset Stats, KFOLD splits, Fold Stratifed Balance
    All settings must be changed in config/settings.yaml folder

"""
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import KFold

import utils
from stats import DatasetAnalysis
import dataset_feature as feature
from sklearn.model_selection import train_test_split


@hydra.main(config_path="../../config", config_name="settings")
def main(config: DictConfig):
    """Run the entire pipe
    Contains Stats, Kfold splits and fold balancing


    Args:
        config (DictConfig): All settings in settings.yaml file
    """

    random_state = config["UTILS"].get("random_state", 0)
    utils.fix_seed(random_state)

    FILENAME = config["DATASET"]["filename"]
    path_file = os.path.join(config["DATASET"]["folder_preprocessed"], FILENAME)

    SAVE_FOLDER = config["DATASET"].get("folder_processed", "processed_data")
    save_path = os.path.join(
        SAVE_FOLDER, config["DATASET"].get("experiment_name", "output")
    )

    N_KFOLD = config["KFOLD"].get("n_fold", 5)  # DEFAULT VALUE OF 5

    print("Loading Dataset")
    df = utils.conll2pandas(path_file)  # LOAD THE DATASET FROM CONLL FILE
    print("Dataset loaded")

    # assert the version do not exists
    assert os.path.exists(save_path) is False, "The version already exists"
    os.makedirs(save_path)

    # ---------------------- ALL DATA ANALYSIS ----------------------

    analysis_fulldataset = DatasetAnalysis(df=df)
    stats = analysis_fulldataset.generate_dataset_info(is_alldata=True)
    with open(os.path.join(save_path, "stats_full.txt"), "w", encoding="utf-8") as f:
        f.writelines(stats)

    analysis_fulldataset.convert_stats2excel(save_path)
    analysis_fulldataset.plot_graphs(
        save_path, verbose=config["UTILS"].get("plot_verbose", False)
    )

    # SPLIT TEST PORTION
    if config["DATASET"].get("TEST").get("generate_test_file", False):
        test_size = config["DATASET"].get("TEST").get("test_size", 0.10)

        df, test_dataset = train_test_split(
            df, test_size=test_size, random_state=random_state, shuffle=True
        )

        utils.pandas2conll(test_dataset, save_path + "/test.conll")
        utils.pandas2json(test_dataset, save_path + "/test.json")

    # Feature Engineering
    FeatureProcessor = feature.FeatureEnginnering(config["FEATURES"])
    df = FeatureProcessor.run(df)

    with open(
        os.path.join(save_path, "features_snapshot.yaml"),
        "w",
        encoding="utf-8",
    ) as f:
        f.writelines(OmegaConf.to_yaml(config["FEATURES"]))

    # --------------- SPLIT IN K FOLDS AND GENERATE ANALYSIS ----------
    # KFOLD
    kf = KFold(n_splits=N_KFOLD, random_state=random_state, shuffle=True)

    for i, (train_index, test_index) in enumerate(kf.split(df)):
        path_fold = save_path + "/" + "fold-" + str(i) + "/"  # PATH TO SAVE
        os.makedirs(path_fold)  # CREATE THE FOLDER VERSION AND SUBFOLDER

        # get the data from indexes
        train_data, test_data = df.loc[train_index], df.loc[test_index]

        # FOLD ANALYSIS
        stats = []
        analysis_train = DatasetAnalysis(df=train_data)
        analysis_test = DatasetAnalysis(df=test_data)
        stats.extend(
            analysis_train.generate_dataset_info(n_fold=i, train_data=True)
        )  # TRAIN DATA
        stats.extend(
            analysis_test.generate_dataset_info(n_fold=i, train_data=False)
        )  # TEST DATA
        # save stats

        # SAVE KFOLD SPLIT DATASET
        # SAVE IN CONLL
        utils.pandas2conll(train_data, path_fold + "train.conll")
        utils.pandas2conll(test_data, path_fold + "dev.conll")

        # SAVE IN JSON
        utils.pandas2json(train_data, path_fold + "train.json")
        utils.pandas2json(test_data, path_fold + "dev.json")

        with open(os.path.join(path_fold, "stats.txt"), "w", encoding="utf-8") as f:
            f.writelines(stats)

        print(f"Save dataset and stats for fold-{i}")

        if config["DATASET"].get("save_only_first_fold", True):
            print("SAVING ONLY FOLD 0")
            break

    print("Done!")


if __name__ == "__main__":
    main()
