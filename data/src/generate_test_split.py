# import numpy as np
from utils import pandas2conll, pandas2json
import dataset_preprocessing as preprocessing
import pandas as pd
import numpy as np
import os
import utils

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="settings", version_base=None)
def generate_test_split(config: DictConfig):
    folder = config["DATASET"].get("folder")
    current_dataset_path = os.path.join(folder, config["DATASET"].get("filename"))
    newer_dataset_path = os.path.join(
        folder, config["DATASET"].get("TEST").get("generate_test_from")
    )
    length_to_truncate = config["DATASET"].get("TEST").get("length_to_truncate_test")
    SAVE_FOLDER = config["SAVE"].get("save_folder", "output_folder")

    save_name = (
        "test-v"
        + current_dataset_path.split(".")[0][-2:]
        + "-v"
        + newer_dataset_path.split(".")[0][-2:]
    )
    df = utils.conll2pandas(current_dataset_path)

    df_newer = utils.conll2pandas(newer_dataset_path)

    diff_row = len(df_newer) - len(df)
    test_dataset = df_newer.iloc[:diff_row]

    # FILTER SENTENCES WITH ENTITIES
    tags_to_remove = config["PREPROCESSING"].get("fill_O_tags", "")
    test_dataset = preprocessing.fill_O_tags(test_dataset, tags_to_remove)

    # Datas_do_contrato e Datas_dos_fatos PARA Datas
    datas_to_change = config["PREPROCESSING"].get("datas_aggregation")
    print("Datas Aggretation to Generic Datas", datas_to_change)
    # hardcoded due to business decision
    test_dataset = preprocessing.datas_change(
        test_dataset, datas_to_change=datas_to_change
    )
    test_dataset = preprocessing.remove_label_punctuaction(test_dataset)

    test_dataset["len"] = test_dataset["text"].apply(len)
    truncate_data = test_dataset[test_dataset["len"] > length_to_truncate]
    test_dataset = test_dataset.drop(truncate_data.index, axis=0)

    texts = []
    labels = []
    for i, rows in truncate_data.iterrows():
        text = rows["text"]
        tags = rows["tags"]
        length = rows["len"]
        steps = np.arange(0, length, length_to_truncate).tolist() + [length]
        for i in range(len(steps) - 1):
            texts.append(text[steps[i] : steps[i + 1]])
            labels.append(tags[steps[i] : steps[i + 1]])

    test_dataset = test_dataset.drop("len", axis=1)
    truncate_data = pd.DataFrame.from_dict({"text": texts, "tags": labels})
    test_dataset = pd.concat([test_dataset, truncate_data])
    pandas2conll(test_dataset, os.path.join(SAVE_FOLDER, save_name + ".conll"))
    pandas2json(test_dataset, os.path.join(SAVE_FOLDER, save_name + ".json"))


if __name__ == "__main__":
    generate_test_split()
    print("DONE! TEST DATASET")
