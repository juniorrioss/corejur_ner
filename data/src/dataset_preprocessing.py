from typing import List
from unidecode import unidecode
import numpy as np


def trucate_sentence_max_length(df, max_length=256):
    """Truncate sentences length

    Args:
        df (pd.DataFrame): The dataframe object
        length_to_filter (int, optional): The sentence max length accepted.
        Defaults to 256.

    Returns:
        pd.DataFrame: The dataframe object filtered
    """
    assert max_length > 0, "Length must be positive"

    def truncate_sentences(row, max_length=256):
        if len(row["text"]) > max_length:
            row["text"] = row["text"][:max_length]
            row["tags"] = row["tags"][:max_length]

        return row

    df[["text", "tags"]] = df[["text", "tags"]].apply(
        lambda row: truncate_sentences(row, max_length=max_length),
        axis=1,
        result_type="expand",
    )
    return df


def fill_O_tags(df, tags_to_remove: List):
    """Function to REPLACE a list of entities (tags) in dataFrame to 'O'
    All tags to be removed are replaced to 'O'

    Args:
        df (pd.DataFrame): The dataframe object
        tags_to_remove (List): List of entities (tags) to be removed

    Returns:
        pd.Dataframe: The dataframe object without the list of tags
    """
    print("Fill O tags", tags_to_remove)

    df["tags"] = df["tags"].apply(
        lambda tags: ["O" if tag[2:] in tags_to_remove else tag for tag in tags]
    )

    return df


# def filter_entities(df, minimum_entity_ratio: 0):
#     assert (
#         minimum_entity_ratio > 0 and minimum_entity_ratio < 1
#     ), "Ratio must be between 0 and 1"

#     # TODAS AS TAGS DIFERENTES DE 'O'
#     labels = {
#         tag[2:]: 0 for tags in df["tags"] for tag in tags if tag != "O"
#     }  # cria todas as labels com valor 0
#     # verifica a quantidade das labels
#     tags = np.array([tag[2:] for tags in df["tags"] for tag in tags if tag != "O"])
#     # associa a label com a quantidade
#     for k in labels:
#         labels[k] = sum(tags == k)

#     # ORDENA CRESCENTE E DIVIDE PELO TOTAL (GERANDO PORCENTAGEM)
#     total_valid_tags = sum(labels.values())
#     labels = {
#         k: v / total_valid_tags
#         for k, v in sorted(labels.items(), key=lambda item: item[1], reverse=False)
#     }

#     entities_to_remove = []
#     for k, v_ratio in labels.items():
#         if v_ratio < minimum_entity_ratio:
#             entities_to_remove.append(k)
#         else:
#             # array está em ordem crescente, ou seja, todos os outros serão > minimo

#             break
#     return fill_O_tags(df, entities_to_remove=entities_to_remove)


def remove_jurisprudencia_sentence(df):
    # REMOVE JURISPRUDENCIA
    df["haveJurisprudencia"] = df["tags"].apply(lambda x: "B-Jurisprudência" in x)
    print("SENTENÇAS COM JURISPRUDENCIA ", df["haveJurisprudencia"].sum())
    df = df[df["haveJurisprudencia"] == False].reset_index()
    return df[["text", "tags"]]


def datas_change(df, datas_to_change=["Data_do_contrato", "Data_dos_fatos"]):
    # AGGREGATE datas to change with generic Datas
    df["tags"] = df["tags"].apply(
        lambda x: [
            tag[:2] + "Datas" if tag[2:] in datas_to_change else tag for tag in x
        ]
    )

    return df.reset_index()


def remove_label_punctuaction(df):
    df["tags"] = df["tags"].apply(lambda tags: [unidecode(label) for label in tags])
    return df


class PreProcessDataset:
    def __init__(self, config):
        self.config = config

    def run(self, df):
        # FILTER TAGS WITH MINIMUM RATIO # removendo abaixo de 0.5%
        # df = utils.filter_entities(df, minimum_entity_ratio=0.005)

        # FILTER SENTENCES WITH ENTITIES
        tags_to_remove = self.config.get("fill_O_tags", "")
        if tags_to_remove:
            df = fill_O_tags(df, tags_to_remove)

        if self.config.get("remove_jurisprudencia_sentence"):
            df = remove_jurisprudencia_sentence(df)

        # Datas_do_contrato e Datas_dos_fatos PARA Datas
        datas_to_change = self.config.get("datas_aggregation")
        if datas_to_change:
            print("Datas Aggretation to Generic Datas", datas_to_change)
            # hardcoded due to business decision
            df = datas_change(df, datas_to_change=datas_to_change)

        # A MUST STEP
        # FILTER MAX_LENGHT SENTENCES
        df = trucate_sentence_max_length(
            df, max_length=self.config.get("max_length_sentence", 512)
        )

        df = remove_label_punctuaction(df)

        return df
