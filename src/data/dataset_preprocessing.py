from typing import List
from unidecode import unidecode
import numpy as np
from dataset_feature import undersampling_negative_sentences
import pandas as pd


class PreProcessDataset:
    def __init__(self, config):
        self.config = config

    def filter_null_sentences_from_older_versions(self, df):
        diff_df = (
            len(df) - self.config["length_filter_null_sentences_from_older_versions"]
        )
        older_df = df.iloc[diff_df:]

        older_df["isNullSentences"] = older_df["tags"].apply(
            lambda tags: all([tag == "O" for tag in tags])
        )

        older_df = older_df.query("isNullSentences == False")
        df = pd.concat([df[:diff_df], older_df]).reset_index(drop=True)

        return df

    def fill_O_tags(self, df, tags_to_remove: List):
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

    def remove_jurisprudencia_sentence(self, df):
        # REMOVE JURISPRUDENCIA
        df["haveJurisprudencia"] = df["tags"].apply(lambda x: "B-Jurisprudência" in x)
        print("SENTENÇAS COM JURISPRUDENCIA ", df["haveJurisprudencia"].sum())
        df = df[df["haveJurisprudencia"] == False].reset_index(drop=True)
        return df[["text", "tags"]]

    def datas_change(self, df, datas_to_change=["Data_do_contrato", "Data_dos_fatos"]):
        # AGGREGATE datas to change with generic Datas
        df["tags"] = df["tags"].apply(
            lambda x: [
                tag[:2] + "Datas" if tag[2:] in datas_to_change else tag for tag in x
            ]
        )

        return df.reset_index()

    def remove_label_punctuaction(self, df):
        df["tags"] = df["tags"].apply(lambda tags: [unidecode(label) for label in tags])
        return df

    def run(self, df):

        # Filter negatives samples until a older version
        # Because of extraction pipeline some sentences were not properly annotated
        # If true - Default to v17 (103594)
        if self.config.get("length_filter_null_sentences_from_older_versions", None):
            print(" [ INFO ] FILTERING NULL SENTENCES FROM OLD VERSION")
            df = self.filter_null_sentences_from_older_versions(df)

        # Fill tags to O
        tags_to_remove = self.config.get("fill_O_tags", "")
        if tags_to_remove:
            print(" [ INFO ] FILLING TAGS TO O ", tags_to_remove)

            df = self.fill_O_tags(df, tags_to_remove)

        # Drop all samples with Jurisprudencia tag
        if self.config.get("remove_jurisprudencia_sentence"):
            print(" [ INFO ] REMOVING JURISPRUDENCIA SENTENCES")

            df = self.remove_jurisprudencia_sentence(df)

        # Specific Data to Generic Data
        # Because of the low occurrence
        datas_to_change = self.config.get("datas_aggregation")
        if datas_to_change:
            print(" [ INFO ] DATAS AGGREGATION TO GENERIC DATAS", datas_to_change)
            df = self.datas_change(df, datas_to_change=datas_to_change)

        # For logging process (MLFLOW) is recommended only unicode characters
        # Convert é -> e
        # Convert petição -> peticao
        df = self.remove_label_punctuaction(df)

        return df
