def undersampling_entity(df, undersampling_tags, ratio_to_remove=0.5):
    """Apply undersampling with specific tags

    Args:
        df (pd.dataFrame): Dataframe object
        undersampling_tags (List[String]): A List of Tags to apply undersampling
        ratio_to_remove (float, optional): Undersampling Ratio. Defaults to 0.5.

    Returns:
        pd.dataFrame: Dataframe object with tags undersampled
    """

    # sentences with at least one TAG
    df["withEntity"] = df["tags"].apply(
        lambda tags: any([tag[2:] in undersampling_tags for tag in tags])
    )

    df2 = df[df["withEntity"]].sample(frac=ratio_to_remove, random_state=0)

    # todos os index que não estão nos retirados
    dataset_filtered = df[~df.index.isin(df2.index)]
    # remover a coluna criada e resetar os indexes
    dataset_filtered = dataset_filtered.drop("withEntity", axis=1)

    return dataset_filtered.reset_index(drop=True)


def undersampling_negative_sentences(df, ratio_to_remove=0.8):
    """Apply undersampling in sentences with full tags 'O'

    Args:
        df (pd.Dataframe): dataframe object
        ratio_to_remove (float, optional): undersampling Ratio. Defaults to 0.8.

    Returns:
        pd.dataFrame: DataFrame with undersampling
    """

    # sentences with ALL TAGS '0'
    df["nullSentences"] = df["tags"].apply(
        lambda tags: all([tag == "O" for tag in tags])
    )

    df2 = df[df["nullSentences"]].sample(frac=ratio_to_remove, random_state=0)

    # todos os index que não estão nos retirados
    dataset_filtered = df[~df.index.isin(df2.index)]
    # remover a coluna criada e resetar os indexes
    dataset_filtered = dataset_filtered.drop("nullSentences", axis=1)

    return dataset_filtered.reset_index(drop=True)


class FeatureEnginnering:
    def __init__(self, config):
        self.config = config

    def run(self, df):
        # UNDERSAMPLING SENTENCES WITH FULL 'O' TAGS
        # ONLY IN TRAIN
        if self.config.get("undersampling_negative_sentences"):
            print("UNDERSAMPLING NEGATIVE SENTENCES")
            df = undersampling_negative_sentences(
                df,
                ratio_to_remove=self.config.get(
                    "ratio_of_undersample_negative_sentences", 0.8
                ),
            )

        # UNDERSAMPLING TAGs
        undersampling_tags = self.config.get("undersampling_tags")
        if undersampling_tags:
            print("Undersampling tags ", undersampling_tags)
            df = undersampling_entity(
                df,
                undersampling_tags=undersampling_tags,
                ratio_to_remove=self.config.get("ratio_of_undersample_tags", 0.5),
            )

        return df
