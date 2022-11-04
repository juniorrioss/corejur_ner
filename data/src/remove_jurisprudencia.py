from utils import conll2pandas
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenizer = TreebankWordDetokenizer()


def verify_sentence(text):
    text = detokenizer.detokenize(text)
    # return text in a["text"].tolist()
    return "APELAÇÃO" in text


df = conll2pandas("raw_data/corejur_ner_v19.conll")

df_class = pd.read_csv("raw_data/corejur_classification_v19.csv")
a = df_class[df_class["label"] == "Jurisprudência"]

df["teste"] = df["text"].apply(verify_sentence)
print("Done!")
