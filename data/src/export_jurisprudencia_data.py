from utils import conll2pandas
import json
from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenizer = TreebankWordDetokenizer()
texts = []
df = conll2pandas("raw_data/corejur_ner_v19.conll")
df["haveJurisprudencia"] = df["tags"].apply(lambda x: "B-Jurisprudência" in x)

for i in range(len(df[df["haveJurisprudencia"]])):
    text = df[df["haveJurisprudencia"]].iloc[i]["text"]
    text = detokenizer.detokenize(text)
    texts.append({"text": text, "label": "jurisprudencia"})

with open("jurisprudencia_sentences.json", "w") as f:
    json.dump(texts, f, ensure_ascii=False, indent=4)
print("Done!")
