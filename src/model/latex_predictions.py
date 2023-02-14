import numpy as np

red = "\033[91m"
green = "\033[92m"
end = "\033[0m"
yellow = "\033[93m"


def print_wrong_inference(examples):
    text = np.array(examples["text"].copy())
    label = np.array(examples["tags"].copy())
    preds = np.array(examples["predicted_tags"].copy())
    # text_not_read = examples["text_not_read"].copy()

    idx_diff = np.argwhere(label != preds).flatten()

    if len(idx_diff) > 0:
        text[idx_diff] = [
            red
            + word
            + end
            + "("
            + red
            + predicted
            + end
            + ","
            + green
            + target
            + end
            + ")"
            for word, predicted, target in zip(
                text[idx_diff], preds[idx_diff], label[idx_diff]
            )
        ]

    final_text = " ".join(text) + " " + yellow + end  # " ".join(text_not_read) + end

    print(final_text)


def latex_inference(examples):
    text = np.array(examples["text"].copy(), dtype=np.dtype("U100"))
    preds = np.array(examples["predicted_tags"].copy(), dtype=np.dtype("U20"))
    label = np.array(examples["tags"].copy(), dtype=np.dtype("U20"))[: len(preds)]
    # text_not_read = examples['text_not_read'].copy()

    idx_diff = np.argwhere(label != preds).flatten()

    text[idx_diff] = [
        r"\textcolor{red}{%s} ( \textcolor{red}{%s} , \textcolor{green}{%s} )"
        % (word, predicted.replace("_", " "), target.replace("_", " "))
        for word, predicted, target in zip(
            text[idx_diff], preds[idx_diff], label[idx_diff]
        )
    ]

    final_text = " ".join(text)  # +' '+ yellow + " ".join(text_not_read)+end
    final_text = final_text.replace("%", "\%").replace("$", "\$")

    return final_text


def generate_latex_df(df, top_k=50, fname="output"):
    init_doc = """\\documentclass{article}\n
                \\usepackage[utf8]{inputenc}\n
                \\usepackage{xcolor}\n                
                \\usepackage{geometry}\n
                \\geometry{a4paper, total={170mm,257mm}, left=20mm, top=20mm, }\n
                \\begin{document}\n"""
    end_doc = "\n\\end{document}"
    all_text = []

    df["lenght_missed"] = df.apply(
        lambda row: sum(
            [tag != pred for tag, pred in zip(row["tags"], row["predicted_tags"])]
        ),
        axis=1,
    )
    all_text.append("\\section{Top {%s} sentenças em erros de predições}" % (top_k))
    for i, example in (
        df.sort_values("lenght_missed", ascending=False).iloc[:top_k].iterrows()
    ):
        all_text.append(latex_inference(example).replace("&", "\&"))

    all_text = r" \\ \par \vspace{25mm} ".join(all_text)

    document = init_doc + all_text + end_doc
    with open(f"{fname}.tex", "w", encoding="utf-8") as f:
        f.write(document)
