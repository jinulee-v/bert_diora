import os
from nltk import Tree
from nltk.tokenize.treebank import TreebankWordDetokenizer

WSJ_SPLIT = {
    "train": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", 
              "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
              "20", "21"],
    "dev": ["22"],
    "test": ["23"]
}
detok = TreebankWordDetokenizer()

BASE = "data/ptb/wsj_bin"
for split in ["train", "dev", "test"]:
    raw_data = []
    for subset in WSJ_SPLIT[split]:
        for file in os.listdir(os.path.join(BASE, subset)):
            if file.endswith(".parse"):
                with open(os.path.join(BASE, subset, file)) as f:
                    trees = f.read().split("\n\n")
                    trees = [tree for tree in trees if len(tree.strip())]
                    for tree in trees:
                        tree = Tree.fromstring(tree) # convert to NLTK tree
                        if split == "test":
                            tree = tree.pformat(margin=9999999999)
                            raw_data.append(tree)
                        else:
                            tree = tree.leaves() # and to list of tokens
                            tree = detok.detokenize(tree, convert_parentheses=True) # and finally to detoked string
                            raw_data.append(tree)
    with open(f"data/ptb_{split}.txt", "w", encoding="UTF-8") as split_file:
        split_file.write("\n".join(raw_data) + "\n")