import os
import json
from nltk import Tree

TRAIN_TREE_MAX_LEN=20
# TRAIN_TREE_MAX_LEN=9999999

def snli_to_nltk_tree(text):
    tree = text.split(' ')
    tree = [(f'( _ {token})' if token not in '()' else token) for token in tree]
    tree = ' '.join(tree)
    return tree


BASE = "data/"
DATASETS = ["multinli_1.0", "snli_1.0"]
FILE = {
    "multinli_1.0": {
        "train": "multinli_1.0_train.jsonl",
        "dev": "multinli_1.0_dev_matched.jsonl",
        "test": None
    },
    "snli_1.0": {
        "train": "snli_1.0_train.jsonl",
        "dev": "snli_1.0_dev.jsonl",
        "test": "snli_1.0_test.jsonl",
    },
}
for split in ["train", "dev", "test"]:
    cache = set()
    raw_data = []
    for dataset in DATASETS:
        file = FILE[dataset][split]
        if file:
            with open(os.path.join(BASE, dataset, file)) as f:
                lines = f.read().splitlines()
                for line in lines:
                    data = json.loads(line) # Read JSONL line

                    if data["sentence1"] not in cache:
                        tree = data["sentence1_binary_parse"]
                        tree = snli_to_nltk_tree(tree)
                        tree = Tree.fromstring(tree)
                        cache.add(data["sentence1"])
                        if split == "test":
                            raw_data.append(tree.pformat(margin=9999999999))
                        elif len(tree.leaves()) <= TRAIN_TREE_MAX_LEN:
                            raw_data.append(' '.join(tree.leaves()))

                    if data["sentence2"] not in cache:
                        tree = data["sentence2_binary_parse"]
                        tree = snli_to_nltk_tree(tree)
                        tree = Tree.fromstring(tree)
                        cache.add(data["sentence2"])
                        if split == "test":
                            raw_data.append(tree.pformat(margin=9999999999))
                        elif len(tree.leaves()) <= TRAIN_TREE_MAX_LEN:
                            raw_data.append(' '.join(tree.leaves()))
    with open(f"data/nli_{split}.txt", "w", encoding="UTF-8") as split_file:
        split_file.write("\n".join(raw_data) + "\n")