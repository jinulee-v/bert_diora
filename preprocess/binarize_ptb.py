from nltk import Tree
from nltk.tree.transforms import chomsky_normal_form
import os

SOURCE = "data/ptb/wsj/"
DESTINATION = "data/ptb/wsj_bin/"
FACTOR = "right"
# FACTOR = "left"

if __name__ == "__main__":
    for folder in os.listdir(SOURCE):
        print(folder)
        os.makedirs(DESTINATION + folder, exist_ok=True)
        for file in os.listdir(SOURCE + folder):
            with open(os.path.join(SOURCE, folder, file), "r") as rf:
                with open(os.path.join(DESTINATION, folder, file), "w", encoding="UTF-8") as wf:
                    for tree in rf.read().split("\n\n"):
                        if not tree.split():
                            continue
                        tree = Tree.fromstring(tree)
                        chomsky_normal_form(tree, FACTOR)
                        wf.write(str(tree) + "\n\n")