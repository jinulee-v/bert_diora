from nltk import Tree
from nltk.tree.transforms import chomsky_normal_form
import os

SOURCE = "data/ptb/wsj/"
DESTINATION = "data/ptb/wsj_bin/"
FACTOR = "right"
# FACTOR = "left"

def ptb_remove_labels(tree: Tree, delete_labels=["-NONE-", ",", ":", "``", "''"], peel_labels=["TOP"]):
    """
    Peel label:
        DELETE_LABEL TOP
    Delete label:
        DELETE_LABEL -NONE-
        DELETE_LABEL ,
        DELETE_LABEL :
        DELETE_LABEL ``
        DELETE_LABEL ''
    """
    # Postfix DFS of the tree.
    if isinstance(tree, Tree):
        # is a nltk.Tree

        # if node to remove or node to peel off,
        if tree.label() in delete_labels:
            return None
        elif tree.label() in peel_labels:
            if len(tree) != 1:
                print(tree)
                raise ValueError()
            return ptb_remove_labels(tree[0])
        
        # else, recursively traverse the tree
        new_children = []
        for child in tree:
            child_cleaned = ptb_remove_labels(child, delete_labels)
            if child_cleaned is not None:
                new_children.append(child_cleaned)
        # Recursively delete a node if all of its child node is empty
        if len(new_children) == 0:
            return None
        else:
            return Tree(tree.label(), new_children)
    else:
        return tree

if __name__ == "__main__":
    print(ptb_remove_labels(Tree.fromstring("(TOP (S (VP (VBD reported) (SBAR (-NONE- 0) (S (-NONE- *T*-1)))) (. .)))")))
    for folder in os.listdir(SOURCE):
        print(folder)
        os.makedirs(DESTINATION + folder, exist_ok=True)
        for file in os.listdir(SOURCE + folder):
            with open(os.path.join(SOURCE, folder, file), "r") as rf:
                with open(os.path.join(DESTINATION, folder, file), "w", encoding="UTF-8") as wf:
                    for tree in rf.read().split("\n\n"):
                        if not tree.split():
                            continue
                        tree = ptb_remove_labels(Tree.fromstring(tree)) # Remove traces/puncts
                        chomsky_normal_form(tree, FACTOR, horzMarkov=2)
                        wf.write(str(tree) + "\n\n")