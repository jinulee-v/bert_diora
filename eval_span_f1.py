from argparse import ArgumentParser
import json
from tqdm import tqdm
import os, sys
import logging
import re
import gc

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from bert_diora.models import BertDiora, BertDora
from bert_diora.utils import TokenizedLengthSampler

from nltk import Tree
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer

def main(args):
    # Set torch
    torch.manual_seed(args.torch_seed)

    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Make checkpoint/log directory
    model_store_path = os.path.join(args.model_store_path, args.model_postfix)
    try:
        os.mkdir(model_store_path)
    except FileExistsError:
        if args.secure:
            prompt = input("WARNING: overwriting directory " + model_store_path + ". Continue? (y/n)")
            if prompt != "y":
                exit()

    # Init logger
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    if not args.secure:
        # Remove original log file
        if os.path.exists(os.path.join(model_store_path, "eval_span_f1.log")):
            os.remove(os.path.join(model_store_path, "eval_span_f1.log"))
    file_handler = logging.FileHandler(os.path.join(model_store_path, "eval_span_f1.log"))
    file_handler.setFormatter(formatter)
    logger = logging.getLogger('')
    logger.handlers.clear()
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    # Log basic info
    logger.info("Training arguments:")
    for arg, value in sorted(vars(args).items()):
        logger.info("- %s: %r", arg, value)
    logger.info("")
    
    Arch = {
        "diora": BertDiora,
        "dora": BertDora,
    }[args.arch]
    model = Arch(
        args.model_id,
        freeze=not args.unfreeze,
        device=device
    ).to(device)

    # Load data
    with open(args.test_data, "r", encoding='UTF-8') as file:
        test_data = file.readlines()
        test_data = [Tree.fromstring(x) for x in test_data]
    test_data_flattened = [TreebankWordDetokenizer().detokenize(x.leaves()) for x in test_data]
    test_loader = DataLoader(test_data_flattened, batch_sampler=TokenizedLengthSampler(test_data_flattened, args.batch_size, seed=args.torch_seed))

    model.eval()
    epoch_size = len(test_loader)

    pred_trees = []
    gold_trees = []

    if args.remove_trivial_spans:
        def spans(tree, max_len):
            """
            Convert a parse tree into a list of spans. Remove trivial spans, that is length==max_len or length==1
            """
            if isinstance(tree, Tree):
                if len(tree.leaves()) == max_len:
                    return [].extend([spans(child) for child in tree])
                elif len(tree.leaves()) == 1:
                    return []
                else:
                    return [tree.leaves()].extend([spans(child) for child in tree])
            else:
                return []
    else:
        def spans(tree, _):
            """
            Convert a parse tree into a list of spans.
            """
            if isinstance(tree, Tree):
                return [tree.leaves()].extend([spans(child) for child in tree])
            else:
                return []

    for i, batch in enumerate(tqdm(test_loader, total=epoch_size)):
        sents, idx = batch
        with torch.no_grad():
            pred_batch = model.parse(sents)
            pred_batch = [spans(tree, len(tree.leaves())) for tree in pred_batch]
            # try:
            #     trees = model.parse(batch)
            # except Exception as e:
            #     logger.warning(str(e))
            #     logger.info("Exception occured; skip batch")
            #     gc.collect()
            #     torch.cuda.empty_cache()
            #     gc.collect()
            #     torch.cuda.empty_cache()

        gold_batch = [spans(test_data[i], len(test_data[i].leaves())) for i in idx]
        
        pred_trees.extend(pred_batch)
        gold_trees.extend(gold_batch)

    # Evaluate F1 score
    p_list = []
    r_list = []
    f1_list = []
    for pred_tree, gold_tree in zip(pred_trees, gold_trees):
        tp = 0
        for pred_span in pred_tree:
            if pred_span in gold_tree:
                tp += 1
        fn = len(gold_tree) - tp
        fp = len(pred_tree) - tp
        
        # append
        p_list.append(tp / (tp+fp))
        r_list.append(tp / (tp+fn))
        f1_list.append((2*tp) / (2*tp + fp + fn))
    
    sorted_f1_list = sorted(f1_list)
    logger.info("F1 score")
    logger.info(f"  avg: {sum(f1_list) / len(f1_list)}")
    logger.info(f"  min: {sorted_f1_list[0]}")
    logger.info(f"  q1 : {sorted_f1_list[1*len(f1_list)/4]}")
    logger.info(f"  q2 : {sorted_f1_list[2*len(f1_list)/4]}")
    logger.info(f"  q3 : {sorted_f1_list[3*len(f1_list)/4]}")
    logger.info(f"  max: {sorted_f1_list[-1]}")

        

if __name__ == "__main__":
    parser = ArgumentParser()
    # Dataset
    parser.add_argument("--test_data", required=True, help="Test set(PTB tree, linebreaked)")

    # Base model/checkpoint configuration
    parser.add_argument("--model_id", required=False, default="bert-base-uncased", help="Base model for DIORA architecture.")
    parser.add_argument("--arch", required=False, default="diora", choices=["diora", "dora"], help="Recursive autoencoder architecture")

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=8, help="training batch size")
    parser.add_argument("--remove_trivial_spans", required=False, default=True, help="Remove trivial span when evaluating F1 score.")

    # PyTorch/CUDA configuration
    parser.add_argument("--gpu", type=int, default=0, help="CUDA index for training")
    parser.add_argument("--torch_seed", type=int, default=0, help="torch_seed() value")

    # Checkpoint configs
    parser.add_argument("--model_store_path", required=False, default='checkpoints', help="Directory to store model checkpoints.")
    parser.add_argument("--model_postfix", required=False, help="Name for the model. defaulted to {model_id}-arch")
    parser.add_argument("--secure", required=False, action="store_true", help="")

    args = parser.parse_args()

    # Post-modification of args

    if args.model_postfix is None:
        args.model_postfix = args.model_id + '-' + args.arch

    main(args)