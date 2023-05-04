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
        device=device
    )
    # Load from checkpoint
    # assert os.path.isdir(args.model_store_path)
    # model_load_path = os.path.join(args.model_store_path, args.model_postfix)
    # assert os.path.isdir(model_load_path)
    # last_checkpoint = sorted([
    #     (int(re.search("epoch_([0-9]*)", f).group(1)), int(re.search("step_([0-9]*)", f).group(1)), f) for f in os.listdir(model_load_path) if f.endswith(".pt")], reverse=True
    # )[0][2]
    # model_load_path = os.path.join(model_load_path, last_checkpoint)
    # model.load_state_dict(torch.load(model_load_path, map_location=device))
    # model.device = device
    model = model.to(device)

    # Load data
    with open(args.test_data, "r", encoding='UTF-8') as file:
        test_data = file.readlines()
        test_data = [Tree.fromstring(x) for x in test_data]
    test_data_flattened = [TreebankWordDetokenizer().detokenize(x.leaves()) for x in test_data]
    test_loader = DataLoader(test_data_flattened, shuffle=False, batch_size=args.batch_size)

    model.eval()
    epoch_size = len(test_loader)

    pred_trees = []
    for i, batch in enumerate(tqdm(test_loader, total=epoch_size)):
        sents = batch
        with torch.no_grad():
            pred_batch = model.parse(sents)
            pred_batch_trees = []
            for tree in pred_batch:
                pred_batch_trees.append(tree.pformat(margin=9999999))
        
        pred_trees.extend(pred_batch_trees)
        
    with open(os.path.join(model_store_path, "parse.txt"), "w", encoding="UTF-8") as file:
        file.write("\n".join(pred_trees) + "\n")

if __name__ == "__main__":
    parser = ArgumentParser()
    # Dataset
    parser.add_argument("--test_data", required=True, help="Test set(PTB tree, linebreaked)")

    # Base model/checkpoint configuration
    parser.add_argument("--model_id", required=False, default="bert-base-uncased", help="Base model for DIORA architecture.")
    parser.add_argument("--arch", required=False, default="diora", choices=["diora", "dora"], help="Recursive autoencoder architecture")

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=8, help="training batch size")
    parser.add_argument("--remove_trivial_spans", required=False, action="store_true", help="Remove trivial span when evaluating F1 score.")

    # PyTorch/CUDA configuration
    parser.add_argument("--gpu", type=int, default=0, help="CUDA index for training")
    parser.add_argument("--torch_seed", type=int, default=0, help="torch_seed() value")

    # Checkpoint configs
    parser.add_argument("--model_store_path", required=False, default='checkpoints', help="Directory to store model checkpoints.")
    parser.add_argument("--model_postfix", required=False, help="Name for the model. defaulted to {model_id}-arch")
    parser.add_argument("--secure", required=False, action="store_true", help="")

    args = parser.parse_args()

    # Post-modification of args

    main(args)