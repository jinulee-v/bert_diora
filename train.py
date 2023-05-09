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

from bert_diora.models import BertDiora
from bert_diora.utils import TokenizedLengthSampler

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
        if os.path.exists(os.path.join(model_store_path, "train.log")):
            os.remove(os.path.join(model_store_path, "train.log"))
    file_handler = logging.FileHandler(os.path.join(model_store_path, "train.log"))
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
    }[args.arch]
    model = Arch(
        args.model_id,
        freeze=not args.unfreeze,
        device=device,
        loss=args.loss,
        loss_margin_k=args.loss_margin_k,
        loss_margin_lambda=args.loss_margin_lambda
    ).to(device)
    logger.info(model)
    resume_training = False
    if args.from_checkpoint is not None:
        # Fine-tune from a local checkpoint
        assert os.path.isdir(args.model_store_path)
        model_load_path = os.path.join(args.model_store_path, args.from_checkpoint)
        assert os.path.isdir(model_load_path)
        last_checkpoint = sorted([
            (int(re.search("epoch_([0-9]*)", f).group(1)), int(re.search("step_([0-9]*)", f).group(1)), f) for f in os.listdir(model_load_path) if f.endswith(".pt")], reverse=True
        )[0][2]
        model_load_path = os.path.join(model_load_path, last_checkpoint)
        model.load_state_dict(torch.load(model_load_path, map_location=device))
        model.device = device
        model = model.to(device)
        if args.from_checkpoint == args.model_postfix:
            # If resume training from an error,
            resume_training=True
            resume_epoch = int(re.search("epoch_([0-9]*)", last_checkpoint).group(1))
            resume_step = int(re.search("step_([0-9]*)", last_checkpoint).group(1))
            resume_epoch_step = (resume_epoch, resume_step)
            logger.info(f"Resume training from checkpoint: epoch {resume_epoch}, step {resume_step}")

    # Load data
    with open(args.train_data, "r", encoding='UTF-8') as file:
        train_data = file.read().splitlines()
    with open(args.dev_data, "r", encoding='UTF-8') as file:
        dev_data = file.read().splitlines()
    train_loader = DataLoader(train_data, batch_sampler=TokenizedLengthSampler(train_data, args.batch_size, seed=args.torch_seed))
    dev_loader = DataLoader(dev_data, batch_sampler=TokenizedLengthSampler(dev_data, args.batch_size, seed=args.torch_seed))

    # Define optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)
    optimizer.zero_grad()

    min_loss = 1e+10
    early_stop_count = 0
    loss = 0
    for epoch in range(args.epoch):  # loop over the dataset multiple times
        if resume_training:
            # If resume training from an error, skip to the halted epoch/step
            if (epoch, len(train_loader) * 100) <= resume_epoch_step: 
                continue
        logger.info(f"< epoch {epoch} >")
        # Train phase
        model.train()
        epoch_size = len(train_loader)
        for i, batch in enumerate(tqdm(train_loader, total=epoch_size)):
            if resume_training:
                # If resume training from an error, skip to the halted epoch/step
                if (epoch, i) <= resume_epoch_step: 
                    continue

            sent = batch
            # try:
            if True:
                # forward + backward + optimize
                loss = model(sent)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if i % args.update_freq == args.update_freq - 1 or i == epoch_size-1:
                    optimizer.step()
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    loss = 0
            # except Exception as e:
            #     logger.warning(str(e))
            #     logger.info("Exception occured; returning to training")
            #     gc.collect()
            #     torch.cuda.empty_cache()
            #     gc.collect()
            #     torch.cuda.empty_cache()
            # finally:
            #     if i % args.update_freq == args.update_freq - 1 or i == epoch_size-1:
            #         loss = 0

            if i % args.log_interval == args.log_interval-1 or i == epoch_size-1:
                # Eval phase (on dev set)
                model.eval()
                with torch.no_grad():
                    total = len(dev_data)
                    dev_loss = 0
                    first_batch=True
                    for dev_batch in dev_loader:
                        dev_sents = dev_batch
                        if first_batch:
                            # test_input = gen_inputs[0]
                            # test_outputs = model.generate([test_input])[0]
                            dev_loss += (model(dev_sents)).item() * len(dev_sents)
                            first_batch=False
                        else:
                            dev_loss += (model(dev_sents)).item() * len(dev_sents)
                logger.info("=================================================")
                logger.info(f"epoch {epoch}, step {i}")
                logger.info(f"dev loss = {dev_loss/total}")
                logger.info("")
                # logger.info("Test generation result")
                # logger.info(f"input: {test_input}")
                # logger.info(f"output:")
                # for test_output in test_outputs:
                #     logger.info(f"  {test_output}")
                # logger.info("")
                if dev_loss/total < min_loss:
                    logger.info(f"Updating min_loss = {min_loss} -> {dev_loss/total}")
                    min_loss = dev_loss / total
                    logger.info("Save model checkpoint because reduced loss...")
                    name = f"Model_{args.model_postfix}_epoch_{epoch}_step_{i+1}.pt"
                    torch.save(model.state_dict(), os.path.join(model_store_path, name))
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                    logger.info(f"Min loss not updated for {early_stop_count} validation routines...")
                    if early_stop_count >= args.early_stop:
                        logger.info("Early stopping....")
                        return
                logger.info("=================================================")

if __name__ == "__main__":
    parser = ArgumentParser()
    # Dataset
    parser.add_argument("--train_data", required=True, help="Training set(raw text, linebreaked)")
    parser.add_argument("--dev_data", required=True, help="Validation set(raw text, linebreaked)")

    # Base model/checkpoint configuration
    parser.add_argument("--from_checkpoint", required=False, default=None, help="Pretrained checkpoint to load and resume training.")
    parser.add_argument("--model_id", required=False, default="bert-base-uncased", help="Base model for DIORA architecture.")
    parser.add_argument("--arch", required=False, default="diora", choices=["diora", "dora"], help="Recursive autoencoder architecture")
    parser.add_argument("--loss", required=False, default="cossim", choices=["cossim", "token_ce", "token_margin"], help="Loss function to apply to DIORA")
    parser.add_argument("--loss_margin_k", type=int, required=False, default=50, help="(loss=token_margin) How many negative tokens to compare")
    parser.add_argument("--loss_margin_lambda", type=float, required=False, default=1.0, help="(loss=token_margin) max-margin value")
    parser.add_argument("--max_grad_norm", type=float, required=False, default=5, help="Max L2 norm for radient cipping")

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=8, help="training batch size")
    parser.add_argument("--update_freq", type=int, default=1, help="gradient accumulation for virtually larger batches")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate (default: Adam optimizer)")
    parser.add_argument("--epoch", type=int, default=5, help="epoch count")
    parser.add_argument("--unfreeze", action='store_true', help="If set, we also train the underlying parameter too.")

    parser.add_argument("--log_interval", type=int, default=20000, help="validating / checkpoint saving interval. Validates at the end of each epoch for default.")
    parser.add_argument("--early_stop", type=int, default=4, help="if valid loss does not decrease for `early_stop` validations, stop training.")

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
        short_model_name = args.model_id.split("-")[0].split("_")[0]
        args.model_postfix = short_model_name + '-' + args.arch + "-" + args.loss

    main(args)