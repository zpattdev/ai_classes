import argparse
import sys
import os
import time
from datetime import datetime

import math
import torch
from torch import nn, optim
from tqdm import tqdm

from submission import ParserModel, minibatches, load_and_preprocess_data, AverageMeter, train


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--debug', action='store_true', default=False)
    args_parser.add_argument("--device", type=str, default="gpu")
    args_parser.add_argument('--compile', action='store_true', default=False)
    args_parser.add_argument("--backend", type=str, default="inductor", choices=['inductor', 'aot_eager', 'cudagraphs'])

    args = args_parser.parse_args()

    if args.device == "gpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        args.device = torch.device("mps")
    elif args.device == "gpu" and torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    print("Using device: ", args.device)

    assert (torch.__version__ >= "1.0.0"), "Please install torch version 1.0.0 or greater"

    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(args.debug)

    start = time.time()
    model = ParserModel(embeddings)

    if(args.compile == True):
        try:
            model = torch.compile(model, backend=args.backend)
            print(f"Parser model compiled")
        except Exception as err:
            print(f"Model compile not supported: {err}")

    model.to(args.device)

    parser.model = model
    print("took {:.2f} seconds\n".format(time.time() - start))

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")

    output_dir = "run_results_(soln)/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.weights"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005, device=args.device)

    if not args.debug:
        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Restoring the best model weights found on the dev set")
        parser.model.load_state_dict(torch.load(output_path))
        print("Final evaluation on test set",)
        parser.model.eval()
        UAS, dependencies = parser.parse(test_data, args.device)
        print("- test UAS: {:.2f}".format(UAS * 100.0))
        print("Done!")
