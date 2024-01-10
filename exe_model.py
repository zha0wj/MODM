import argparse
import torch
import datetime
import json
import yaml
import os
import torch.nn as nn

from main_model import TSB_eICU, ResNet
from dataset import get_dataloader, get_ext_dataloader
from utils import train, evaluate, test
from rtdl import FTTransformer

parser = argparse.ArgumentParser(description="MODM")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=9999)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

train_loader, valid_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
)

ext_loader = get_ext_dataloader(seed=0, batch_size=config["train"]["batch_size"], missing_ratio=0)

model = TSB_eICU(config, args.device).to(args.device)
model1 = FTTransformer.make_baseline(
    n_num_features=41,
    cat_cardinalities=[2, 6],
    n_blocks=32,
    d_token=32,
    attention_dropout=0.1,
    ffn_d_hidden=16,
    ffn_dropout=0.1,
    residual_dropout=0.1,
    d_out=2,
).to(torch.device("cuda:0"))


print('')

if args.modelfolder == "":
    train(
        model,
        model1,
        config["train"],
        train_loader,
        valid_loader=valid_loader
    )

test(model, model1, config["train"], ext_loader)

