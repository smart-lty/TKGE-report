from dataset.datasets import TemporalDataset
from modelAndRegularizer.model import TRESCAL
import torch
import json
import argparse
from torch import optim
from modelAndRegularizer.regularizer import *
from optimizer import Optimizer
from utils import avg_both
import re

dataset = ['ICEWS14', 'ICEWS05-15']
parser = argparse.ArgumentParser(
    description="Tensor Factorization for Knowledge Graph Completion"
)

parser.add_argument(
    '--dataset', choices=dataset,
    help="Dataset in {}".format(dataset)
)

optimizers = ['Adagrad', 'Adam', 'SGD']

parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)

parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid', default=10, type=float,
    help="Number of epochs before valid."
)
parser.add_argument(
    '--rank', default=100, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=128, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--reg', default=0.1, type=float,
    help="Regularization weight"
)

parser.add_argument(
    '--time_reg', default=0.1, type=float,
    help="Regularization weight"
)

parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)
args = parser.parse_args()


dataset = TemporalDataset(args.dataset)
examples = torch.from_numpy(dataset.get_train().astype('int64'))

model = TRESCAL(dataset.get_shape(), rank=args.rank)
for name, p in model.named_parameters():
    print(name)
emb_reg = TDURA_RESCAL(args.reg)
time_reg = Lambda3(args.time_reg)

device = 'cuda'
model.to(device)
emb_reg.to(device)
time_reg.to(device)


optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()
optimizer = Optimizer(model, emb_reg, time_reg, optim_method, args.batch_size, device=device)

for epoch in range(args.max_epochs):
    print("Epoch: {}".format(epoch + 1))
    model.train()
    optimizer.epoch(examples)

    if (epoch + 1) % args.valid == 0:
        valid, test, train = [
            avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
            for split in ['valid', 'test', 'train']
        ]

        print("\t TRAIN: ", train)
        print("\t VALID: ", valid)
        print("\t TEST: ", test)

test = avg_both(*dataset.eval(model, 'test', 50000))
print("\t TEST : ", test)



