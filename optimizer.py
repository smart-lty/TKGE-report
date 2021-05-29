import tqdm
import torch
from torch import nn
from torch import optim
from modelAndRegularizer.regularizer import Regularizer


class Optimizer(object):
    def __init__(
            self, model,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True, device: str = 'cuda'
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = device

    def epoch(self, examples: torch.LongTensor):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[b_begin:b_begin + self.batch_size].to(self.device)
                predictions, factors, time = self.model.forward(input_batch)
                truth = input_batch[:, 2]
                l_fit = loss(predictions, truth)
                l_reg = self.emb_regularizer.forward(factors)
                l_time = torch.zeros_like(l_reg)
                if time is not None:
                    l_time = self.temporal_regularizer.forward(time)
                l = l_fit + l_reg + l_time
                self.optimizer.zero_grad()

                l.backward()

                if torch.any(torch.isnan(self.optimizer.param_groups[0]['params'][2].grad)):
                    a = self.optimizer.param_groups[0]['params'][2].grad
                    self.optimizer.param_groups[0]['params'][2].grad = torch.where(torch.isnan(a),
                                                                                   torch.full_like(a, 0), a)
                self.optimizer.step()

                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(
                    loss=f'{l_fit.item():.5f}',
                    reg=f'{l_reg.item():.5f}',
                    cont=f'{l_time.item():.5f}'
                )
