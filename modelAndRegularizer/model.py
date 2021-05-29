from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch
from torch import nn
from tqdm import tqdm


class KBCModel(nn.Module, ABC):
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int, int], List[int]],
            batch_size: int = 1000):
        """
        Returns filtered ranking for each queries.
        即获取每个实体的排名rank
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :return:
        """
        ranks = torch.ones(len(queries))
        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    scores, _, _ = self.forward(these_queries)
                    targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                        filter_out += [queries[b_begin + i, 2].item()]  # Add the tail of this (b_begin + i) query
                        scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                    bar.update(batch_size)
        return ranks


class TRESCAL(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int, no_time_emb=False, init_size: float = 1e-2
    ):
        super(TRESCAL, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], rank),
            nn.Embedding(sizes[1], rank * rank),
        ])

        nn.init.xavier_uniform_(tensor=self.embeddings[0].weight)
        nn.init.xavier_uniform_(tensor=self.embeddings[1].weight)

        self.lhs = self.embeddings[0]
        self.rel = self.embeddings[1]
        self.rhs = self.embeddings[0]
        self.activate = torch.sin
        self.te = nn.Embedding(sizes[3], rank)
        self.W = nn.Embedding(sizes[0], rank)
        self.b = nn.Embedding(sizes[0], rank)

        nn.init.xavier_uniform_(tensor=self.te.weight)
        nn.init.xavier_uniform_(tensor=self.W.weight)
        nn.init.xavier_uniform_(tensor=self.b.weight)

    def get_time_embedding(self, entity, time):
        entity_eb = self.lhs(entity)
        time_eb = self.te(time)
        w = self.W(time)
        b = self.b(time)
        assert w.shape == b.shape
        return entity_eb * self.activate(w * time_eb + b)

    def forward(self, x):
        sub, rela, obj, time = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        time_eb = self.te(time)
        lhs = self.lhs(sub) * self.activate(self.W(sub) * time_eb + self.b(sub))
        rel = self.rel(x[:, 1]).reshape(-1, self.rank, self.rank)
        objs = self.lhs(obj) * self.activate(self.W(obj) * time_eb + self.b(obj))
        rhs1 = self.rhs.weight.unsqueeze(0).repeat(obj.shape[0], 1, 1)  # n1 × n × r
        rhs = rhs1 * self.activate(self.W.weight.unsqueeze(0) * time_eb.unsqueeze(1) + self.b.weight.unsqueeze(0))
        predictions = torch.bmm(torch.bmm(lhs.unsqueeze(1), rel), rhs.transpose(1, 2)).squeeze(1)
        if torch.any(torch.isnan(predictions)):
            print("w", torch.any(torch.isnan(self.W.weight.unsqueeze(0))))
            print("t", torch.any(torch.isnan(time_eb.unsqueeze(1))))
            print("W*T", torch.any(torch.isnan(self.W.weight.unsqueeze(0) * time_eb.unsqueeze(1))))
            print("b", torch.any(torch.isnan(self.b.weight.unsqueeze(0))))
            print("activate", torch.any(torch.isnan(self.activate(self.W.weight.unsqueeze(0) * time_eb.unsqueeze(1) + self.b.weight.unsqueeze(0)))))
            print("lhs", torch.any(torch.isnan(torch.bmm(lhs.unsqueeze(1), rel))))
            raise ValueError
        return predictions, [(lhs, rel, objs)], self.te.weight
