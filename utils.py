from typing import Dict

import numpy as np
import torch


def examplesCompletion(examples, n_entities, neg_ratio=1):
    """
    this function is used to make completion for examples.
    Formly we generate neg_ratio × num of negative examples and make concat.
    """
    pos_neg_group_size = 1 + neg_ratio
    examp = examples.cpu()
    neg_example_replace_head = np.repeat(np.copy(examp), pos_neg_group_size, axis=0)
    neg_example_replace_tail = np.copy(neg_example_replace_head)
    rand_nums1 = np.random.randint(low=1, high=n_entities, size=neg_example_replace_head.shape[0])
    rand_nums2 = np.random.randint(low=1, high=n_entities, size=neg_example_replace_tail.shape[0])

    for i in range(neg_example_replace_head.shape[0] // pos_neg_group_size):
        rand_nums1[i * pos_neg_group_size] = 0
        rand_nums2[i * pos_neg_group_size] = 0

    neg_example_replace_head[:, 0] = (neg_example_replace_head[:, 0] + rand_nums1) % n_entities
    neg_example_replace_tail[:, 2] = (neg_example_replace_tail[:, 2] + rand_nums2) % n_entities

    new_examples = np.concatenate((neg_example_replace_head, neg_example_replace_tail), axis=0)
    return torch.from_numpy(new_examples.astype('int64'))


def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}
