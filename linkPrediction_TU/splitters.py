import torch
import random
import numpy as np


def random_split(dataset, frac_train, frac_valid=0.1, frac_test=0.1, seed=0):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    num_datas = len(dataset)
    random.seed(seed)
    all_idx = list(range(num_datas))
    random.shuffle(all_idx)

    train_idx = all_idx[:int(frac_train * num_datas)]
    valid_idx = all_idx[int(frac_train * num_datas):int(frac_valid * num_datas)
                                                   + int(frac_train * num_datas)]
    test_idx = all_idx[int(frac_valid * num_datas) + int(frac_train * num_datas):]

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == num_datas
    
    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    return train_dataset, valid_dataset, test_dataset