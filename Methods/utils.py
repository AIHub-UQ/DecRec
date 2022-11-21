import numpy as np
import torch
import torch.nn.functional as F

def mini_batch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size')

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def get_label(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        ground_truth = test_data[i]
        predict_top_k = pred_data[i]
        pred = list(map(lambda x: x in ground_truth, predict_top_k))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def precision_recall_at_k(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = right_pred / recall_n
    precis = right_pred / precis_n
    return {'recall': recall, 'precision': precis}


def ndcg_at_k(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return ndcg


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def uniform_sample_bpr(dataset):
    user_num = dataset.train_data_size
    users = np.random.randint(0, dataset.num_user, user_num)
    all_pos = dataset.all_pos
    result = []
    for i, user in enumerate(users):
        user_pos = all_pos[user]
        if len(user_pos) == 0:
            continue
        pos_index = np.random.randint(0, len(user_pos))
        pos_item = user_pos[pos_index]
        while True:
            neg_item = np.random.randint(0, dataset.num_item)
            if neg_item in user_pos:
                continue
            else:
                break
        result.append([user, pos_item, neg_item])
    return torch.tensor(result)


def uniform_sample_single_user(dataset, user):
    result = []
    user_pos = dataset.all_pos[user]
    for i in user_pos:
        pos_index = np.random.randint(0, len(user_pos))
        pos_item = user_pos[pos_index]
        while True:
            neg_item = np.random.randint(0, dataset.num_item)
            if neg_item in user_pos:
                continue
            else:
                break
        result.append([user, pos_item, neg_item])
    return torch.tensor(result)


def uniform_sample_cluster_user(dataset, cluster_user):
    user_num = 0
    for user in cluster_user:
        user_num += len(dataset.all_pos[user])
    users = cluster_user[np.random.randint(0, len(cluster_user), user_num)]
    all_pos = dataset.all_pos
    result = []
    for i, user in enumerate(users):
        user_pos = all_pos[user]
        if len(user_pos) == 0:
            continue
        pos_index = np.random.randint(0, len(user_pos))
        pos_item = user_pos[pos_index]
        while True:
            neg_item = np.random.randint(0, dataset.num_item)
            if neg_item in user_pos:
                continue
            else:
                break
        result.append([user, pos_item, neg_item])
    return torch.tensor(result)


def uniform_sample_2(dataset):
    result = []
    for user in range(dataset.num_user):
        for index, pos_item in enumerate(dataset.all_pos[user]):
            result.append([user, pos_item, index])
    return torch.tensor(result)


def get_all_history(dataset):
    all_history = [torch.tensor(dataset.all_pos[user]) for user in range(dataset.num_user)]
    max_len = max([len(_) for _ in all_history])
    all_history = [F.pad(x, pad=(0, max_len - x.numel()), mode='constant', value=-1) for x in all_history]
    all_history = torch.stack(all_history)
    return all_history


def all_single_user(dataset, user):
    result = []
    user_pos = dataset.all_pos[user]
    for pos_item in user_pos:
        while True:
            neg_item = np.random.randint(0, dataset.num_item)
            if neg_item in user_pos:
                continue
            else:
                break
        result.append([user, pos_item, neg_item])
    return torch.tensor(result)


def all_cluster_user(dataset, cluster_user):
    result = []
    for user in cluster_user:
        user_pos = dataset.all_pos[user]
        for pos_item in user_pos:
            while True:
                neg_item = np.random.randint(0, dataset.num_item)
                if neg_item in user_pos:
                    continue
                else:
                    break
            result.append([user, pos_item, neg_item])
    return torch.tensor(result)


def InfoNCE(view1, view2, temperature):
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.mean(cl_loss)


def split_list(iterable, granularity):
    result = []
    base_length = len(iterable) // granularity
    ideal_length_difference = len(iterable) / granularity - base_length
    remain = 0
    base = 0
    for i in range(0, granularity):
        length = base_length
        remain += ideal_length_difference
        if remain > 1:
            remain -= 1
            length += 1
        if i == granularity - 1 and remain > 0:
            length += 1
        result.append(iterable[base:base + length])
        base += length
    return result
