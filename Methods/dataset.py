import numpy
import pandas
import torch
from torch.utils.data import Dataset
from Code.utils import split_list


class CustomizeDataset(Dataset):
    def __init__(self, path='Datasets/amazon-book'):
        train_path = path + '/train.txt'
        test_path = path + '/test.txt'
        self.path = path
        self.num_user = 0
        self.num_item = 0
        self.train_data_size = 0
        train_user_unique, train_user, train_item = [], [], []
        test_user_unique, test_user, test_item = [], [], []
        self.all_pos = dict()
        with open(train_path) as file:
            for row in file:
                row = list(map(int, row.strip().split(' ')))
                train_user.extend([row[0]] * (len(row) - 1))
                train_item.extend(row[1:])
                train_user_unique.append(row[0])
                self.num_user = max(self.num_user, row[0] + 1)
                self.num_item = max(self.num_item, max(row[1:]) + 1)
                self.all_pos[row[0]] = []
                self.all_pos[row[0]].extend(row[1:])
                self.train_data_size += len(row[1:])
        with open(test_path) as file:
            for row in file:
                row = list(map(int, row.strip().split(' ')))
                test_user.extend([row[0]] * (len(row) - 1))
                test_item.extend(row[1:])
                test_user_unique.append(row[0])
                self.num_user = max(self.num_user, row[0] + 1)
                if len(row) > 1:
                    self.num_item = max(self.num_item, max(row[1:]) + 1)
        self.train_user_unique = numpy.array(train_user_unique)
        self.train_user = numpy.array(train_user)
        self.train_item = numpy.array(train_item)
        self.test_user_unique = numpy.array(test_user_unique)
        self.test_user = numpy.array(test_user)
        self.test_item = numpy.array(test_item)
        self.train_item_tmp = torch.tensor(train_item) + self.num_user
        self.train_user_tmp = torch.tensor(train_user)
        self.edge_index = torch.cat(
            [torch.stack([self.train_user_tmp, self.train_item_tmp]),
             torch.stack([self.train_item_tmp, self.train_user_tmp])], dim=1)
        # self.all_neg = []
        # all_items = set(range(self.num_item))
        # for user in range(self.num_user):
        #     pos = set(self.all_pos[user])
        #     neg = all_items - pos
        #     self.all_neg.append(torch.tensor(list(neg)))
        self.test_dict = self.build_test()
        counts = numpy.bincount(self.train_item)
        self.frequency = counts / numpy.sum(counts)
        self.user_active_num = numpy.array([len(self.all_pos[i]) for i in range(self.num_user)])
        self.users_sorted = numpy.argsort(-self.user_active_num)
        print(f"Total {self.train_data_size} lines of record")
        print(f"Max records of a user: {max(self.all_pos)}")

    def build_test(self):
        test_data = {}
        for i, item in enumerate(self.test_item):
            user = self.test_user[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def get_user_pos_item(self, users):
        result = []
        for user in users:
            result.append(self.all_pos[user])
        return result

    def __len__(self):
        return len(self.train_user)

    def __getitem__(self, item):
        return self.train_user[item], self.train_item[item]


class CustomDatasetRF:
    def __init__(self, dataset='ml-100k', method='split', granularity=1):
        self.method = method
        self.granularity = granularity
        self.user_data_train = []
        self.user_data_test = []
        self.num_user = 0
        self.num_item = 0
        with open('Datasets/' + dataset + '/data.txt') as file:
            for row in file:
                row = list(map(int, row.strip().split(' ')))
                self.user_data_test.append(row[-1])
                self.num_item = max(self.num_item, max(row[1:]) + 1)
                row = row[1:-1]
                chunks = split_list(row, granularity)
                self.user_data_train.append(numpy.array(chunks))
                self.num_user += 1


class CustomSeqDataset:
    def __init__(self, path='Dataset/ml-100k/train.txt', sep=',', session_key='SessionID', item_key='ItemID',
                 time_key='Time', item_map=None, df=None, remap_item=True):
        self.item_map = item_map
        if df is None:
            self.df = pandas.read_csv(path, sep=sep, dtype={session_key: int, item_key: int, time_key: float})
        else:
            self.df = df
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        if remap_item is True:
            self.add_item_indices()
            self.df.sort_values([session_key, time_key], inplace=True)
        else:
            self.df['item_idx'] = self.df[self.item_key]
        self.click_offsets = numpy.zeros(self.df[self.session_key].nunique() + 1, dtype=numpy.int32)
        self.click_offsets[1:] = self.df.groupby(self.session_key).size().cumsum()
        self.session_idx_arr = numpy.arange(self.df[self.session_key].nunique())
        if remap_item is True:
            self.num_item = self.item_map[self.item_key].nunique()

    def add_item_indices(self):
        if self.item_map is None:
            item_ids = self.df[self.item_key].unique()  # type is numpy.ndarray
            item2idx = pandas.Series(data=numpy.arange(len(item_ids)),
                                     index=item_ids)
            # Build itemmap is a DataFrame that have 2 columns (self.item_key, 'item_idx)
            itemmap = pandas.DataFrame({self.item_key: item_ids,
                                        'item_idx': item2idx[item_ids].values})
            self.item_map = itemmap
        self.df = pandas.merge(self.df, self.item_map, on=self.item_key, how='inner')


class CustomSeqDataLoader:
    def __init__(self, dataset, batch_size=50):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        df = self.dataset.df
        click_offsets = self.dataset.click_offsets
        session_idx_arr = self.dataset.session_idx_arr

        iters = numpy.arange(self.batch_size)
        maxiter = self.batch_size - 1
        start = click_offsets[session_idx_arr[iters]]
        end = click_offsets[session_idx_arr[iters] + 1]
        mask = []  # indicator for the sessions to be terminated
        finished = False

        while not finished:
            minlen = (end - start).min()
            # Item indices(for embedding) for clicks where the first sessions start
            idx_target = df.item_idx.values[start]

            for i in range(minlen - 1):
                # Build inputs & targets
                idx_input = idx_target
                idx_target = df.item_idx.values[start + i + 1]
                input = torch.LongTensor(idx_input)
                target = torch.LongTensor(idx_target)
                yield input, target, mask

            # click indices where a particular session meets second-to-last element
            start = start + (minlen - 1)
            # see if how many sessions should terminate
            mask = numpy.arange(len(iters))[(end - start) <= 1]
            for idx in mask:
                maxiter += 1
                if maxiter >= len(click_offsets) - 1:
                    finished = True
                    break
                # update the next starting/ending point
                iters[idx] = maxiter
                start[idx] = click_offsets[session_idx_arr[maxiter]]
                end[idx] = click_offsets[session_idx_arr[maxiter] + 1]
