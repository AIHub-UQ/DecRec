import numpy
import torch
from torch.nn import Embedding, Linear, LeakyReLU, Sequential, ModuleList, BatchNorm1d, Dropout, ReLU
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits, softplus
from Code.BaseModel import BaseModel
from collections import OrderedDict
from numpy import random, array
from tqdm import tqdm
from Code.utils import uniform_sample_single_user, get_all_history
from torch_geometric.nn import LGConv
from copy import deepcopy


class FedYouTubeDNN(BaseModel):
    def __init__(self, dataset, embed_dim=64, dnn_dims=None, num_layer=3, user_ratio=0.01, step_function=LeakyReLU):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_user = dataset.num_user
        self.num_item = dataset.num_item
        self.dataset = dataset
        self.user_embedding = Embedding(self.num_user, embedding_dim=embed_dim)
        self.item_embedding = Embedding(self.num_item, embedding_dim=embed_dim)
        torch.nn.init.xavier_uniform_(self.user_embedding.weight)
        torch.nn.init.xavier_uniform_(self.item_embedding.weight)
        self.layers = torch.nn.ModuleList()
        for i in range(0, num_layer):
            self.layers.append(LGConv())
        self.batch_size = int(self.num_user * user_ratio)
        self.all_history = get_all_history(dataset)
        self.edge_index = dataset.edge_index

    def compute(self):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        all_emb = torch.cat([user_emb, item_emb])
        result = [all_emb]
        for layer in self.layers:
            all_emb = layer(all_emb, self.edge_index)
            result.append(all_emb)
        result = torch.stack(result, dim=1)
        result = torch.mean(result, dim=1)
        all_user, all_item = torch.split(result, [self.num_user, self.num_item])
        return all_user, all_item

    def get_embedding(self, user, pos, neg):
        all_user, all_item = self.compute()
        user_emb = all_user[user]
        pos_emb = all_item[pos]
        neg_emb = all_item[neg]
        user_emb_ego = self.user_embedding.weight[user]
        pos_emb_ego = self.item_embedding.weight[pos]
        neg_emb_ego = self.item_embedding.weight[neg]
        return user_emb, pos_emb, neg_emb, user_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, user, pos, neg):
        (user_emb, pos_emb, neg_emb, user_emb_ego, pos_emb_ego, neg_emb_ego) = self.get_embedding(user, pos, neg)
        reg_loss = (1 / 2) * (user_emb_ego.norm(2).pow(2) +
                              pos_emb_ego.norm(2).pow(2) +
                              neg_emb_ego.norm(2).pow(2)) / float(len(user))
        pos_score = torch.mul(user_emb, pos_emb)
        pos_score = torch.sum(pos_score, dim=1)
        neg_score = torch.mul(user_emb, neg_emb)
        neg_score = torch.sum(neg_score, dim=1)
        loss = torch.mean(softplus(neg_score - pos_score))
        return loss, reg_loss

    def loss(self, *args, **kwargs):
        sample = uniform_sample_single_user(self.dataset, kwargs['user'])
        user = sample[:, 0].to('cuda:0')
        pos_item = sample[:, 1].to('cuda:0')
        neg_item = sample[:, 2].to('cuda:0')
        # all_items = self.item_model(self.item_embedding.weight)
        # user = self.user_model(self.get_history_embedding(user))
        # pos_item = all_items[pos_item]
        # neg_item = all_items[neg_item]
        loss, reg_loss = self.bpr_loss(user, pos_item, neg_item)
        return loss + reg_loss * 0.0005

    def train_one_epoch(self, *args, **kwargs):
        users = numpy.random.permutation(range(self.num_user))[:self.batch_size]
        total_number = numpy.sum(self.dataset.user_active_num[users])
        current_state_dict = deepcopy(self.state_dict())
        global_state_dict = None
        optimizer = kwargs['optimizer']
        for user in tqdm(users):
            loss = self.loss(user=user)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ratio = self.dataset.user_active_num[user] / total_number
            temp_state_dict = self.state_dict()
            for key in temp_state_dict.keys():
                temp_state_dict[key] *= ratio
            if global_state_dict is None:
                global_state_dict = deepcopy(temp_state_dict)
            else:
                for key in global_state_dict.keys():
                    global_state_dict[key] += temp_state_dict[key]
            self.load_state_dict(current_state_dict)
        self.load_state_dict(global_state_dict)

# 1943008
