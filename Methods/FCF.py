import torch
from torch.nn.functional import softplus
from Code.BaseModel import BaseModel
from Code.utils import all_single_user
import tqdm
import numpy as np


class FCF(BaseModel):
    def __init__(self, dataset, embedding_dim=64):
        super().__init__()
        self.num_user = dataset.num_user
        self.num_item = dataset.num_item
        self.dataset = dataset
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.num_user, embedding_dim=embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.num_item, embedding_dim=embedding_dim)
        torch.nn.init.xavier_uniform_(self.user_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.item_embedding.weight.data)

    def compute(self):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        return user_emb, item_emb

    def bpr_loss(self, user, pos, neg):
        user_emb = self.user_embedding(user)
        pos_emb = self.item_embedding(pos)
        neg_emb = self.item_embedding(neg)
        reg_loss = (1 / 2) * (user_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(user))
        pos_score = torch.mul(user_emb, pos_emb)
        pos_score = torch.sum(pos_score, dim=1)
        neg_score = torch.mul(user_emb, neg_emb)
        neg_score = torch.sum(neg_score, dim=1)
        loss = torch.mean(softplus(neg_score - pos_score))
        return loss, reg_loss

    def loss(self, *args, **kwargs):
        loss, reg_loss = self.bpr_loss(args[0], args[1], args[2])
        return loss + reg_loss * 1e-4

    def forward(self, users, items):
        user_emb, item_emb = self.compute()
        user = user_emb[users]
        item = item_emb[items]
        result = torch.mul(user, item)
        return result

    def train_one_epoch(self, *args, **kwargs):
        users = np.random.permutation(range(self.num_user))
        optimizer = kwargs['optimizer']
        empty_flag = True
        user, pos_item, neg_item = [], [], []
        for i, user_id in tqdm.tqdm(enumerate(users)):
            sample = all_single_user(self.dataset, user_id)
            if empty_flag:
                empty_flag = False
                user = sample[:, 0].to('cuda:0')
                pos_item = sample[:, 1].to('cuda:0')
                neg_item = sample[:, 2].to('cuda:0')
            else:
                user = torch.hstack([user, sample[:, 0].to('cuda:0')])
                pos_item = torch.hstack([pos_item, sample[:, 1].to('cuda:0')])
                neg_item = torch.hstack([neg_item, sample[:, 2].to('cuda:0')])
            if len(user) >= kwargs['batch_size'] or i == len(users) - 1:
                loss = self.loss(user, pos_item, neg_item)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.)
                empty_flag = True
