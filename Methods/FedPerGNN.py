import numpy
import tqdm
from torch.nn.functional import softplus
from torch_geometric.nn import LGConv
from Code.BaseModel import BaseModel
from Code.utils import *


class FedPerGNN(BaseModel):
    def __init__(self, dataset, num_layer=2, embedding_dim=64):
        super().__init__()
        self.num_user = dataset.num_user
        self.num_item = dataset.num_item
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.num_user, embedding_dim=embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.num_item, embedding_dim=embedding_dim)
        self.dataset = dataset
        torch.nn.init.xavier_uniform_(self.user_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.item_embedding.weight.data)
        self.layers = torch.nn.ModuleList()
        for i in range(0, num_layer):
            self.layers.append(LGConv())
        self.edge_index = dataset.edge_index
        self.user_edge_index = []
        self.prepare()

    def prepare(self):
        edge_target_range = {}
        edge_index = self.edge_index.cpu().tolist()
        print("Generating edge target dict")
        for idx, _ in tqdm.tqdm(enumerate(edge_index[0])):
            if _ not in edge_target_range.keys():
                edge_target_range[_] = []
            edge_target_range[_].append(edge_index[1][idx])
        print("Done")
        print("Preparing user edge_index")
        for i in tqdm.tqdm(range(self.num_user)):
            src, dst = [], []
            for j in edge_target_range[i]:
                src.extend([j] * len(edge_target_range[j]))
                dst.extend(edge_target_range[j].copy())
            src = torch.tensor(src)
            dst = torch.tensor(dst)
            edge_index = torch.cat(
                [torch.stack([src, dst]),
                 torch.stack([dst, src])], dim=1)
            self.user_edge_index.append(edge_index)
        print("Done")

    def compute(self):
        with torch.no_grad():
            all_user = torch.zeros_like(self.user_embedding.weight).to('cuda:0')
            for user in range(self.num_user):
                user_emb, _ = self.compute_single(self.user_edge_index[user].to('cuda:0'))
                all_user[user] = user_emb[user]
            return all_user, self.item_embedding.weight

    def compute_single(self, edge_index=None):
        if edge_index is None:
            edge_index = self.edge_index
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        all_emb = torch.cat([user_emb, item_emb])
        result = [all_emb]
        for layer in self.layers:
            all_emb = layer(all_emb, edge_index)
            result.append(all_emb)
        temp_result = torch.mean(torch.stack(result, dim=1), dim=1)
        all_user, _ = torch.split(temp_result, [self.num_user, self.num_item])
        all_item = item_emb
        # temp_result = torch.mean(torch.stack(result[:-1], dim=1), dim=1)
        # _, all_item = torch.split(temp_result, [self.num_user, self.num_item])
        return all_user, all_item

    def get_embedding(self, user, pos, neg, edge_index):
        all_user, all_item = self.compute_single(edge_index)
        user_emb = all_user[user]
        pos_emb = all_item[pos]
        neg_emb = all_item[neg]
        user_emb_ego = self.user_embedding.weight[user]
        pos_emb_ego = self.item_embedding.weight[pos]
        neg_emb_ego = self.item_embedding.weight[neg]
        return user_emb, pos_emb, neg_emb, user_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, user, pos, neg, edge_index):
        (user_emb, pos_emb, neg_emb, user_emb_ego, pos_emb_ego, neg_emb_ego) = self.get_embedding(user, pos, neg,
                                                                                                  edge_index)
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
        loss, reg_loss = self.bpr_loss(args[0], args[1], args[2], args[3])
        return loss + reg_loss * kwargs['weight_decay'][0]

    def train_one_epoch(self, *args, **kwargs):
        users = numpy.random.permutation(range(self.num_user))
        optimizer = kwargs['optimizer']
        cul_user = 0
        for i, user_id in tqdm.tqdm(enumerate(users)):
            # sample = uniform_sample_single_user(self.dataset, user_id)
            sample = all_single_user(self.dataset, user_id)
            user = sample[:, 0].to('cuda:0')
            pos_item = sample[:, 1].to('cuda:0')
            neg_item = sample[:, 2].to('cuda:0')
            loss = self.loss(user, pos_item, neg_item, self.user_edge_index[user_id].to('cuda:0'),
                             weight_decay=[1e-4])
            loss.backward()
            cul_user += len(user)
            if cul_user >= kwargs['batch_size'] or i == len(users) - 1:
                optimizer.step()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.)
                optimizer.zero_grad()
                cul_user = 0
