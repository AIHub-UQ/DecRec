import torch
import torch_geometric
import tqdm
from torch.nn.functional import softplus
from torch_geometric.nn import LGConv


class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def compute(self, *args, **kwargs):
        pass

    def prepare(self):
        pass

    def loss(self, *args, **kwargs):
        pass

    def forward(self, users, items):
        pass

    def train_one_epoch(self, *args, **kwargs):
        pass

    def get_user_rating(self, users):
        all_users, all_items = self.compute()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = torch.sigmoid(torch.matmul(users_emb, items_emb.t()))
        return rating
