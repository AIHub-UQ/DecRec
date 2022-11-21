import numpy
import torch
from torch.nn import Embedding, Linear, LeakyReLU, Sequential, ModuleList, BatchNorm1d, Dropout, ReLU
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from Code.BaseModel import BaseModel
from collections import OrderedDict
from numpy import random, array
from tqdm import tqdm
from Code.utils import get_all_history


class DCCL(BaseModel):
    def __init__(self, dataset, embed_dim=64, dnn_dims=None, num_neighbour=2047, num_batch=1024, optimizer=None,
                 start_up_epochs=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_user = dataset.num_user
        self.num_item = dataset.num_item
        self.item_embedding = Embedding(num_embeddings=self.num_item, embedding_dim=embed_dim)
        torch.nn.init.xavier_uniform_(self.item_embedding.weight)
        if dnn_dims is None:
            dnn_dims = [64]
        dnn_dims = [embed_dim] + dnn_dims + [embed_dim]
        self.user_model = ModuleList()
        self.dataset = dataset
        self.neighbour = []
        self.num_neighbour = num_neighbour
        self.num_batch = num_batch
        self.item_tower = Sequential(
            # Linear(64, 64),
            # LeakyReLU(),
            # BatchNorm1d(64)
        )
        for i in tqdm(range(self.num_user)):
            # constructing user tower for each user
            user_model = OrderedDict()
            user_model.update({"Dropout": Dropout(0.2)})
            for i in range(len(dnn_dims) - 1):
                user_model.update({"Layer" + str(i): Linear(dnn_dims[i], dnn_dims[i + 1])})
                user_model.update({"Activation" + str(i): ReLU()})
                torch.nn.init.xavier_uniform_(user_model['Layer' + str(i)].weight)

            if torch.cuda.is_available():
                self.user_model.append(Sequential(user_model))
            else:
                self.user_model.append(Sequential(user_model))

            # randomly generate neighbours
            population = numpy.arange(self.num_user - 1)
            population[i:] += 1
            neighbour = random.choice(population, num_neighbour, False)
            self.neighbour.append(neighbour)
        if optimizer is None:
            optimizer = torch.optim.Adam(params=self.parameters(), lr=0.001, weight_decay=0.0005)
        self.optimizer = optimizer
        self.all_history = get_all_history(dataset)
        self.epoch = 0
        self.start_up_epochs = start_up_epochs
        self.item_emb = None

    def compute(self, user_index=None, tower_index=None, item_index=None, mask_index=None):
        user_emb = []
        if user_index is None:
            user_index = list(range(self.num_user))

        # getting user embedding by aggregating the item embeddings in the user history
        user_embedding_history = self.get_history_embedding(user_index, mask_index)

        # using specific user tower to calculate user embedding
        if tower_index is not None:
            user_emb = self.user_model[tower_index](user_embedding_history)
        else:
            # calculating user embedding with their own user tower
            for index, i in enumerate(user_index):
                temp_emb = self.user_model[i](user_embedding_history[index])
                with torch.no_grad():
                    user_emb.append(temp_emb)
            user_emb = torch.stack(user_emb)
        item_emb = self.item_emb
        if item_index is not None:
            item_emb = item_emb[item_index]
        return user_emb, item_emb

    def loss(self, user_index, weight_decay=5e-4):
        # randomly pick a positive item
        masks = numpy.random.choice(numpy.arange(len(self.dataset.all_pos[user_index])), 1).tolist()
        items = [self.dataset.all_pos[user_index][masks[-1]]]
        users = [user_index]
        for neighbour in self.neighbour[user_index]:
            users.append(neighbour)
            # randomly pick an item from a neighbour
            masks.extend(numpy.random.choice(numpy.arange(len(self.dataset.all_pos[user_index])), 1).tolist())
            items.extend([self.dataset.all_pos[user_index][masks[-1]]])
        user_emb, item_emb = self.compute(user_index=users, tower_index=user_index, item_index=items, mask_index=masks)
        pred = torch.inner(user_emb, item_emb)
        # adjusting the prediction value by substituting the logarithm of frequency of each item
        pred = pred - torch.log(torch.tensor(self.dataset.frequency[items]).to('cuda:0'))

        # label is now a square matrix with only the main diagonal filled with 1 and other entries are all 0.
        label = torch.eye(len(users), len(items)).to('cuda:0')
        loss = cross_entropy(pred, label.float())

        # l2 regularization
        reg_loss = (user_emb.norm(2).pow(2) / (user_emb.shape[0] * user_emb.shape[1])
                    + item_emb.norm(2).pow(2) / (item_emb.shape[0] * item_emb.shape[1]))
        return loss + weight_decay * reg_loss

    def train_one_epoch(self, *args, **kwargs):
        self.item_emb = self.item_tower(self.item_embedding.weight)
        user_list = list(range(self.num_user))
        numpy.random.shuffle(user_list)
        loss = None
        cnt = 0
        for i in tqdm(user_list):
            if loss is None:
                loss = self.loss(i)
            else:
                loss += self.loss(i)
            cnt += 1

            # backward the gradients every 500 user to save training time.
            if cnt == 1:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.)
                self.optimizer.step()
                self.item_emb = self.item_tower(self.item_embedding.weight)
                loss = None
                cnt = 0
        # in case the number of user is not an integer multiple of 500
        if loss is not None:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.)
            self.optimizer.step()
        self.calculate_neighbours()
        self.epoch += 1

    def calculate_neighbours(self):
        with torch.no_grad():
            # for first several epochs, neighbours are randomly selected
            if self.epoch < self.start_up_epochs:
                for user in tqdm(range(self.num_user)):
                    population = numpy.arange(self.num_user - 1)
                    population[user:] += 1
                    self.neighbour[user] = random.choice(population, self.num_neighbour, False)
            else:
                # calculating neighbours based on the cosine similarity of the user embeddings
                user_emb, _ = self.compute()
                for user in tqdm(range(self.num_user)):
                    user_weight = user_emb[user]
                    similarity = torch.cosine_similarity(user_weight, user_emb)
                    _, new_neighbours = torch.topk(similarity, self.num_neighbour + 1)
                    new_neighbours = new_neighbours[1:]
                    self.neighbour[user] = new_neighbours.cpu().numpy()

    def get_history_embedding(self, user, mask=None):
        temp_history = torch.clone(self.all_history[user]).to('cuda:0')
        if mask is not None:
            temp_history[torch.arange(len(mask)), mask] = -1
        padding_length = (temp_history == -1).sum(dim=1, keepdim=True)
        non_padding_length = (temp_history != -1).sum(dim=1, keepdim=True)
        temp_history[temp_history == -1] = 0
        embedding_matrix = self.item_embedding(torch.flatten(temp_history)).reshape(
            len(user), -1, self.embed_dim)
        sum_pooling_matrix = torch.sum(embedding_matrix, dim=1) - padding_length * self.item_embedding.weight[[0]]
        all_history_embedding = sum_pooling_matrix / (non_padding_length.float() + 1.e-12)
        return all_history_embedding
