import os
import sys
import time
import tqdm
import argparse
from Code.RF import RF
from Code.dataset import CustomizeDataset
from Code.LightGCN import LightGCN
from Code.SGL import SGL
from Code.SwarmRec import SwarmRec
from Code.utils import *
from Code.YoutubeDNN import YouTubeDNN
from Code.FedYouTubeDNN import FedYouTubeDNN
from Code.MF import MF
from Code.FCF import FCF
from Code.MetaMF import MetaMF
from Code.FedMF import FedMF
from Code.SemiDRec import SemiDRec
from Code.FedPerGNN import FedPerGNN
from Code.NCF import NCF

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=['yelp-raw', 'ml-100k'], default='ml-100k')
    parser.add_argument('-m', '--model',
                        choices=['ncf', 'gru'], default='gru')
    parser.add_argument('-s', '--seed', type=int, default=2020)
    parser.add_argument('-ep', '--epoch', type=int, default=40)
    parser.add_argument('-me', '--method', choices=['split', 'latest', 'oldest'], default='split')
    parser.add_argument('-g', '--granularity', choices=[1, 2, 4, 8, 16], type=int, default=1)
    args = parser.parse_args()
    print(f'epoch: {args.epoch}\n' +
          f'model: {args.model}\n' +
          f'dataset: {args.dataset}\n' +
          f'method: {args.method}\n' +
          f'granularity: {args.granularity}')
    RF = RF(args.epoch, args.model, args.dataset, args.method, args.granularity)
    ndcg, dispose = RF.simulate()
    print(f'Final Result: ndcg: {sum(ndcg[-5:]) / 5}, dispose: {sum(dispose[-5:]) / 5}')
