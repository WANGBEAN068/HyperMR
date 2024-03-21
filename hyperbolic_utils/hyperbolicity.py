import os
import pickle as pkl
import sys
import time

import networkx as nx
import numpy as np
from tqdm import tqdm
import pandas as pd
import csv
# from hyperbolic_utils.data_utils import load_data_lp


def hyperbolicity_sample(G, num_samples=50000):
    curr_time = time.time()
    hyps = []
    for i in tqdm(range(num_samples)):
        curr_time = time.time()
        node_tuple = np.random.choice(G.nodes(), 4, replace=False)
        s = []
        try:
            d01 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[1], weight=None)
            d23 = nx.shortest_path_length(G, source=node_tuple[2], target=node_tuple[3], weight=None)
            d02 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[2], weight=None)
            d13 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[3], weight=None)
            d03 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[3], weight=None)
            d12 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[2], weight=None)
            s.append(d01 + d23)
            s.append(d02 + d13)
            s.append(d03 + d12)
            s.sort()
            hyps.append((s[-1] - s[-2]) / 2)
        except Exception as e:
            continue
    print('Time for hyp: ', time.time() - curr_time)
    return max(hyps)

def load_kvqa_kg(data_path):
    data = []
    with open(data_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:  # 将csv 文件中的数据保存到data中
            data.append(row)


if __name__ == '__main__':
    # dataset = 'KGfacts-CloseWorld-refine.csv'
    # data_path = os.path.join('/home/nlp306/Data/User_file/wb/kbvqa-public/data/kvqa/raw/', dataset)
    # dataset = '3H-kb.txt'
    dataset = 'PQL2-KB.txt'
    data_path = os.path.join('/home/nlp306/Data/User_file/wb/kbvqa-public/data/PathQuestion/raw/', dataset)

    head_list = []
    tail_list =[]
    head_tail_dict = {}
    with open(data_path) as file:
        # reader = csv.reader(file)
        reader = file.readlines()
        for row in reader:
            # if dataset == '3H-kb.txt':
            row = row.split('\t')
            head, tail = row[0], row[-1]
            head_list.append(head)
            tail_list.append(tail)
    node_list = list(set(head_list + tail_list))
    adjacency_dict = {}
    node_token_dict = {}

    for i in range(len(node_list)):
        node_token_dict[node_list[i]] = i

    if dataset == 'KGfacts-CloseWorld-refine.csv':
        node_list = node_list[1:]

    for i in tqdm(range(len(node_list))):
        node_adj = []
        for head_index in range(len(head_list)):
            if head_list[head_index] == node_list[i]:
                node_adj.append(node_token_dict[tail_list[head_index]])
        adjacency_dict[i] = node_adj
    # np.save('kvqa_adj.npy', adjacency_dict)
    # adjacency_dict = np.load('kvqa_adj.npy', allow_pickle=True).item()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(adjacency_dict))
    graph = nx.from_scipy_sparse_matrix(adj)
    print('Computing hyperbolicity', graph.number_of_nodes(), graph.number_of_edges())
    hyp = hyperbolicity_sample(graph)
    print('Hyp: ', hyp)

    # data = load_kvqa_kg(dataset)
    # graph = nx.from_scipy_sparse_matrix(data['adj_train'])
    # print('Computing hyperbolicity', graph.number_of_nodes(), graph.number_of_edges())
    # hyp = hyperbolicity_sample(graph)
    # print('Hyp: ', hyp)

