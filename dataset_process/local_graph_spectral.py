#节点及一阶邻居构成的局部图

import torch
from torch.utils.data import Dataset
import argparse
from pygod.utils import load_data
import copy
from torch_geometric.utils import add_self_loops, to_dense_adj
import numpy as np
from torch_geometric.data import Data
from pygod.utils.utility import check_parameter


# 归一化
def _normalize(x):
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min)/x_max
    return x_norm
#随机产生joint 异常
def gen_joint_structural_outlier(data, m, n, random_state=None):
    """
    We randomly select n nodes from the network which will be the anomalies 
    and for each node we select m nodes from the network. 
    We connect each of n nodes with the m other nodes.

    Parameters
    ----------
    data : PyTorch Geometric Data instance (torch_geometric.data.Data)
        The input data.
    m : int
        Number nodes in the outlier cliques.
    n : int
        Number of outlier cliques.
    p : int, optional
        Probability of edge drop in cliques. Default: ``0``.
    random_state : int, optional
        The seed to control the randomness, Default: ``None``.

    Returns
    -------
    data : PyTorch Geometric Data instance (torch_geometric.data.Data)
        The structural outlier graph with injected edges.
    y_outlier : torch.Tensor
        The outlier label tensor where 1 represents outliers and 0 represents
        regular nodes.
    """

    if not isinstance(data, Data):
        raise TypeError("data should be torch_geometric.data.Data")

    if isinstance(m, int):
        check_parameter(m, low=0, high=data.num_nodes, param_name='m')
    else:
        raise ValueError("m should be int, got %s" % m)

    if isinstance(n, int):
        check_parameter(n, low=0, high=data.num_nodes, param_name='n')
    else:
        raise ValueError("n should be int, got %s" % n)

    check_parameter(m * n, low=0, high=data.num_nodes, param_name='m*n')

    if random_state:
        np.random.seed(random_state)


    outlier_idx = np.random.choice(data.num_nodes, size=n, replace=False)
    all_nodes = [i for i in range(data.num_nodes)]
    rem_nodes = []
    
    for node in all_nodes:
        if node is not outlier_idx:
            rem_nodes.append(node)
    
    
    
    new_edges = []
    
    # connect all m nodes in each clique
    for i in range(0, n):
        other_idx = np.random.choice(data.num_nodes, size=m, replace=False)
        for j in other_idx:
            new_edges.append(torch.tensor([[i, j]], dtype=torch.long))
                    

    new_edges = torch.cat(new_edges)


    y_outlier = torch.zeros(data.x.shape[0], dtype=torch.long)
    y_outlier[outlier_idx] = 1

    data.edge_index = torch.cat([data.edge_index, new_edges.T], dim=1)

    return data, y_outlier

def local_graph(data, device):
    
    in_nodes = data.edge_index[0,:]
    out_nodes = data.edge_index[1,:]

    #不包含自身的邻居节点
    neighbor_dict = {}
    #包含自身和邻居节点
    #neighbor_dict_self = {}
    #自身和邻居节点之间的所有边
    neighbor_edge_dict = {}
    neighbor_edge_dict2 = {}
    for in_node, out_node in zip(in_nodes, out_nodes):
        if in_node.item() not in neighbor_dict:
            #neighbor_dict_self[in_node.item()] = []
            #neighbor_dict_self[in_node.item()].append(in_node.item())
            neighbor_dict[in_node.item()] = []
            neighbor_edge_dict [in_node.item()] = [[], []]
            neighbor_edge_dict2 [in_node.item()] = [[], []]
        neighbor_dict[in_node.item()].append(out_node.item())
        #neighbor_dict_self[in_node.item()].append(out_node.item())
        #for node in neighbor_dict[in_node.item()]:
        neighbor_edge_dict [in_node.item()][0].append(in_node.item())
        neighbor_edge_dict [in_node.item()][1].append(out_node.item())
        neighbor_edge_dict2 [in_node.item()][0].append(0)
        #neighbor_edge_dict2 [in_node.item()][1].append(len(neighbor_edge_dict [in_node.item()][1]))

    #将节点的邻居节点之间的边加入局部子图
    #遍历所有节点
    for in_node in range(data.x.shape[0]):
        #print("in_node.item()",in_node)
        #邻居词典中没有in_node节点，意味着in_node节点没有邻居，则无局部子图
        if(neighbor_edge_dict.get(in_node) == None):
            continue
        #若有局部子图，找到in_node节点的一阶邻居，然后获取节点的一阶邻居存入neighbor_edge，将节点和节点之间的边，构成局部子图
        neighbor_edge = copy.deepcopy(neighbor_edge_dict [in_node][1])
        #遍历in_node节点的一阶邻居
        for i, node in enumerate(neighbor_edge):
            #如果in_node第i个邻居没有邻居
            if(neighbor_dict.get(node) == None):
                continue
            #遍历in_node第i个邻居所有邻居
            for node_n in neighbor_dict[node]:
                #print("neighbor_edge_list[in_node.item()][1]",neighbor_edge_list[in_node.item()][1])
                #print("neighbor_dict[node]",neighbor_dict[node])
                #如果 node_n 是 in_node 的邻居
                if node_n in neighbor_edge:
                    neighbor_edge_dict [in_node][0].append(node)
                    neighbor_edge_dict [in_node][1].append(node_n)
                    #若0：{17，186， 432}，17：{72,289}，186：{17,478}，432：{17,72,289}，则：neighbor_edge_dict2 [in_node][0] = 
                    neighbor_edge_dict2 [in_node][0].append(i+1)
                #neighbor_edge_dict2 [in_node][1].append(len(neighbor_edge_dict [in_node][1]))
        neighbor_edge = 0

    #print(neighbor_edge_dict,neighbor_edge_dict2)

    # 对neighbor_edge_dict2 中的节点归一化
    #{0: [[0, 0, 0, 17, 17, 186, 186, 432, 432, 432],
    #  [17, 186, 432, 72, 289, 17, 478, 17, 72, 289]],
    #转化为： [[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [1, 2, 3, 4, 5, 1, 6, 1, 4, 5]]
    for in_node in neighbor_edge_dict2.keys():
        index_node = max(neighbor_edge_dict2[in_node][0])
        for i, edge_node in enumerate(neighbor_edge_dict[in_node][1]):
        #for i, edge_node in zip(range(len(neighbor_edge_dict[in_node][1])),neighbor_edge_dict[in_node][1]):
            index = neighbor_dict[in_node].index(edge_node)
            neighbor_edge_dict2[in_node][1].append(index+1)


    neighbor_edge_spectral = {}
    # 将边的关系转化为邻接矩阵，求 local graph spectral
    for key in neighbor_edge_dict2.keys():
        adj = to_dense_adj(torch. tensor(neighbor_edge_dict2[key]))
        E, V = torch.linalg.eig(adj)
        neighbor_edge_spectral[key] = E
        
    
                          
    neighbor_num_list = []
    for i in neighbor_dict:
        neighbor_num_list.append(len(neighbor_dict[i]))
    
    neighbor_num_list = torch.tensor(neighbor_num_list).to(device)
    return neighbor_dict, neighbor_num_list, neighbor_edge_spectral

# 对neighbor_edge_spectral，中的向量进行截断或padding，是其长度为32
def truncate_and_pad(vector, length):
    """
    截断或填充向量到指定长度。 
    :param vector: 输入向量（列表）
    :param length: 目标长度
    :return: 处理后的向量
    """
    if len(vector.squeeze(0)) > length:
        # 截断
        return vector.squeeze(0)[:length]
    else:
        # 填充
        #print("vector:",vector,vector.shape,len(vector))
        padding_length = length - len(vector.squeeze(0))
        return torch.from_numpy(np.pad(vector.squeeze(0), (0, padding_length), 'constant'))

def neighbor_pad_or_truncate(vector_dict, k):
    processed_dict = {}
    for key, vector in vector_dict.items():
        
        processed_vector = truncate_and_pad(vector, k)
        processed_dict[key] = processed_vector
    return processed_dict

