import torch
import torch_geometric
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree, dropout_adj

# @functional_transform('edge dropping')
class Edge_dropping(BaseTransform):
    def __init__(self, method=None, node_centrality=None, is_undirected=True):
        '''
        :param method: GCA \ Random
        :param node_centrality: Param which belong to GCA
        :param is_undirected: Param which belong to GCA
        '''
        self.method = method
        self.node_centrality = node_centrality
        self.is_undirected = is_undirected
        # 控制最小丢弃率
        self.p_e = 0.2
        # 控制最大丢弃率
        self.p_t = 0.8


    def __call__(self, data):
        row, col = data.edge_index
        if self.method == 'GCA':
            # 1 计算节点中心指标 - degree centrality, eigen-vector centrality, and PageRank
            if self.node_centrality == 'degree':
                node_row_centrality = degree(row)
                node_col_centrality = degree(col)
            elif self.node_centrality == 'eigen':
                pass
            elif self.node_centrality == 'pagerank':
                pass
            # 2 对于无向图，边中心指标为相连节点的中心指标的平均，对于有向图，边中心指标为指向节点的中心指标的平均
            if self.is_undirected is True:
                edge_centrality = (node_row_centrality + node_col_centrality) / 2
            else:
                edge_centrality = node_col_centrality
            # 3 将边中心指标对数化
            edge_centrality = torch.log(edge_centrality)
            # print(f"edge_centrality is {edge_centrality}")
            # print(f"max edge_centrality is {torch.max(edge_centrality)}")
            # print(f"mean edge_centrality is {torch.mean(edge_centrality)}")
            # 4 公式
            prob = torch.min((torch.max(edge_centrality) - edge_centrality
                                         / torch.max(edge_centrality) - torch.mean(edge_centrality)) * self.p_e, torch.tensor(self.p_t))
            # print("prob", prob)

        if self.method == 'Random':
            pass
        edge_index, edge_attr = self.dropout_edge(edge_index=data.edge_index, edge_attr=data.edge_attr, p=prob)
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        return data

    # 对每条边考虑丢弃率
    def dropout_edge(self, edge_index, edge_attr=None, p=None):
        # 此处随机概率值得商榷
        mask = torch.randn((p.shape)) <= p
        print(f"mask is {mask}")
        indice_mask = torch.tensor([i for i in range(mask.shape[0]) if mask[i].item() is True], dtype=torch.int64)
        print(f"indice_mask is {indice_mask}")
        row, col = edge_index

        print(f"edge_index is {edge_index.shape}")
        row = torch.tensor([row[i] for i in range(row.shape[0]) if row[i] not in indice_mask])
        col = torch.tensor([col[i] for i in range(col.shape[0]) if col[i] not in indice_mask])
        edge_index = torch.stack([row, col], dim=0)
        print(f"edge_index is {edge_index.shape}")

        print(f"edge_attr is {edge_attr.shape}")
        edge_attr = edge_attr[row]
        print(f"edge_attr is {edge_attr.shape}")
        return edge_index, edge_attr


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


# class MyAug_EdgeRemoving(BaseTransform):
#
#     def __init__(self, prob=None):
#         self.prob = prob
#
#     def __call__(self, data: Data) -> Data:
#         # data.edge_attr = None
#         # batch = data.batch if 'batch' in data else None
#         edge_index, edge_attr = dropout_adj(edge_index=data.edge_index, edge_attr=data.edge_attr, p=self.prob, num_nodes=data.x.size(0))
#         if edge_index.size(1) == 0:
#             return data
#         data = Data(x=data.x, edge_index=edge_index, edge_attr=edge_attr, y=data.y)
#         return data
#
#     def __repr__(self):
#         return '{}(prob={})'.format(self.__class__.__name__, self.prob)


path = '../data/'
transform = T.Compose([Edge_dropping('GCA', 'degree')])
dataset = TUDataset(path, name='MUTAG')
print(dataset[0])
data = transform(dataset[0])
print(data)





