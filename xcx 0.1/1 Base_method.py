import torch
import torch_geometric
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


# @functional_transform('base_method')
class Base_method(BaseTransform):
    def __init__(self):
        pass

    def __call__(self, data):
        print('hello')
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


path = '../data/'
transform = T.Compose([Base_method()])
dataset = TUDataset(path, name='MUTAG')
dataset = transform(dataset)
print(dataset.data)
# loader = DataLoader(dataset, shuffle=True, batch_size=1)
# for l in iter(loader):
#     print(l)




