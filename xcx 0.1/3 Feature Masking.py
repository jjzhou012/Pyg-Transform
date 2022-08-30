import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv


@functional_transform('feature_masking')
class Feature_masking(BaseTransform):
    def __init__(self, method=None, p=0.5):
        self.method = method
        self.p = p

    def __call__(self, data: Data) -> Data:
        if self.method == 'Bernoulli':
            node_masking = torch.rand(data.x.shape[1]) < self.p
            data.x[:, node_masking] = 0
        if self.method == 'calculate':
            pass
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


path = '../data/'
transform = T.Compose([Feature_masking('Bernoulli', p=0.2)])
dataset = Planetoid(root=path, name='Cora', transform=transform)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    print(f"loss is {loss}")

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')




