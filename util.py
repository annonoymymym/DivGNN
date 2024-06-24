import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import os
import dgl
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def visualize_dgl_graph(dgl_graph,categories):
    """
    Visualizes a DGL graph with node categories using NetworkX and matplotlib.

    Parameters:
    - dgl_graph: A DGL graph object with node data indicating categories.

    Nodes are colored according to their categories.
    """
    # 将DGL图转换为NetworkX图
    nx_graph = dgl.to_networkx(dgl_graph)

    # 假设每个节点都有一个 'category' 属性，你需要根据实际情况调整
    # 准备颜色映射，这里是示例颜色，根据你的类别数量调整
    color_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow'}

    # 为每个节点分配颜色
    node_colors = [color_map[category] for category in categories.tolist()]
    # 计算布局
    pos = nx.spring_layout(nx_graph)
    # 绘制网络图，使用节点类别对应的颜色
    plt.figure(figsize=(12, 8))  # 设置图的大小
    nx.draw(nx_graph, pos, with_labels=True, node_color=node_colors, node_size=700,
            edge_color='k', linewidths=1, font_size=15, arrows=True)

    # 显示图形
    plt.show()


class Standardize(torch.nn.Module):
    def __init__(self,features_dim, mode='zscore'):
        super(Standardize, self).__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == 'zscore':
            mean = x.mean(dim=0)
            std = x.std(dim=0)
            return (x - mean) / (std + 1e-6)
        elif self.mode == 'minmax':
            min_val = x.min(dim=0)[0]  # Correctly extract the minimum values
            max_val = x.max(dim=0)[0]  # Correctly extract the maximum values
            return (x - min_val) / (max_val - min_val + 1e-6)
        else:
            raise ValueError("Unsupported normalization mode. Choose 'zscore' or 'minmax'.")



def visualize_graph_with_special_edges(dgl_graph, special_adj_matrix, node_labels,save_path=None):
    # 将DGL图转换为NetworkX图
    nx_graph = dgl.to_networkx(dgl_graph)
    pos = nx.spring_layout(nx_graph)  # 为两个图计算相同的布局

    # 处理特殊边
    special_edges_raw = np.argwhere(special_adj_matrix == 1)
    special_edges_np = special_edges_raw.numpy()
    special_edges = [(special_edges_np[0, i], special_edges_np[1, i]) for i in range(special_edges_np.shape[1])]

    # 将标签从张量转换为普通数字
    node_labels = [label.item() if isinstance(label, torch.Tensor) else label for label in node_labels]

    # 准备不同标签的颜色
    unique_labels = set(node_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))  # 使用Set3颜色映射，适合明亮且清晰区分的颜色
    label_color_map = dict(zip(unique_labels, colors))

    # 根据标签分配颜色
    node_colors = [label_color_map[label] for label in node_labels]

    # 创建一个带有两个子图的图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 绘制原始图
    nx.draw(nx_graph, pos, ax=axes[0], with_labels=True, node_color=node_colors)
    axes[0].set_title("Original Graph")

    # 绘制标记过特殊边的图
    nx.draw(nx_graph, pos, ax=axes[1], with_labels=True, node_color=node_colors)
    nx.draw_networkx_edges(nx_graph, pos, edgelist=special_edges, ax=axes[1], edge_color='red', width=2)
    axes[1].set_title("Graph with Special Edges Highlighted")
    plt.show()



def visualize_embeddings(train_loader, test_loader, model, device, dataset_name, train_acc, test_acc,method='pca',save_path=None):
    batch_labels_epoch = []
    batch_scores_epoch=[]
    test_batch_labels_epoch = []
    test_batch_scores_epoch = []
    for iter, (batch_graphs, batch_labels) in enumerate(train_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        batch_labels_epoch.append(batch_labels)
        batch_scores_epoch.append(batch_scores)
    batch_labels_epoch = torch.cat(batch_labels_epoch,dim=0)
    batch_scores_epoch = torch.cat(batch_scores_epoch,dim=0)

    for iter, (test_batch_graphs, test_batch_labels) in enumerate(test_loader):
        test_batch_graphs = test_batch_graphs.to(device)
        test_batch_x = test_batch_graphs.ndata['feat'].to(device)  # num x feat
        test_batch_e = test_batch_graphs.edata['feat'].to(device)
        test_batch_labels = test_batch_labels.to(device)
        test_batch_scores = model.forward(test_batch_graphs, test_batch_x, test_batch_e)
        test_batch_labels_epoch.append(test_batch_labels)
        test_batch_scores_epoch.append(test_batch_scores)
    test_batch_labels_epoch = torch.cat(test_batch_labels_epoch, dim=0)
    test_batch_scores_epoch = torch.cat(test_batch_scores_epoch, dim=0)

    """
    Visualizes embeddings in 2D. Automatically reduces dimensionality if embeddings are not 2D.

    :param embeddings: Numpy array of shape (n_samples, n_features)
    :param labels: Numpy array of shape (n_samples,)
    :param method: 'pca' or 'tsne'
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    def to_numpy(tensor):
        # Convert PyTorch Tensor to NumPy array
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy()
        return tensor.cpu().numpy()

    def reduce_dim(embeddings):
        # Check if the embeddings are PyTorch Tensors and convert them to NumPy arrays if needed
        if isinstance(embeddings, torch.Tensor):
            embeddings = to_numpy(embeddings)

        # Check if dimension reduction is needed
        if embeddings.shape[1] != 2:
            if method == 'pca':
                reducer = PCA(n_components=2)
            elif method == 'tsne':
                reducer = TSNE(n_components=2, learning_rate='auto', init='random')
            else:
                raise ValueError("Method must be 'pca' or 'tsne'")

            return reducer.fit_transform(embeddings)
        else:
            return embeddings

        # Reduce dimensions if needed

    reduced_batch_scores_epoch = reduce_dim(batch_scores_epoch)
    reduced_test_batch_scores_epoch = reduce_dim(test_batch_scores_epoch)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot for the first set of embeddings
    unique_labels = np.unique(batch_labels_epoch.cpu().numpy())
    for label in unique_labels:
        indices = batch_labels_epoch == label
        axes[0].scatter(reduced_batch_scores_epoch[indices.cpu().numpy(), 0],
                        reduced_batch_scores_epoch[indices.cpu().numpy(), 1], label=str(label))
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')
    axes[0].set_title(f'{dataset_name} - Training Set Embeddings')
    axes[0].legend()
    axes[0].text(0.5, 1.05, f'Train Acc: {train_acc:.2f}', ha='center', va='bottom', transform=axes[0].transAxes)

    # Plot for the second set of embeddings
    unique_labels = np.unique(test_batch_labels_epoch.cpu().numpy())
    for label in unique_labels:
        indices = test_batch_labels_epoch == label
        axes[1].scatter(reduced_test_batch_scores_epoch[indices.cpu().numpy(), 0],
                        reduced_test_batch_scores_epoch[indices.cpu().numpy(), 1], label=str(label))
    axes[1].set_xlabel('Component 1')
    axes[1].set_ylabel('Component 2')
    axes[1].set_title(f'{dataset_name} - Test Set Embeddings')
    axes[1].legend()
    axes[1].text(0.5, 1.05, f'Test Acc: {test_acc:.2f}', ha='center', va='bottom', transform=axes[1].transAxes)

    # Save or show the plot
    if save_path:
        plt.savefig(f'{save_path}/{dataset_name}_{method}_{device}_embeddings.png')
    else:
        plt.show()
# Example usage:
# embeddings = np.random.rand(100, 50)  # Replace with your embeddings
# labels = np.random.randint(0, 5, 100)  # Replace with your labels
# visualize_embeddings(embeddings, labels, method='pca')



def find_special_nodes_and_its_adj(node_indices, adj):
    # print("node_indices", node_indices)
    special_nodes = []
    new_adj = torch.zeros_like(adj)

    for node in range(adj.shape[0]):
        neighbors = (adj[node] > 0).nonzero(as_tuple=False).squeeze()

        # Handle the case when neighbors is a 0-d tensor
        if neighbors.ndim == 0 and adj[node].sum() == 0:
            continue
        elif neighbors.ndim == 0:
            neighbors = torch.tensor([neighbors.item()], dtype=torch.int64)

        neighbor_classes = torch.argmax(node_indices,dim=1)[neighbors]
        # print("neighbor_classes", neighbor_classes, torch.argmax(node_indices,dim=1)[node])

        if neighbor_classes.size(0)>=2 and len(neighbor_classes.unique()) == 1 and neighbor_classes[0] != torch.argmax(node_indices,dim=1)[node]:
            # print("yes----------------------")
            special_nodes.append(node)
            new_adj[node, neighbors] = 1
            new_adj[neighbors, node] = 1

    return special_nodes, new_adj