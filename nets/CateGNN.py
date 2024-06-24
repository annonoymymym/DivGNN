import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPReadout
from layers.graphsage_layer import GraphSageLayer
from layers.autoConv import AUTOGCNLayer
from util import Standardize


class Cate_CNN(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']

        # out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']


        self.n_classes = n_classes
        self.device = net_params['device']
        self.cate_num = net_params['cate_num']
        self.cate_version = net_params['cate_version']
        # self.special_node_version =net_params['special_node_version']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.hetero_conv = net_params['hetero_conv']

        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        # if self.cate_version in ['homo', 'mix']:

        self.layers = nn.ModuleList([CateGNN_layer(hidden_dim, hidden_dim,self.device,dropout) for _ in range(n_layers - 1)])
        self.layers.append(CateGNN_layer(hidden_dim, int(hidden_dim/4),self.device,dropout))

        if self.cate_version in ['homo_hetero', 'hetero']:

            hidden_dim_hetero = net_params['hidden_dim_hetero']
            n_layers_hetero = net_params['L_hetero']
            type = 'graphsage' if self.hetero_conv in ['ego_neigh'] else None
            self.MLP_sage = MLPReadout(self.cate_num, hidden_dim_hetero)
            self.mlp_homo = MLPReadout(int(hidden_dim / 4), int(hidden_dim_hetero / 4))
            self.layer_hetero = nn.ModuleList([CateGNN_layer(hidden_dim_hetero, hidden_dim_hetero, self.device, dropout,type=type) for _ in range(n_layers_hetero - 1)])
            self.layer_hetero.append(CateGNN_layer(hidden_dim_hetero, int(hidden_dim_hetero / 4), self.device, dropout,type=type))

            self.gate = AdditiveGatedLayer(int(hidden_dim_hetero / 4), n_classes)
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.hid1 = int(hidden_dim_hetero / 4)
            self.MLP_layer1 = MLPReadout(self.cate_num * self.hid1, int(hidden_dim_hetero / 4))
            self.attention = AttentionLayer(self.hid1, self.hid1)
            self.homo_Standardize = Standardize(int(hidden_dim/4),mode=net_params['normal_mode'])
            self.hetero_Standardize = Standardize(int(hidden_dim_hetero / 4),mode=net_params['normal_mode'])

            if self.hetero_conv in ['high_pass']:
                opt = net_params['opt']
                self.MLP_high_pass = MLPReadout(self.cate_num, hidden_dim_hetero)
                self.graph_norm = False
                self.layer_hetero = nn.ModuleList(
                    [AUTOGCNLayer(hidden_dim_hetero, hidden_dim_hetero, F.relu,dropout, self.graph_norm, self.batch_norm, num_filters=5,opt=opt) for _ in
                     range(n_layers_hetero - 1)])
                self.layer_hetero.append(AUTOGCNLayer(hidden_dim_hetero, int(hidden_dim_hetero / 4), F.relu,dropout, self.graph_norm, self.batch_norm,num_filters=5,opt=opt))

        # self.attention = AttentionLayer(self.hid, self.hid)
        self.hid = int(hidden_dim/4)
        self.MLP_layer = MLPReadout(self.cate_num * self.hid,self.hid )

        self.n_classes=n_classes

        # self.cate_gnn = CateGNN_layer(in_dim, hidden_dim, n_classes,self.device)

    def forward(self, g, feature, e=None):
    #     homo_f = g.homo
    #     hetero_f = g.hetero
    #     hetero_adjacency_matrix = g.hetero_adjacency_matrix

        homo_p = g.homo_p
        hetero_p = g.hetero_p
        cbatch = g.cbatch
        special_node = None

        # special_node = g.special_node if self.special_node_version == True else None
        cate_counting = cbatch.view(-1, self.cate_num)
        pred_homo, pred_hetero = 0, 0

        h = None
        if self.cate_version in ['homo']:
            # h_p = self.in_feat_dropout(homo_p)
            for CateConv in self.layers:
                h = CateConv(homo_p, h)
            pred = self.readout_function(g, h, cbatch, 'cate')
        elif self.cate_version in ['hetero']:
            if self.hetero_conv in ['gcn']:
                hetero_node_index = (torch.sum(hetero_p, dim=0).squeeze(-1) == 0).nonzero().squeeze(-1)
                h = self.MLP_sage(feature)
                h[hetero_node_index] = 0
                h_p = self.in_feat_dropout(hetero_p)
                for CateConv in self.layer_hetero:
                    h = CateConv(h_p, h)
                pred = self.readout_function(g, h, cbatch, 'attention')
            elif self.hetero_conv in ['ego_neigh']:
                hetero = self.in_feat_dropout(hetero_p)
                hetero_node_index = (torch.sum(hetero,dim=0).squeeze(-1)==0).nonzero().squeeze(-1)
                h_hetero = self.MLP_sage(feature)
                h_hetero[hetero_node_index]=0
                for CateConv1 in self.layer_hetero:
                    h_hetero = CateConv1(hetero, h_hetero)
                pred = self.readout_function(g, h_hetero, cbatch, 'attention')
            elif self.hetero_conv in ['high_pass']:
                hetero_node_index = (torch.sum(hetero_p,dim=0).squeeze(-1)==0).nonzero().squeeze(-1)
                # nput should be full graph
                feature = self.MLP_high_pass(feature)
                feature = self.in_feat_dropout(feature)
                for conv in self.layer_hetero:
                    feature = conv(g, feature)
                g.ndata['h'] = feature
                pred = self.readout_function(g, feature, cbatch, 'mean')
        elif self.cate_version in ['homo_hetero']:
            #homo
            h_homo = None
            for CateConv in self.layers:
                h_homo = CateConv(homo_p, h_homo)
            pred_homo = self.mlp_homo(self.readout_function(g, h_homo, cbatch, 'cate'))
            #ego_feature hetero
            if self.hetero_conv in ['ego_neigh']:
                hetero = self.in_feat_dropout(hetero_p)
                hetero_node_index = (torch.sum(hetero, dim=0).squeeze(-1) == 0).nonzero().squeeze(-1)
                h_hetero = self.MLP_sage(feature)
                h_hetero[hetero_node_index] = 0
                for CateConv1 in self.layer_hetero:
                    h_hetero = CateConv1(hetero, h_hetero)
                pred_hetero = self.readout_function(g, h_hetero, cbatch, 'attention')
            elif self.hetero_conv in ['high_pass']:
                hetero_node_index = (torch.sum(hetero_p, dim=0).squeeze(-1) == 0).nonzero().squeeze(-1)
                # nput should be full graph
                feature = self.MLP_high_pass(feature)
                feature = self.in_feat_dropout(feature)
                for conv in self.layer_hetero:
                    feature = conv(g, feature)
                # feature[hetero_node_index]=0
                g.ndata['h'] = feature
                pred_hetero = self.readout_function(g, feature, cbatch, 'mean')
                # print("pred_hetero", pred_hetero)
            # pred = self.gate(self.homo_Standardize(pred_homo), self.hetero_Standardize(pred_hetero))
            pred = self.gate(pred_homo, pred_hetero)

        return pred
    def readout_function(self, g, h, cbatch, readout):
        if self.readout == "sum":
            g.ndata['h'] = h
            pred = dgl.sum_nodes(g, 'h')
        elif readout == "max":
            g.ndata['h'] = h
            pred = dgl.max_nodes(g, 'h')
        elif readout == "mean":
            g.ndata['h'] = h
            pred = dgl.mean_nodes(g, 'h')
        elif readout == "cate":
            segments = torch.split(h, cbatch.tolist())
            summed_segments = torch.stack(
                [torch.sum(segment, dim=0) if segment.numel() > 0 else torch.zeros(h.size(1)).to(self.device) for
                 segment in segments])
            pred = self.MLP_layer(summed_segments.view(-1, self.hid * self.cate_num))
        elif readout == "cate_without_mlp":
            segments = torch.split(h, cbatch.tolist())
            summed_segments = torch.stack(
                [torch.sum(segment, dim=0) if segment.numel() > 0 else torch.zeros(h.size(1)).to(self.device) for
                 segment in segments])
            pred = summed_segments.view(-1, self.hid)
        elif readout == "attention":
            # segments = torch.split(h, cbatch.tolist())
            attention_value, attention_map = self.attention(h)
            segments = torch.split(attention_value, cbatch.tolist())
            summed_segments = torch.stack([torch.sum(segment, dim=0) if segment.numel() > 0 else torch.zeros(attention_value.size(1)).to(self.device) for segment in segments])
            pred = self.MLP_layer1(summed_segments.view(-1, self.hid1 * self.cate_num))
        elif readout == "attention1":
            # segments = torch.split(h, cbatch.tolist())
            segments = torch.split(h, cbatch.tolist())
            summed_segments = torch.stack([torch.sum(segment, dim=0) if segment.numel() > 0 else torch.zeros(h.size(1)).to(self.device) for segment in segments])
            summed_segments, summed_segments_attention_map = self.attention(summed_segments)
            pred = self.MLP_layer1(summed_segments.view(-1, self.hid1 * self.cate_num))
        else:
            g.ndata['h'] = h
            pred = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        return pred

    def contrastive_loss(self,embedding1, embedding2, temperature=0.1):
        """
        Calculate the contrastive loss to minimize the mutual information between two embeddings.

        Args:
        - embedding1 (Tensor): The first embedding vector.
        - embedding2 (Tensor): The second embedding vector.
        - temperature (float): A temperature scaling factor to control the separation.

        Returns:
        - Tensor: The calculated contrastive loss.
        """
        # Normalize the embeddings
        embedding1_normalized = F.normalize(embedding1, p=2, dim=1)
        embedding2_normalized = F.normalize(embedding2, p=2, dim=1)

        # Compute the similarity matrix
        similarity_matrix = torch.matmul(embedding1_normalized, embedding2_normalized.T) / temperature

        # Diagonal elements are positive examples, off-diagonals are negative examples
        positives = torch.diag(similarity_matrix)

        # Mask to exclude positive examples from the denominator
        mask = ~torch.eye(positives.size(0), device=positives.device).bool()

        # Calculate the denominator
        negatives = similarity_matrix[mask].view(similarity_matrix.size(0), -1)
        denominator = torch.exp(positives) / torch.sum(torch.exp(negatives), dim=1)

        # Compute the contrastive loss
        loss = -torch.log(denominator).mean()

        return loss

    def loss(self, pred, label, other_loss=None):
        criterion = nn.CrossEntropyLoss()
        primary_loss = criterion(pred, label)
        if other_loss is not None:
            # Combine the two losses with the trade-off parameter
            total_loss = self.alpha * primary_loss + (1 - self.alpha) * other_loss
            return total_loss, primary_loss, other_loss
        else:
            total_loss = primary_loss
            return total_loss

class CateGNN_layer(nn.Module):
    """
    A class for special graph convolution where all node features are the same and equal to 1.
    """

    def __init__(self, node_feature_dim, output_dim,device,dropout,type=None):
        """
        Initialize the layer.
        :param node_feature_dim: The dimensionality of the node features (all ones).
        :param output_dim: The dimensionality of the output features.
        """
        super(CateGNN_layer, self).__init__()
        # Initialize weights for linear transformation
        self.device = device
        self.type = type
        self.weight = nn.Parameter(torch.randn(node_feature_dim, output_dim)).to(device)
        if self.type =='graphsage':
            self.ego_weight= nn.Parameter(torch.randn(node_feature_dim, node_feature_dim)).to(device)
            self.MLP_layer_sage = MLPReadout(node_feature_dim+ output_dim, output_dim).to(device)
        # else:
        #     self.conv_weight = nn.Parameter(torch.randn(node_feature_dim, node_feature_dim)).to(device)

        self.MLP_layer = MLPReadout(output_dim, output_dim).to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, adj_matrices, h=None):
        """
        Forward pass of the layer.
        :param adj_matrices: A tensor of adjacency matrices for each graph in the batch.
        :param batch_id_idx: A tensor indicating the graph to which each node belongs.

        Returns:
        - Tensor: Output features for each node.
        """
        # Create node features where each feature is a vector of ones


        if h ==None:
            node_features = torch.ones(adj_matrices.shape[0], self.weight.shape[0])#只在第一层的时候呈上special node的ratio
            # Linear transformation for node features
            transformed_features = torch.matmul(node_features.to(self.device), self.weight.to(self.device))
            # Iterate over each graph in the batch
            output_features = torch.matmul(adj_matrices.to(self.device), transformed_features.to(self.device))
        else:
            # if self.type == 'graphsage':  # 因为这个是为hetero准备的，input h一定不是None
            #     transformed_node_features = torch.matmul(h.to(self.device), self.ego_weight.to(self.device))
            #     transformed_features = torch.matmul(h.to(self.device), self.weight.to(self.device))
            #     neighbors_aggregated = torch.matmul(adj_matrices.to(self.device), transformed_features.to(self.device))
            #
            #     output_features = torch.cat((transformed_node_features, neighbors_aggregated), dim=1)
            #     output_features = self.MLP_layer_sage(output_features)
            # else:
            node_features = h
            # Linear transformation for node features
            transformed_features = torch.matmul(node_features.to(self.device), self.weight.to(self.device))
            # Iterate over each graph in the batch
            output_features = torch.matmul(adj_matrices.to(self.device), transformed_features.to(self.device))

        output_features = self.MLP_layer(output_features)
        return self.dropout(output_features)

class AdditiveGatedLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AdditiveGatedLayer, self).__init__()
        # 第一个特征的转换和门控制制
        self.linear_transform_1 = nn.Linear(input_dim, output_dim)
        self.linear_gate_1 = nn.Linear(input_dim, output_dim)

        # 第二个特征的转换和门控制制
        self.linear_transform_2 = nn.Linear(input_dim, output_dim)
        self.linear_gate_2 = nn.Linear(input_dim, output_dim)

    def forward(self, x1, x2):
        # 处理第一个特征
        transform_1 = self.linear_transform_1(x1)
        gate_1 = torch.sigmoid(self.linear_gate_1(x1))
        gated_output_1 = transform_1 * gate_1

        # 处理第二个特征
        transform_2 = self.linear_transform_2(x2)
        gate_2 = torch.sigmoid(self.linear_gate_2(x2))
        gated_output_2 = transform_2 * gate_2

        # 将两个 gated 输出相加
        return gated_output_1 + gated_output_2


class AttentionLayer(nn.Module):
    def __init__(self, feature_dim, attention_head_dim):
        super(AttentionLayer, self).__init__()
        self.feature_dim = feature_dim
        self.attention_head_dim = attention_head_dim

        # Query, Key, and Value linear transformations
        self.query = nn.Linear(feature_dim, attention_head_dim)
        self.key = nn.Linear(feature_dim, attention_head_dim)
        self.value = nn.Linear(feature_dim, attention_head_dim)

        # Scaling factor to prevent the softmax from having extremely small gradients
        self.scale = torch.sqrt(torch.tensor(attention_head_dim, dtype=torch.float32))

    def forward(self, x):
        # x shape: (batch_size, seq_length, feature_dim)

        # Compute queries, keys, values
        Q = self.query(x)  # shape: (batch_size, seq_length, attention_head_dim)
        K = self.key(x)    # shape: (batch_size, seq_length, attention_head_dim)
        V = self.value(x)  # shape: (batch_size, seq_length, attention_head_dim)

        # Calculate attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_map = F.softmax(attention_scores, dim=-1)

        # Apply the attention map to the values
        attention_output = torch.matmul(attention_map, V)
        return attention_output, attention_map


class HeteroGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(HeteroGCNLayer, self).__init__()
        # 定义权重矩阵
        self.weight = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, node_features, adj, node_counts):
        """
        前向传播函数

        参数:
        - node_features: 节点特征矩阵，形状为 (batch_size * n, in_features)
        - adj: 邻接矩阵，形状为 (batch_size * n, batch_size * n)
        - node_counts: 每个图的节点数量列表或张量，长度为batch_size

        返回:
        - outputs: 每个图的输出特征列表，列表中的每个元素的形状为 (n, out_features)
        """
        # 特征聚合：使用邻接矩阵与节点特征矩阵相乘
        agg_features = torch.mm(adj, node_features)
        # 特征转换：通过权重矩阵转换特征
        transformed_features = torch.mm(agg_features, self.weight)
        # 应用非线性激活函数
        # activated_features = F.relu(transformed_features)

        # 将批次中的节点特征分割为每个图的嵌入


        return transformed_features