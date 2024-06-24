import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse, json
from nets.load_net import gnn_model # import GNNs
from data.data import LoadData # import dataset
import os


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device

#
parser = argparse.ArgumentParser()
parser.add_argument('--config1', help="Please give a config.json file with training/model/data/param details")
parser.add_argument('--config2', help="Please give a config.json file with training/model/data/param details")
#
#
args = parser.parse_args()
with open(args.config1) as f1:
    config1 = json.load(f1)
with open(args.config2) as f2:
    config2 = json.load(f2)

# device
device = gpu_setup(config1['gpu']['use'], config1['gpu']['id'])
# model, dataset, out_dir

MODEL_NAME = config1['model']

# network parameters
net_params1 = config1['net_params']
net_params2 = config2['net_params']

def load_model_json(name, dataset,net_params_json_path, model_path, device='cpu'):
    """
    加载指定路径的模型。

    参数:
    - name: 模型的名称。
    - net_params_json_path: 包含网络参数的JSON文件的路径。
    - model_path: 模型文件的路径。
    - device: 加载模型到这个设备（'cpu'或'cuda'）。

    返回:
    - 加载的模型实例。
    """
    # 首先，从JSON文件中读取net_params
    with open(net_params_json_path, 'r') as f:
        net_params = json.load(f)['net_params']
    print("net_params", net_params)
    net_params['in_dim'] = dataset.all.graph_lists[0].ndata['feat'][0].shape[0]
    num_classes = len(np.unique(dataset.all.graph_labels))
    net_params['n_classes'] = num_classes
    net_params['cate_num'] = net_params['in_dim']

    # 根据读取的参数和其他参数来创建和初始化模型
    model = gnn_model('cate', net_params)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_model(name, net_params, model_path, device='cpu'):
    """
    加载指定路径的模型。

    参数:
    - model_class: 模型的类定义。
    - model_path: 模型文件的路径。
    - device: 加载模型到这个设备（'cpu'或'cuda'）。

    返回:
    - 加载的模型实例。
    """
    model = gnn_model(name, net_params)
    model = model.to(device) # 假设模型的初始化不需要任何参数
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
from sklearn.preprocessing import normalize



def compare_models(models, test_loader, device='cpu'):
    # 存储每个模型的预测结果、confidence、embeddings和真实标签
    all_preds = []
    all_confs = []
    all_embeddings = []
    all_labels = []
    true_labels = []

    for model in models:
        model.eval()  # 设置为评估模式
        model.to(device)
        preds = []
        confs = []
        embeddings = []
        model_labels = []
        for _, (batch_graphs, batch_labels) in enumerate(test_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_labels = batch_labels.to(device)

            # 获取模型输出和embedding
            output, _, _ = model(batch_graphs, batch_x)  # 假设模型返回output和embedding
            conf, pred = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
            preds.extend(pred.cpu().detach().numpy())
            confs.extend(conf.cpu().detach().numpy())
            embeddings.extend(output.cpu().detach().numpy())
            model_labels.extend(batch_labels.cpu().numpy())

        # 归一化embeddings
        embeddings = normalize(np.array(embeddings), norm='l2', axis=1)
        all_preds.append(preds)
        all_confs.append(confs)
        all_embeddings.append(embeddings)
        all_labels.append(model_labels)
        true_labels.append(model_labels)  # assuming batch_labels are the true labels

    # 计算每个模型的准确率及正确分类的样本数量
    for i, preds in enumerate(all_preds):
        acc = accuracy_score(true_labels[i], preds)
        correct_preds = sum(p == t for p, t in zip(preds, true_labels[i]))
        print(f"Model {i + 1} Accuracy: {acc:.4f}, Correctly Classified Samples: {correct_preds}/{len(true_labels[i])}")

    # 计算模型预测正确样本的交集
    correct_preds_sets = [set(np.where(np.array(preds) == np.array(labels))[0]) for preds, labels in zip(all_preds, true_labels)]
    overlap = set.intersection(*correct_preds_sets)
    print(f"Overlap of correct predictions among all models: {len(overlap)} of total test samples: {len(true_labels[0])}")

    # 绘制散点图比较模型的embedding分布
    plt.figure(figsize=(10, 6))
    color_map = [('green', 'blue'), ('red', 'yellow')]  # 为每个模型的每个类别指定颜色

    for i, (embeddings, labels_in_model) in enumerate(zip(all_embeddings, all_labels)):
        for label in [0, 1]:  # 二分类标签
            indices = np.where(np.array(labels_in_model) == label)[0]
            plt.scatter(embeddings[indices, 0], embeddings[indices, 1],
                        color=color_map[i][label],
                        label=f'Model {i + 1} - Class {label}', alpha=0.5)

    # 添加y=x分界线
    xlims = plt.xlim()
    ylims = plt.ylim()
    min_val = min(xlims[0], ylims[0])
    max_val = max(xlims[1], ylims[1])
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x Line')  # 使用黑色虚线

    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.legend()
    plt.title('Comparison of Normalized Model Embeddings by Class')
    plt.savefig('/home/leihan/workspace/Category_homo_hetero/image.png')  # 指定保存路径
    plt.close()  # 关闭图形，释放内存
DATASET_NAME ='MUTAG'
split_number=7
dataset = LoadData(DATASET_NAME, True)
net_params1['in_dim'] = dataset.all.graph_lists[0].ndata['feat'][0].shape[0]
net_params1['cate_num'] = net_params1['in_dim']

net_params2['in_dim'] = dataset.all.graph_lists[0].ndata['feat'][0].shape[0]
num_classes = len(np.unique(dataset.all.graph_labels))
net_params2['cate_num'] = net_params2['in_dim']

# 使用示例：
model_paths = [
               '/home/leihan/workspace/Category_homo_hetero/out/TUs_graph_classification/checkpoints/cpu1/cate_NCI1_GPU1_2024_04_14__18h35m17s/RUN_0/epoch_271.pkl',
     '/home/leihan/workspace/Category_homo_hetero/out/TUs_graph_classification/checkpoints/cpu0/cate_NCI1_GPU0_2024_04_14__18h54m07s/RUN_0/epoch_194.pkl']  # 等等

# 加载模型
json_path1='configs/homohetero_only/homo_MUTAG.json'
json_path2='configs/homohetero_only/hetero_MUTAG.json'
path1=model_paths[0]
path2=model_paths[1]
#
models = [load_model_json(config1['model'],dataset, json_path1, path1, device='cuda' if torch.cuda.is_available() else 'cpu'),
load_model_json(config2['model'],dataset,json_path2, path2, device='cuda' if torch.cuda.is_available() else 'cpu')]

# models = [load_model_json(config1['model'],dataset, json_path1, path1, device='cuda' if torch.cuda.is_available() else 'cpu')]

trainset, valset, testset = dataset.train[split_number], dataset.val[split_number], dataset.test[split_number]
print("trainset, valset, testset ", trainset, valset, testset )
test_loader = DataLoader(testset, batch_size=20, shuffle=False, drop_last=False,collate_fn=dataset.collate)
compare_models(models, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
