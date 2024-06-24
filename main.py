import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from nets.load_net import gnn_model # import GNNs
from data.data import LoadData # import dataset


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

def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param

def train_val_pipeline(dataset,MODEL_NAME, DATASET_NAME, params, net_params, dirs,args,dirs_teacher=None):
    avg_test_acc = []
    avg_train_acc = []
    avg_convergence_epochs = []

    t0 = time.time()
    per_epoch_time = []

    # dataset = LoadData(DATASET_NAME, net_params['cate_reorder'],net_params['special_node_version'])

    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()

    # trainset, valset, testset = dataset.train, dataset.val, dataset.test

    root_log_dir, root_ckpt_dir, write_file_name, write_config_file ,root_vis_dir = dirs
    if dirs_teacher is not None:
        root_log_dir_t, root_ckpt_dir_t, write_file_name_t, write_config_file_t = dirs_teacher
    device = net_params['device']

    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for split_number in range(10):
            t0_split = time.time()
            log_dir = os.path.join(root_log_dir, "RUN_" + str(split_number))
            writer = SummaryWriter(log_dir=log_dir)

            # setting seeds
            random.seed(params['seed'])
            np.random.seed(params['seed'])
            torch.manual_seed(params['seed'])
            if device.type == 'cuda':
                torch.cuda.manual_seed(params['seed'])

            print("RUN NUMBER: ", split_number)
            trainset, valset, testset = dataset.train[split_number], dataset.val[split_number], dataset.test[split_number]
            print("Training Graphs: ", len(trainset))
            print("Validation Graphs: ", len(valset))
            print("Test Graphs: ", len(testset))
            print("Number of Classes: ", net_params['n_classes'])

            model = gnn_model(MODEL_NAME, net_params)
            model = model.to(device)
            # model.name = MODEL_NAME
            optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=params['lr_reduce_factor'],
                                                             patience=params['lr_schedule_patience'],
                                                             verbose=True)

            epoch_train_losses, epoch_val_losses = [], []
            epoch_train_accs, epoch_val_accs = [], []

            # batching exception for Diffpool
            drop_last = True if MODEL_NAME == 'DiffPool' else False

            if MODEL_NAME in ['RingGNN', '3WLGNN']:
                # import train functions specific for WL-GNNs
                from train_TUs_graph_classification import train_epoch_dense as train_epoch, evaluate_network_dense as evaluate_network

                train_loader = DataLoader(trainset, shuffle=True, collate_fn=dataset.collate_dense_gnn)
                val_loader = DataLoader(valset, shuffle=False, collate_fn=dataset.collate_dense_gnn)
                test_loader = DataLoader(testset, shuffle=False, collate_fn=dataset.collate_dense_gnn)

            else:
                # import train functions for all other GCNs
                from train_TUs_graph_classification import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network

                train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, collate_fn=dataset.collate)
                val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)
                test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)

            with tqdm(range(params['epochs'])) as t:
                for epoch in t:

                    t.set_description('Epoch %d' % epoch)

                    start = time.time()

                    if MODEL_NAME in ['RingGNN', '3WLGNN']: # since different batch training function for dense GNNs
                        epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch, params['batch_size'])
                        epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader, epoch, net_params)
                        _, epoch_test_acc = evaluate_network(model, device, test_loader, epoch, net_params)
                    else:   # for all other models common train function
                        if model.name == 'cate' and net_params['cate_contras'] == True:
                            epoch_train_loss,epoch_cls_loss, epoch_contras_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch,net_params)
                            epoch_val_loss,epoch_val_cls_loss,epoch_val_contras_loss ,epoch_val_acc = evaluate_network(model, device, val_loader, epoch,net_params)
                            _, _,_, epoch_test_acc = evaluate_network(model, device, test_loader, epoch, net_params)

                            writer.add_scalar('train/cls_loss', epoch_cls_loss, epoch)
                            writer.add_scalar('train/contras_loss', epoch_contras_loss, epoch)
                            writer.add_scalar('val/cls_loss', epoch_val_cls_loss, epoch)
                            writer.add_scalar('val/contras_loss', epoch_val_contras_loss, epoch)

                        else:
                            epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch,net_params)
                            epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader, epoch,net_params)
                            _, epoch_test_acc = evaluate_network(model, device, test_loader, epoch,net_params)

                    epoch_train_losses.append(epoch_train_loss)
                    epoch_val_losses.append(epoch_val_loss)
                    epoch_train_accs.append(epoch_train_acc)
                    epoch_val_accs.append(epoch_val_acc)

                    writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                    writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                    writer.add_scalar('train/_acc', epoch_train_acc, epoch)
                    writer.add_scalar('val/_acc', epoch_val_acc, epoch)
                    writer.add_scalar('test/_acc', epoch_test_acc, epoch)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                    t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                                  train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                                  train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                                  test_acc=epoch_test_acc)

                    per_epoch_time.append(time.time()-start)

                    # Saving checkpoint
                    ckpt_dir = os.path.join(root_ckpt_dir, "RUN_" + str(split_number))
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                    files = glob.glob(ckpt_dir + '/*.pkl')
                    for file in files:
                        epoch_nb = file.split('_')[-1]
                        epoch_nb = int(epoch_nb.split('.')[0])
                        if epoch_nb < epoch-1:
                            os.remove(file)

                    scheduler.step(epoch_val_loss)

                    if optimizer.param_groups[0]['lr'] < params['min_lr']:
                        print("\n!! LR EQUAL TO MIN LR SET.")
                        break

                    # Stop training after params['max_time'] hours
                    if time.time()-t0_split > params['max_time']*3600/10:       # Dividing max_time by 10, since there are 10 runs in TUs
                        print('-' * 89)
                        print("Max_time for one train-val-test split experiment elapsed {:.3f} hours, so stopping".format(params['max_time']/10))
                        break
            if model.name == 'cate' and net_params['cate_contras'] == True:
                _,_,_, test_acc = evaluate_network(model, device, test_loader, epoch,net_params)
                _, _,_,train_acc = evaluate_network(model, device, train_loader, epoch,net_params)
            else:
                _, test_acc = evaluate_network(model, device, test_loader, epoch, net_params)
                _, train_acc = evaluate_network(model, device, train_loader, epoch, net_params)

            avg_test_acc.append(test_acc)
            avg_train_acc.append(train_acc)
            avg_convergence_epochs.append(epoch)

            print("Test Accuracy [LAST EPOCH]: {:.2f}".format(test_acc))
            print("Train Accuracy [LAST EPOCH]: {:.2f}".format(train_acc))
            print("Convergence Time (Epochs): {:.2f}".format(epoch))

            # if args.vis == 'True':
            #     from util import visualize_embeddings
            #     visualize_embeddings(train_loader, test_loader, model, device, args.dataset, train_acc, test_acc,save_path=root_vis_dir)

            # raise SystemError #只跑完第一个epoch的可视化就够了

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')


    print("TOTAL TIME TAKEN: {:.2f}hrs".format((time.time()-t0)/3600))
    print("AVG TIME PER EPOCH: {:.2f}s".format(np.mean(per_epoch_time)))
    print("AVG CONVERGENCE Time (Epochs): {:.2f}".format(np.mean(np.array(avg_convergence_epochs))))
    # Final test accuracy value averaged over 10-fold
    print("""\n\n\nFINAL RESULTS\n\nTEST ACCURACY averaged: {:.2f} ± {:.2f}"""          .format(np.mean(np.array(avg_test_acc))*100, np.std(avg_test_acc)*100))
    print("\nAll splits Test Accuracies:\n", avg_test_acc)
    print("""\n\n\nFINAL RESULTS\n\nTRAIN ACCURACY averaged: {:.2f} ± {:.2f}"""          .format(np.mean(np.array(avg_train_acc))*100, np.std(avg_train_acc)*100))
    print("\nAll splits Train Accuracies:\n", avg_train_acc)
    #
    # if args.vis == 'True':
    #     from util import visualize_embeddings
    #     visualize_embeddings(train_loader, test_loader, model, device,args.dataset, save_path=root_vis_dir)
    writer.close()

    """
        Write the results in out/results folder
    """
    base_file_name, _ = os.path.splitext(write_file_name)
    directory_name = os.path.dirname(write_file_name)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    new_write_file_name = '{}_test_{:.2f}.txt'.format(base_file_name, np.mean(np.array(avg_test_acc)) * 100)

    with open(new_write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST ACCURACY averaged: {:.2f} ± {:.2f}\nTRAIN ACCURACY averaged: {:.2f} ± {:.2f}\n\n
    Average Convergence Time (Epochs): {:.2f} ± {:.2f}\nTotal Time Taken: {:.2f} hrs\nAverage Time Per Epoch: {:.2f} s\n\n\nAll Splits Test Accuracies: {}""" \
                .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                        np.mean(np.array(avg_test_acc))*100, np.std(avg_test_acc)*100,
                        np.mean(np.array(avg_train_acc))*100, np.std(avg_train_acc)*100,
                        np.mean(avg_convergence_epochs), np.std(avg_convergence_epochs),
                        (time.time()-t0)/3600, np.mean(per_epoch_time), avg_test_acc))


def main():
    """
        USER CONTROLS
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--model_teacher',default= 'GCN', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--L_hetero', default=4, help="Please give a value for L_hetero")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--hidden_dim_hetero', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--cate_reorder', default='True', help="the category homo and hetero preprocess")
    parser.add_argument('--cate_version', default='homo', help="homo/hetero/both")
    parser.add_argument('--hetero_conv', default='gcn', help="homo/hetero/both")
    parser.add_argument('--new_homo', default='False', help="if there is a special node for the homo branch")
    parser.add_argument('--special_node_version', default='False', help="if there is a special node for the homo branch")
    parser.add_argument('--teacher_model', default=False, help="homo/hetero/both")
    parser.add_argument('--cate_contras', default=False, help="homo/hetero/both")
    parser.add_argument('--vis', default=False, help="homo/hetero/both")
    parser.add_argument('--grid', default='False', help="if there is a special node for the homo branch")
    parser.add_argument('--homo_layer_num', default=3, help="if there is a special node for the homo branch")
    parser.add_argument('--opt', help="if there is a special node for the homo branch")
    parser.add_argument('--normal_mode', default='zscore', help="zscore/minmax for mode of normalization for homo and hetero")




    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']

    if args.model_teacher is not None:
        MODEL_NAME_teacher = args.model_teacher
    else:
        MODEL_NAME = config['model']

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.L_hetero is not None:
        net_params['L_hetero'] = int(args.L_hetero)
    if args.homo_layer_num is not None:
        net_params['homo_layer_num'] = int(args.homo_layer_num)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.hidden_dim_hetero is not None:
        net_params['hidden_dim_hetero'] = int(args.hidden_dim_hetero)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual=='True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat=='True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated=='True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.opt is not None:
        net_params['opt'] = args.opt
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred=='True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat=='True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop=='True' else False
    if args.cate_reorder is not None:
        net_params['cate_reorder'] = True if args.cate_reorder == 'True' else False
    # net_params['cate_reorder'] = True if args.model in ['cate'] else False

    if args.cate_version is not None:
        net_params['cate_version'] = args.cate_version
    if args.hetero_conv is not None:
        net_params['hetero_conv'] = args.hetero_conv

    if args.teacher_model is not None:
        net_params['teacher_model'] = True if args.teacher_model == 'True' else False
    if args.cate_contras is not None:
        net_params['cate_contras'] = True if args.cate_contras == 'True' else False
    if args.vis is not None:
        net_params['vis'] = True if args.vis == 'True' else False

    if args.new_homo is not None: #new_Homo and special_node_version捆绑销售
        net_params['new_homo'] = True if args.new_homo=='True' else False
    if args.special_node_version is not None and net_params['new_homo']== True:
        net_params['special_node_version'] = True if args.special_node_version=='True' else False
    else:
        net_params['special_node_version'] = False
    if args.grid is not None:
        net_params['grid'] = True if args.grid=='True' else False
    if args.normal_mode is not None:
        net_params['normal_mode'] = args.normal_mode


    dataset = LoadData(DATASET_NAME, net_params['cate_reorder'],net_params['special_node_version'])
    print("dataset", DATASET_NAME, dataset)

    # TUs
    net_params['in_dim'] = dataset.all.graph_lists[0].ndata['feat'][0].shape[0]
    num_classes = len(np.unique(dataset.all.graph_labels))
    net_params['n_classes'] = num_classes
    net_params['cate_num'] = net_params['in_dim']

    if MODEL_NAME == 'DiffPool':
        # calculate assignment dimension: pool_ratio * largest graph's maximum
        # number of nodes  in the dataset
        num_nodes = [dataset.all[i][0].number_of_nodes() for i in range(len(dataset.all))]
        max_num_node = max(num_nodes)
        net_params['assign_dim'] = int(max_num_node * net_params['pool_ratio']) * net_params['batch_size']

    if MODEL_NAME == 'RingGNN':
        num_nodes = [dataset.all[i][0].number_of_nodes() for i in range(len(dataset.all))]
        net_params['avg_node_num'] = int(np.ceil(np.mean(num_nodes)))

    root_log_dir = out_dir + 'logs/cpu{}/'.format(config['gpu']['id']) + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Y_%m_%d__%Hh%Mm%Ss')
    root_ckpt_dir = out_dir + 'checkpoints/cpu{}/'.format(
        config['gpu']['id']) + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Y_%m_%d__%Hh%Mm%Ss')
    write_file_name = out_dir + 'results/cpu{}/result_'.format(
        config['gpu']['id']) + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Y_%m_%d__%Hh%Mm%Ss')
    write_config_file = out_dir + 'configs/config_'+ MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Y_%m_%d__%Hh%Mm%Ss')
    root_vis_dir = out_dir + 'vis'+ "_GPU" + str(config['gpu']['id'])
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file,root_vis_dir




    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    if net_params['teacher_model'] ==True:
        root_log_dir_t = out_dir + 'logs/' +'teacher'+ MODEL_NAME + "_" + DATASET_NAME
        root_ckpt_dir_t = out_dir + 'checkpoints/' + 'teacher'+ MODEL_NAME + "_" + DATASET_NAME
        write_file_name_t = out_dir + 'results/teacher+result_' + MODEL_NAME + "_" + DATASET_NAME
        write_config_file_t = out_dir + 'configs/teacher+config_' + MODEL_NAME + "_" + DATASET_NAME
        dirs_teacher = root_log_dir_t, root_ckpt_dir_t, write_file_name_t, write_config_file_t
    else:
        dirs_teacher=None
    #     if os.path.exists(root_ckpt_dir_t):
    #         teacher_model = gnn_model(MODEL_NAME, net_params)
    #         checkpoint = torch.load(root_ckpt_dir_t)
    #         teacher_model.load_state_dict(checkpoint['model_state_dict'])
    #     else:
    #         teacher_model = train_val_pipeline_teacher(MODEL_NAME_teacher, DATASET_NAME, params, net_params, dirs_teacher, args)
    # else:
    #     teacher_model=None
    train_val_pipeline(dataset, MODEL_NAME, DATASET_NAME, params, net_params, dirs,args,dirs_teacher=dirs_teacher)


main()

