#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/10/27 14:33
# @File    : GNN.py
# @Software: PyCharm
import os
import time

cmd = "pip install networkx dgl"
cmd1 = "export DGLBACKEND=\"tensorflow\""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.system(cmd)
os.system("python -m dgl.backend.set_default_backend \"tensorflow\"")
os.system(cmd1)


import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
from tensorflow import keras
import s3fs
import dgl
import argparse
from dgl.nn import GraphConv


class Net(keras.Model):
    def __init__(self,args,input_dim):
        super(Net, self).__init__()
        self.args = args
        self.input_dim = input_dim  # 数据的输入维度
        self.hidden_nodes = args.hidden_nodes.split(',')
        if len(self.hidden_nodes) == 1:  # 单层GCN
            self.cov1 = GraphConv(self.input_dim, int(self.hidden_nodes[0]), norm='both', weight=True, bias=True,
                                  allow_zero_in_degree=True)
            self.cov2 = GraphConv(int(self.hidden_nodes[0]), self.args.dim, norm='both', weight=True, bias=True,
                                  allow_zero_in_degree=True)
        elif len(self.hidden_nodes) == 2:  # 双层GCN
            self.cov1 = GraphConv(self.input_dim, int(self.hidden_nodes[0]), norm='both', weight=True, bias=True,
                                  allow_zero_in_degree=True)
            self.cov2 = GraphConv(int(self.hidden_nodes[0]), int(self.hidden_nodes[1]), norm='both', weight=True,
                                  bias=True,
                                  allow_zero_in_degree=True)
            self.cov3 = GraphConv(int(self.hidden_nodes[1]), self.args.dim, norm='both', weight=True, bias=True,
                                  allow_zero_in_degree=True)
        elif len(self.hidden_nodes) == 3:  # 三层GCN
            self.cov1 = GraphConv(self.input_dim, int(self.hidden_nodes[0]), norm='both', weight=True, bias=True,
                                  allow_zero_in_degree=True)
            self.cov2 = GraphConv(int(self.hidden_nodes[0]), int(self.hidden_nodes[1]), norm='both', weight=True,
                                  bias=True,
                                  allow_zero_in_degree=True)
            self.cov3 = GraphConv(int(self.hidden_nodes[1]), int(self.hidden_nodes[2]), norm='both', weight=True,
                                  bias=True,
                                  allow_zero_in_degree=True)
            self.cov4 = GraphConv(int(self.hidden_nodes[2]), self.args.dim, norm='both', weight=True, bias=True,
                                  allow_zero_in_degree=True)
    def call(self, g, feat):
        if self.args.activation_fun == "relu":
            activation_fun = tf.nn.relu
        elif self.args.activation_fun == "sigmoid":
            activation_fun = tf.nn.sigmoid
        elif self.args.activation_fun == "tanh":
            activation_fun = tf.nn.tanh
        elif self.args.activation_fun == "leaky_relu":
            activation_fun = tf.nn.leaky_relu
        elif self.args.activation_fun == "elu":
            activation_fun = tf.nn.elu
        elif self.args.activation_fun == "gelu":
            activation_fun = tf.nn.gelu
        elif self.args.activation_fun == "relu6":
            activation_fun = tf.nn.relu6
        elif self.args.activation_fun == "silu":
            activation_fun = tf.nn.silu
        if len(self.hidden_nodes) == 1:  # 单层GCN
            h = self.cov1(g, feat)
            h = activation_fun(h)
            h = self.cov2(g, h)
        elif len(self.hidden_nodes) == 2:  # 双层GCN
            h = self.cov1(g, feat)
            h = activation_fun(h)
            h = self.cov2(g, h)
            h = activation_fun(h)
            h = self.cov3(g, h)
        elif len(self.hidden_nodes) == 3:  # 三层GCN
            h = self.cov1(g, feat)
            h = activation_fun(h)
            h = self.cov2(g, h)
            h = activation_fun(h)
            h = self.cov3(g, h)
            h = activation_fun(h)
            h = self.cov4(g, h)
        return h

def Loss(res, data, args):
    result = tf.norm(tf.matmul(res,tf.transpose(res)) - data,2)
    return result

class S3FileSystemPatched(s3fs.S3FileSystem):
    def __init__(self, *k, **kw):
        super(S3FileSystemPatched, self).__init__(*k,
                                                  key=os.environ['AWS_ACCESS_KEY_ID'],
                                                  secret=os.environ['AWS_SECRET_ACCESS_KEY'],
                                                  client_kwargs={'endpoint_url': 'http://' + os.environ['S3_ENDPOINT']},
                                                  **kw
                                                  )
class S3Filewrite:
    def __init__(self, args):
        super(S3Filewrite, self).__init__()
        self.output_path = args.data_output
        self.num = args.csv_sample_num

    def write(self, data):
        data = np.array(data)  # 转化为np.array类型
        s3fs.S3FileSystem = S3FileSystemPatched
        fs = s3fs.S3FileSystem()
        start = time.time()
        data_len = data.shape[0] #数据的长度
        n = self.num #每个csv文件的数据量
        for i in range(data_len):
            file_idx = int(i/n)
            with fs.open(self.output_path + 'result_{}.csv'.format(file_idx), mode="a") as resultfile:
                line = "{},{},{}\n".format(data[i,0], data[i,1], data[i,2])
                resultfile.write(line)
        cost = time.time() - start
        print("write {} lines with {:.2f}s".format(len(data), cost))

def reshapeGraph(data):
    '''
    当图中节点编号不连续时，对图中的节点进行重新编号
    data:图的边矩阵，n行2列,每一行表示一条边,narrary类型
    '''
    nodeset = np.unique(data)
    row_len = data.shape[0]
    col_len = data.shape[1]
    # 重新对节点进行编号
    for i in range(row_len):
        for j in range(col_len):
            data[i, j] = int(np.argwhere(nodeset == data[i, j])[0, 0])
    return data, nodeset

def readts3fToarrary(input_files):#读取图的边数据
    df = pd.read_csv("s3://" + input_files, sep=',', header=None)
    return df.to_numpy()

if __name__ == "__main__":
    # 需要传入的参数
    parser = argparse.ArgumentParser(description='算法的参数')
    parser.add_argument("--signal", help="是否是无向图(0:无向图, 1:有向图)", type=int, default=0)
    parser.add_argument("--activation_fun", help="激活函数(sigmoid,relu,tanh,leaky_relu,elu,gelu,relu6,silu)",
                        type=str, default='relu')
    parser.add_argument("--dim", help="数据的输出维数", type=int, default=32)
    parser.add_argument("--files_num", help="每次读取的文件数目(防止内存溢出)", type=int, default=10)
    parser.add_argument("--hidden_nodes", help="各个隐含层神经元数目，格式如‘200,100,32’表示该GNN有3个隐含层,每个隐含层的节点数分别为200,100,32",
                        type=str, default='300,150,64')
    parser.add_argument("--lr", help="学习率", type=float, default=0.01)
    parser.add_argument("--stop_num", help="执行Early Stopping的最低epoch", type=int, default=10)
    parser.add_argument("--epoch", help="迭代次数", type=int, default=500)
    parser.add_argument("--csv_sample_num", help="输出的.csv文件每个文件所写的数据量", type=int, default=500000)
    parser.add_argument("--data_input", help="输入数据的位置", type=str, default='')
    parser.add_argument("--data_output", help="数据的输出位置", type=str, default='')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')
    args = parser.parse_args()
    start = time.time()
    '''读取s3fs数据部分'''
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    #result = []
    """get file list"""
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    print("数据被分割装入{}个子文件.".format(len(input_files)))
    count = 0
    for file in input_files:
        count = count + 1
        print("当前正在处理{}个文件,文件路径:{}......".format(count,file))
        data = readts3fToarrary(file)
        edges_data, nodeset = reshapeGraph(data)
        edges = [tuple(x) for x in edges_data.tolist()]
        node_num = len(nodeset)
        feat = tf.eye(node_num)  # 一共有node_num个节点
        g = nx.Graph()
        g.add_nodes_from([i for i in range(node_num)])  # 读入节点
        g.add_edges_from(edges)  # 读入边
        net = Net(args, g.number_of_nodes())  # 定义神经网络
        optimizer=keras.optimizers.Adagrad(learning_rate=args.lr)
        before_loss = 2 ** 63 - 1  # 上一次的loss
        before_res = 0  # 上一次的结果
        stop_num = 0  # 若stop_num达到一定数目，此时loss不下降，则执行early stopping
        if args.signal == 1:#有向图
            G = dgl.from_networkx(g)
        else:#无向图
            G = dgl.to_bidirected(dgl.from_networkx(g))
        for epoch in range(args.epoch):
            with tf.GradientTape() as tape:
                res = net(G,feat)
                loss = Loss(res, tf.convert_to_tensor(nx.adjacency_matrix(g).todense(),dtype=tf.float32),args)
                print("epoch:{} loss={}".format(epoch, loss.numpy()))
                if loss < before_loss:
                    before_loss = loss
                    before_res = res
                    stop_num = 0
                else:
                    stop_num = stop_num + 1
                    if stop_num >= args.stop_num:
                        print("Early stopping!")
                        break
            gradients = tape.gradient(loss, net.trainable_variables)
            optimizer.apply_gradients(zip(gradients, net.trainable_variables))
        res = (before_res -tf.reduce_min(before_res,0))/(tf.reduce_max(before_res,0)-tf.reduce_min(before_res,0)) # 节点的特征,进行了归一化
        similarity = []
        for i in range(node_num):
            for j in range(i + 1):
                src_id = int(nodeset[i])  # 源节点在图中的id
                dst_id = int(nodeset[j])  # 目标节点在图中的id
                sim_score = tf.tensordot(res[i, :],res[j, :],1)/(tf.norm(res[i, :], 2)*tf.norm(res[j, :], 2)) # 余弦相似度
                #print("源节点:{} 目标节点:{} 两个节点之间的相似度:{}".format(src_id,dst_id,sim_score.numpy()))
                similarity.append([str(src_id), str(dst_id), sim_score.numpy()])  # 记录结果
        del res
        del net
        s3fswrite = S3Filewrite(args)
        s3fswrite.write(similarity)
        del similarity
    end = time.time()
    print("Data processing is completed！")
    print("The time cost is", end - start)
