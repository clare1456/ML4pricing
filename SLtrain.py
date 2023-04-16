'''
File: train.py
Project: ML4pricing
File Created: Sunday, 16th April 2023 3:03:47 pm
Author: Charles Lee (lmz22@mails.tsinghua.edu.cn)
'''

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import torch, numpy as np
import matplotlib.pyplot as plt
import math, time, json, datetime
from torch.utils.tensorboard import SummaryWriter
from Net import GAT


class Args:
    def __init__(self):
        ################################## 算法超参数 ####################################
        self.epoch_num = 10 # 训练的回合数
        self.batch_size = 100 # 每次训练的batch大小
        self.learning_rate = 1e-5 # 学习率
        ################################################################################

        ################################# 数据集参数 ################################
        self.file_name = "" # 数据集的文件名
        self.test_size = 0.2 # 测试集的比例
        self.node_num = 100 # 限制节点个数
        self.node_feature_dim = 6 # 节点特征维度
        self.column_feature_dim = 3 + self.node_num # 列特征维度
        ################################################################################

        ################################# 其他参数 ################################
        self.curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间 
        self.curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径 
        self.data_path = self.curr_path+"/data/" # 读取数据集的路径
        self.result_path = self.curr_path+"/outputs/" + self.file_name + \
            '/'+self.curr_time+'/results/'  # 保存结果的路径
        self.model_path = self.curr_path+"/outputs/" + self.file_name + \
            '/'+self.curr_time+'/models/'  # 保存模型的路径
        ################################################################################


class Trainer:
    def __init__(self, args):
        self.args = args
        # set random seed
        if args.seed is not None:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
        # load data
        self.mini_batches = self.read_data(args.file_name)
        self.train_mini_batches, self.test_mini_batches = self.split_data(self.mini_batches)
    
    def read_data(self, file_name):
        """ read data from mini_batches 

            mini_batches: list of mini_batches
            mini_batch : Dict["X", "y"]

        """
        mini_batches = json.load(self.args.data_path + file_name)

    def split_data(self, mini_batches, test_size):
        """ split mini_batches into train / test dataset """
        np.random.shuffle(mini_batches)
        train_mini_batches = mini_batches[:int(len(mini_batches)*(1-test_size))]
        test_mini_batches = mini_batches[int(len(mini_batches)*(1-test_size)):]
        return train_mini_batches, test_mini_batches
    
    def split_batches(self, train_mini_batches):
        """ split train_mini_batches into batches """
        np.random.shuffle(train_mini_batches)
        batches = []
        for i in range(0, len(train_mini_batches), self.args.batch_size):
            batches.append(train_mini_batches[i:i+self.args.batch_size])
        return batches

    def test(self, model, test_mini_batches):
        """ test predict accuracy """
        test_loss_list = []
        for mini_batch in test_mini_batches:
            node_features, column_features, edges = mini_batch["node_features"], mini_batch["column_features"], mini_batch["edges"]
            true_y = mini_batch["dual_offsets"]
            pred_y = self.model(node_features, column_features, edges)
            loss = torch.nn.MSELoss(pred_y, true_y)
            test_loss_list.append(loss.detach().numpy())
        return test_loss_list

    def run(self):
        """ training process """
        # build model
        self.model = GAT(self.args)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.writer = SummaryWriter(self.args.result_path + "tb_logs")
        # train model
        iter_cnt = 0
        for epoch in range(self.args.epoch_num):
            batches = self.split_batches(self.train_mini_batches)
            for batch in batches:
                train_loss_list = []
                pred_list = []
                self.optim.zero_grad()
                for mini_batch in batch:
                    node_features, column_features, edges = mini_batch["node_features"], mini_batch["column_features"], mini_batch["edges"]
                    true_y = mini_batch["dual_offsets"]
                    pred_y = self.model(node_features, column_features, edges)
                    loss = torch.nn.MSELoss(pred_y, true_y)
                    loss.backward()
                    train_loss_list.append(loss.detach().numpy())
                    pred_list += pred_y.detach().numpy().tolist()
                self.optim.step()
                self.writer.add_scalar("loss/train_loss", np.mean(train_loss_list), iter_cnt)
                self.writer.add_scalar("value/mean_pred", np.mean(pred_list), iter_cnt)
                test_loss_list = self.test(self.model, self.test_mini_batches)
                self.writer.add_scalar("loss/test_loss", np.mean(test_loss_list), iter_cnt)
                iter_cnt += 1
            # save model
            self.model.save_model(self.args.model_path + "model_epoch" + str(epoch) + ".pth")
    

if __name__ == "__main__":
    args = Args()
    trainer = Trainer(args)
    trainer.run()