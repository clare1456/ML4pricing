'''
File: utils.py
Project: ML4pricing
File Created: Sunday, 16th April 2023 8:40:02 pm
Author: Charles Lee (lmz22@mails.tsinghua.edu.cn)
'''
import os
import datetime

class TestArgs:
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