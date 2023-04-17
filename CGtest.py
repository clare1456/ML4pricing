'''
File: test.py
Project: ML4pricing
File Created: Sunday, 16th April 2023 3:03:40 pm
Author: Charles Lee (lmz22@mails.tsinghua.edu.cn)
'''

import sys
sys.path.append("D:\Code\ML4pricing\CGAlgs")
import numpy as np
import pandas as pd
import GraphTool
import ColumnGeneration
from Net import GAT

class CGWithPricing(ColumnGeneration.ColumnGenerationWithLabeling):
    def __init__(self, graph):
        super(CGWithPricing, self).__init__(graph)
        self.OutputFlag = False  
    
    def solve_RLMP_and_get_duals(self):
        is_feasible = super(CGWithPricing, self).solve_RLMP_and_get_duals()
        if not is_feasible:
            return None
        duals = np.zeros(self.graph.nodeNum)
        for i in range(self.graph.nodeNum):
            cons_name = "R{}".format(i)
            duals[i] = self.duals_of_RLMP[cons_name]
        return duals
    
    def column_generation_after_pricing(self, fixed_duals):
        # set fixed duals
        for i in range(self.graph.nodeNum):
            cons_name = "R{}".format(i)
            self.duals_of_RLMP[cons_name] = fixed_duals[i]
        # solve SP
        time1 = time.time()
        self.solve_SP()
        self.timeRecord += time.time() - time1
        # record information
        self.cg_iter_cnt += 1
        self.output_info()
        # break if can't improve anymore
        if self.SP_obj >= -self.EPS:
            return 1
        # get columns and add into RLMP
        self.get_columns_and_add_into_RLMP()


class CGtestor:
    def __init__(self, args, model, save_path=None):
        self.args = args
        self.model = model
        self.save_path = save_path
        self.file_list = []
    
    def origin_test(self, graph):
        CGAlg = ColumnGeneration.ColumnGenerationWithLabeling(graph)
        CGAlg.run()
        return CGAlg.cg_iter_cnt, CGAlg.RLMP_obj
    
    def model_test(self, graph, model):
        CGAlg = ColumnGeneration.ColumnGenerationWithLabeling(graph)
        while True:
            # solve RLMP and get original duals
            duals = CGAlg.solve_RLMP_and_get_duals()
            if duals is None:
                raise Exception("Model Infeasible")
            # model predict and fix duals
            offsets = model(duals).detach().numpy()
            fixed_duals = duals + offsets
            # solve SP and get columns
            CGAlg.column_generation_after_pricing(fixed_duals)
        return CGAlg.cg_iter_cnt, CGAlg.RLMP_obj
    
    def run(self):
        df = pd.DataFrame(columns=['file_name', 'origin_iter_cnt', 'origin_obj', 'model_iter_cnt', 'model_obj'])
        for file_name in self.file_list:
            graph = GraphTool.Graph(self.args.data_path + file_name + '.json') 
            origin_iter_cnt, origin_obj = self.origin_test(graph)
            model_iter_cnt, model_obj = self.model_test(graph, self.model)
            print(f"{file_name}: \n column_generation : {origin_iter_cnt} {origin_obj} \n ml4pricing : {model_iter_cnt} {model_obj}")
            df = df.append({'file_name': file_name, 'origin_iter_cnt': origin_iter_cnt, 'origin_obj': origin_obj, 'model_iter_cnt': model_iter_cnt, 'model_obj': model_obj}, ignore_index=True)
        if self.save_path:
            df.to_csv(self.save_path + 'CGtest_result.csv', index=False)

if __name__ == "__main__":
    from utils import TestArgs
    args = TestArgs()
    model_path = ""
    model = GAT(args)
    if model_path:
        model.load_model(path=model_path)
    testor = CGtestor(args, model)    
    testor.run()
    
            

