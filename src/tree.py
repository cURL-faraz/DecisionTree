import numpy as np 

import pandas as pd 

class Node :
    
    def __init__(self,depth,data = None , feats_dict = None):
        
        self.feats_dict = feats_dict
        self.data = data
        self.depth = depth 
        self.child = {}
        self.selected_feat = None
        self.gini = float('inf')
        self.gain = 0 
    
class DecisionTree : 
    
    def __init__(self):
        
        self.root = Node(0)
        self.nodes = {0 : [self.root]}
        self.num_leaf = 0 

    def add_node(self , vertex) :
        
        try : 
            self.nodes[vertex.depth].append(vertex)
        except : 
            self.nodes[vertex.depth] = [vertex]

    def impurity_check(self,vertex_data):

        return (vertex_data["satisfaction"] == 0 ).sum() not in [0 , vertex_data.shape[0]]

    def depth_check(self , vertex_depth):
        
        return (self.hyper_params["max_depth"] is None) or (vertex_depth < self.hyper_params["max_depth"])
    
    def min_split_check(self , vertex_data):

        return vertex_data.shape[0] >= self.hyper_params["min_samples_split"]
    
    def soft_max_leaf_nodes_check(self):

        return self.num_leaf + 1 <= self.hyper_params["soft_max_leaf_nodes"]
    
    def soft_min_samples_leaf(self , child_data):

        return child_data.shape[0] >= self.hyper_params["soft_min_samples_leaf"]
    
    def valid_leaf_check(self , parent , child_data , child_depth):
        
        if self.impurity_check(child_data) and self.depth_check(child_depth) and self.min_split_check(child_data) and len(parent.feats_dict.keys()) > 1:
            # exception :(
            return True 
        else :
            return self.soft_max_leaf_nodes_check() and self.soft_min_samples_leaf(child_data)
                
    def conditional_probs_squared(self , child_data) : 

        prob_df = child_data.groupby('satisfaction').agg({"satisfaction" : "count"})
        prob_df['satisfaction'] /= prob_df['satisfaction'].sum()
        prob_df['satisfaction'] **= 2
        return 1 - (prob_df['satisfaction'].sum())
    
    def cal_entropy(self , data) :
        
        prob_df = data.groupby('satisfaction').agg({"satisfaction" : "count"})
        prob_df['satisfaction'] /= prob_df['satisfaction'].sum()
        prob_df['satisfaction'] *= np.log2(prob_df['satisfaction'])
        return (-1) * (prob_df['satisfaction'].sum())
    
    def cal_gini_index(self , vertex , feat , categories):
        
        gini = 0 

        for category in categories :
            child_data = vertex.data[vertex.data[feat] == category]
            child_depth = vertex.depth + 1 
            if self.valid_leaf_check(vertex , child_data , child_depth) :
                gini += (child_data.shape[0] / vertex.data.shape[0]) * self.conditional_probs_squared(child_data)
            else :
                return None
        
        return gini 

    def cal_gained_information(self , vertex , feat , categories):
        
        childs_entropy = 0 

        for category in categories :
            child_data = vertex.data[vertex.data[feat] == category]
            child_depth = vertex.depth + 1
            if self.valid_leaf_check(vertex , child_data , child_depth) :
                childs_entropy += (child_data.shape[0] / vertex.data.shape[0]) * self.cal_entropy(child_data)
            else :
                return None 
        
        parent_entropy = self.cal_entropy(vertex.data)
        return parent_entropy - childs_entropy
    
    def new_leaf(self , vertex) :

        self.num_leaf += 1 
        num_0 = (vertex.data["satisfaction"] == 0).sum()
        vertex.output_class = int(num_0 < (0.5 * vertex.data.shape[0])) + np.random.choice([0 , int(num_0 == (0.5 * vertex.data.shape[0]))])

    def tree_generation(self,vertex):
        
        if self.impurity_check(vertex.data) and self.depth_check(vertex.depth) and self.min_split_check(vertex.data) and len(vertex.feats_dict.keys()) > 0 :
            
            # refactor 
            for feat in vertex.feats_dict.keys() :
                if self.hyper_params["criterion"] == "gini" : 
                    feat_gini = self.cal_gini_index(vertex , feat , vertex.feats_dict[feat])
                    if feat_gini is not None and feat_gini <= vertex.gini :
                        vertex.gini = feat_gini 
                        vertex.selected_feat = feat 
                else :
                    feat_gain = self.cal_gained_information(vertex , feat , vertex.feats_dict[feat])
                    if feat_gain is not None and feat_gain >= vertex.gain :
                        vertex.gain = feat_gain
                        vertex.selected_feat = feat 
            
            # exception :( 
            if vertex.selected_feat is None :
                self.new_leaf(vertex)
            else :
                for category in vertex.feats_dict.pop(vertex.selected_feat):
                    vertex.child[category] = Node(vertex.depth + 1 , vertex.data[vertex.data[vertex.selected_feat] == category] , vertex.feats_dict.copy())
                    self.add_node(vertex.child[category])
                    self.tree_generation(vertex.child[category])
        else :
            self.new_leaf(vertex)

    def training(self , train_data , hyper_params , feats_dict):
        
        self.root.data = train_data 
        self.hyper_params = hyper_params
        self.root.feats_dict = feats_dict
        self.tree_generation(self.root)

    def tree_traverse(self , sample , vertex):
        if len(vertex.child.keys()) > 0 :
            return self.tree_traverse(sample , vertex.child[sample[vertex.selected_feat]]) 
        else :
            return vertex.output_class 

    def predict(self , test_data_X):
        
        predictions = [] 

        for _ , sample in test_data_X.iterrows():
            predictions.append(self.tree_traverse(sample , self.root))

        return pd.Series(predictions)
    
    def cal_precision(self , predictions , test_data_Y):
        
        TP = (predictions & test_data_Y).sum()
        FP = ((predictions ^ test_data_Y) & predictions).sum()
        return TP / (TP + FP)
    
    def cal_recall(self , predictions , test_data_Y):
        
        TP = (predictions & test_data_Y).sum()
        FN = ((predictions ^ test_data_Y) & test_data_Y).sum()
        return TP / (TP + FN)
    
    def F1_evaluation(self , test_data ):
        
        test_data_Y = test_data['satisfaction']
        test_data_X = test_data.drop(["satisfaction"] , axis = 1)
        predictions = self.predict(test_data_X)
        Precision = self.cal_precision(predictions , test_data_Y)
        Recall = self.cal_recall(predictions , test_data_Y)
        return 2 * Precision * Recall / (Precision + Recall )


