import numpy as np 

import pandas as pd 

class Node :
    
    def __init__(self , feats_dict , depth ):
    
        self.feats_dict = feats_dict 
        self.depth = depth 
        self.child = {}
        self.selected_feat = None
        self.crit = {"gini" : float('inf') , 'gain' : float('-inf')}
        self.pruned = False
        self.output_class = None 

class DecisionTree : 
    
    def __init__(self):
        
        self.nodes = {}
        self.id = 0

    def add_node(self , vertex) :
        
        vertex.id = self.id 
        self.id += 1 
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
               
    def conditional_probs_squared(self , child_data) : 

        prob_df = child_data.groupby('satisfaction').agg({"satisfaction" : "count"})
        prob_df['satisfaction'] /= prob_df['satisfaction'].sum()
        prob_df['satisfaction'] **= 2
        return 1 - (prob_df['satisfaction'].sum())
    
    def cal_entropy(self , data) :
        
        prob_df = data.groupby('satisfaction').agg({"satisfaction" : "count"})
        prob_df['satisfaction'] /= prob_df['satisfaction'].sum()
        prob_df['satisfaction'] = np.where(prob_df['satisfaction'] > 0 , prob_df['satisfaction'] * np.log2(prob_df['satisfaction']) , 0)
        return (-1) * (prob_df['satisfaction'].sum())
    
    def cal_gini_index(self , vertex , feat , vertex_data ):
        
        gini = 0 
        for category in vertex.feats_dict[feat] :
            child_data = vertex_data[vertex_data[feat] == category]
            if not child_data.empty :
                gini += (child_data.shape[0] / vertex_data.shape[0]) * self.conditional_probs_squared(child_data)

        return gini 

    def cal_gained_information(self , vertex , feat , vertex_data):
        
        childs_entropy = 0 
        for category in vertex.feats_dict[feat] :
            child_data = vertex_data[vertex_data[feat] == category]
            if not child_data.empty :
                childs_entropy += (child_data.shape[0] / vertex_data.shape[0]) * self.cal_entropy(child_data)
        
        parent_entropy = self.cal_entropy(vertex_data)
        return parent_entropy - childs_entropy
    
    def find_output_class(self , vertex_data):
        
        num_0 = (vertex_data['satisfaction'] == 0 ).sum()
        if num_0 == 0.5 * vertex_data.shape[0] :
            return np.random.choice([0,1])
        else :
            return int(num_0 < 0.5 * vertex_data.shape[0]) 

    def tree_generation(self,vertex,vertex_data):
        
        if self.impurity_check(vertex_data) and self.depth_check(vertex.depth) and self.min_split_check(vertex_data) and len(vertex.feats_dict) > 0 :
            for feat in vertex.feats_dict.keys() :
                if self.hyper_params["criterion"] == "gini" : 
                    feat_gini = self.cal_gini_index(vertex , feat , vertex_data)
                    if feat_gini <= vertex.crit["gini"] :
                        vertex.crit["gini"] = feat_gini 
                        vertex.selected_feat = feat 
                else :
                    feat_gain = self.cal_gained_information(vertex , feat , vertex_data)
                    if feat_gain >= vertex.crit["gain"] :
                        vertex.crit["gain"] = feat_gain
                        vertex.selected_feat = feat 

            vertex.output_class = self.find_output_class(vertex_data)
            if vertex.selected_feat is not None  :
                for category in vertex.feats_dict.pop(vertex.selected_feat):
                    vertex.child[category] = Node(vertex.feats_dict.copy() , vertex.depth + 1)
                    self.add_node(vertex.child[category])
                    self.tree_generation(vertex.child[category],vertex_data[vertex_data[vertex.selected_feat] == category])
        else :
            vertex.output_class = self.find_output_class(vertex_data)
    
    def training(self , train_data , hyper_params , feats_dict):
        
        self.root = Node( feats_dict , 0 )
        self.add_node(self.root)
        self.hyper_params = hyper_params
        self.tree_generation(self.root , train_data)

    def tree_traverse(self , sample , vertex):
        
        if len(vertex.child.keys()) > 0 and not vertex.pruned :
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
        try : 
            return TP / (TP + FP)
        except :
            return 0 
        
    def cal_recall(self , predictions , test_data_Y):
        
        TP = (predictions & test_data_Y).sum()
        FN = ((predictions ^ test_data_Y) & test_data_Y).sum()
        try :
            return TP / (TP + FN)
        except : 
            return 0 

    def cal_F1(self , predictions , test_data_Y):

        precision = self.cal_precision(predictions , test_data_Y)
        recall = self.cal_recall(predictions , test_data_Y)
        try :
            return 2 * precision * recall / (precision + recall)
        except :
            return 0 
    
    def cal_accuracy(self , predictions , test_data_Y):

        return (((predictions & test_data_Y) | (~predictions & ~test_data_Y)).sum()) / (test_data_Y.shape[0])

    def tree_evaluation(self , test_data ):
        
        test_data_Y = test_data['satisfaction'].astype(bool)
        test_data_X = test_data.drop(["satisfaction"] , axis = 1)
        predictions = self.predict(test_data_X).astype(bool)
        F1_Score = self.cal_F1(predictions , test_data_Y)
        acc = self.cal_accuracy(predictions , test_data_Y)
        return F1_Score , acc 
    
    def tree_visualization(self , vertex , graph , original_column_categories) :
        if vertex.selected_feat is None :
            graph.node(str(vertex.id) , f'Class : {original_column_categories['satisfaction'][vertex.output_class]}')
        else :
            graph.node(str(vertex.id) , f'Feature : {vertex.selected_feat}\t{self.hyper_params["criterion"]} : {vertex.crit[self.hyper_params["criterion"]]}\ttested features : {list(vertex.feats_dict.keys())}')
            for category in vertex.child.keys():     
                graph.edge(str(vertex.id),str(vertex.child[category].id),f'Category : {original_column_categories[vertex.selected_feat][category]}')
                self.tree_visualization(vertex.child[category] , graph , original_column_categories)

    def tree_post_pruning(self , primary_F1 , primary_Acc , test_data):
        
        for depth in sorted(self.nodes.keys() , reverse = True)[1:-1] :
            for node in self.nodes[depth] :
                if len(node.child) > 0 :
                    node.pruned = True 
                    current_F1 , current_Acc = self.tree_evaluation(test_data) 
                    node.pruned = (( current_F1 * 0.3 + current_Acc * 0.7 ) >= (primary_F1 * 0.3 + primary_Acc * 0.7 ) )
