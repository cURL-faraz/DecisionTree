import pandas as pd

from scipy.stats import boxcox

import numpy as np 

from sklearn.decomposition import PCA

def detect_categories(dataset):
    
    column_categories = {}
    
    for column in dataset.columns :
        if dataset[column].nunique() <= 10 :
            column_categories[column] = sorted(list(dataset[column].unique()))
        else :
            column_categories[column] = None 

    return column_categories

def load_dataset():
    
    dataset = pd.read_csv("../data/Airplane.csv" , index_col = 0)
    column_categories = detect_categories(dataset)
    return dataset , column_categories

def encode_feats(dataset , column_categories):

    dataset_dtypes = dataset.dtypes 
    for feat in column_categories.keys() : 
        if dataset_dtypes[feat] not in [int , float] :
            categories =  column_categories[feat]
            categories = dict([(categories[i] , i) for i in range(len(categories))])
            dataset[feat] = dataset[feat].apply(lambda x : categories[x])

def drop_feats(dataset , feats_list , column_categories):
    
    for feat in feats_list:
        try :
            dataset = dataset.drop(feat , axis = 1)
            del column_categories[feat]
        except :
            print(f"{feat} isn't a valid column to be dropped !") 

    return dataset

def numerical_fillna(dataset , feats_list):
    
    for feat in feats_list :
        try :
            dataset[feat] = dataset[feat].fillna(dataset[feat].median())
        except : 
            print(f"{feat} isn't a valid column for numerical filling of missing values !") 

def skew_reduction(dataset , feats_list):

    for feat in feats_list :
        try :
            dataset[feat] = boxcox(dataset[feat]+1)[0]
        except :
            print(f"{feat} isn't a valid column for reducing skew !")

def quantile_binning(dataset , feats_dict , column_categories):
    
    for feat in feats_dict.keys() :
        k = feats_dict[feat]
        try :
            quantiles = dataset[feat].quantile(np.linspace(0,1,k+1)).values 
            quantiles = np.unique(quantiles)
            column_categories[feat] = quantiles.tolist()
            dataset[feat] = (np.searchsorted(quantiles , dataset[feat] , side = "right") - 1 ).astype(int)
            dataset[feat] = np.clip(dataset[feat] , 0 , k-1)
        except :
            print(f"{feat} isn't a valid column for quatile binning !")

def equal_width_binning(dataset , feats_dict , column_categories):
    
    for feat in feats_dict.keys():
        k = feats_dict[feat]
        try :
            bins = np.linspace(dataset[feat].min() , dataset[feat].max() , k+1)
            column_categories[feat] = bins.tolist()
            dataset[feat] = (np.searchsorted(bins , dataset[feat] , side = "right") - 1).astype(int)
        except :
            print(f"{feat} isn't a valid column for equal width binning !")

def merge_by_mean(dataset , feats_list):
    
    feat_col = pd.Series([0 for _ in range(dataset.shape[0])])

    for feat in feats_list :
        try :
            feat_col += dataset[feat]
        except :
            print(f"{feat} isn't a valid column for merging by mean! ")
    try :
        return feat_col / len(feats_list)
    except :
        print("zero division due to empty features list !")

def merge_by_pca(dataset , feats_list):
    
    pca = PCA(n_components = 1)
    try :
        sub_df = dataset.loc[: , feats_list]
    except :
        pass 

    for feat in feats_list :
        try :
            miu = sub_df[feat].mean()
            sigma = sub_df[feat].std()
            sub_df[feat] = sub_df[feat].apply(lambda x : (x-miu) / sigma)
        except :
            print(f"{feat} isn't a valid column for merging by PCA !")
    try :
        super_feat = pd.Series((pca.fit_transform(sub_df)).flatten())
        return super_feat
    except :
        print(f"merging by PCA using these {feats_list} !")

def merge_into_super_feat(dataset , feats_dict , column_categories):
    
    for feats_tuple in feats_dict.keys():
        try :
            method = feats_dict[feats_tuple][0]
            super_feat_name = feats_dict[feats_tuple][1]
        except :
            print("pair of merging method and super feature name is invalid ! ")

        if method == "Mean" : 
            dataset[super_feat_name] = merge_by_mean(dataset , feats_tuple)
        else :
            dataset[super_feat_name] = merge_by_pca(dataset ,feats_tuple)
        quantile_binning(dataset , {super_feat_name : 4} , column_categories)
        dataset = drop_feats(dataset , feats_tuple , column_categories)
    return dataset


def dataset_preprocessing(drop_feats_list,numerical_fillna_feats_list,skewed_feats_list,quantile_feats_dict,equal_width_feats_dict,merging_feats_dict):
    
    dataset , column_categories = load_dataset()
    dataset = drop_feats(dataset , drop_feats_list , column_categories)
    encode_feats(dataset , column_categories)
    numerical_fillna(dataset , numerical_fillna_feats_list)
    skew_reduction(dataset , skewed_feats_list)
    quantile_binning(dataset , quantile_feats_dict , column_categories)
    equal_width_binning(dataset , equal_width_feats_dict , column_categories)
    dataset = merge_into_super_feat(dataset , merging_feats_dict , column_categories)
    return dataset , column_categories