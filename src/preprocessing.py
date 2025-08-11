import pandas as pd

from scipy.stats import boxcox

import numpy as np 

from sklearn.decomposition import PCA

def detect_categories(dataset):
    
    column_categories = {}
    
    for column in dataset.columns :
        if dataset[column].nunique() <= 10 :
            column_categories[column] = list(dataset[column].unique())
        else :
            column_categories[column] = None 

    return column_categories

def load_dataset():
    
    dataset = pd.read_csv("../data/Airplane.csv" , index_col = 0)
    column_categories = detect_categories(dataset)
    return dataset , column_categories

def encode_feats(dataset):

    dataset_dtypes = dataset.dtypes 

    for feat in dataset.columns : 
        if dataset_dtypes[feat] not in [int , float] :
            categories = list(dataset[feat].unique())
            categories = dict([(categories[i] , i) for i in range(len(categories))])
            dataset[feat] = dataset[feat].apply(lambda x : categories[x])

def drop_feats(dataset , feats_list , column_categories):
    
    for feat in feats_list:
        dataset = dataset.drop(feat , axis = 1)
        del column_categories[feat]

    return dataset

def numerical_fillna(dataset , feats_list):
    
    for feat in feats_list :
        dataset[feat] = dataset[feat].fillna(dataset[feat].median())

def skew_reduction(dataset , feats_list):

    for feat in feats_list :
        dataset[feat] = boxcox(dataset[feat]+1)[0]

def quantile_binning(dataset , feats_dict , column_categories):
    
    for feat in feats_dict.keys() :
        k = feats_dict[feat]
        quantiles = dataset[feat].quantile(np.linspace(0,1,k+1)).values 
        quantiles = np.unique(quantiles)
        column_categories[feat] = quantiles.tolist()
        dataset[feat] = (np.searchsorted(quantiles , dataset[feat] , side = "right") - 1 ).astype(int)
        dataset[feat] = np.clip(dataset[feat] , 0 , k-1)

def equal_width_binning(dataset , feats_dict , column_categories):
    
    for feat in feats_dict.keys():
        k = feats_dict[feat]
        bins = np.linspace(dataset[feat].min() , dataset[feat].max() , k+1)
        column_categories[feat] = bins.tolist()
        dataset[feat] = (np.searchsorted(bins , dataset[feat] , side = "right") - 1).astype(int)

def merge_by_mean(dataset , feats_list):
    
    feat_col = pd.Series([0 for _ in range(dataset.shape[0])])

    for feat in feats_list :
        feat_col += dataset[feat]
    
    return feat_col / len(feats_list)

def merge_by_pca(dataset , feats_list):
    
    pca = PCA(n_components = 1)
    sub_df = dataset.loc[: , feats_list]

    for feat in feats_list :
        miu = sub_df[feat].mean()
        sigma = sub_df[feat].std()
        sub_df[feat] = sub_df[feat].apply(lambda x : (x-miu) / sigma)

    super_feat = pd.Series((pca.fit_transform(sub_df)).flatten())
    return super_feat

def merge_into_super_feat(dataset , feats_dict , column_categories):
    
    for feats_tuple in feats_dict.keys():
        method = feats_dict[feats_tuple][0]
        super_feat_name = feats_dict[feats_tuple][1]
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
    encode_feats(dataset)
    numerical_fillna(dataset , numerical_fillna_feats_list)
    skew_reduction(dataset , skewed_feats_list)
    quantile_binning(dataset , quantile_feats_dict , column_categories)
    equal_width_binning(dataset , equal_width_feats_dict , column_categories)
    dataset = merge_into_super_feat(dataset , merging_feats_dict , column_categories)
    return dataset , column_categories