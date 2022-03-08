#coding=UTF-8
import os, sys, gzip
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_df(path_s):
    def parse(path):
        g = gzip.open(path, 'rb')
        for l in g:
            yield eval(l)

    def get_raw_df(path):
        df = {}
        for i, d in tqdm(enumerate(parse(path)), ascii=True):
            if "" not in d.values():
                df[i] = d
        df = pd.DataFrame.from_dict(df, orient='index')
        # df = df[["reviewerID", "asin", "reviewText", "overall"]]
        return df

    csv_path_s = path_s.replace('.json.gz', '.csv')

    if os.path.exists(csv_path_s):
        df_s = pd.read_csv(csv_path_s, usecols=[0, 1, 3])
        print('Load raw data from %s.' % csv_path_s)
    else:
        df_s = get_raw_df(path_s)

        df_s.to_csv(csv_path_s, index=False)
        print('Build raw data to %s.' % csv_path_s)

    return df_s

def filterout(df_s):
    index_s = df_s[["overall", "asin"]].groupby('asin').count() >= 20
    item_s = set(index_s[index_s['overall'] == True].index)
    df_s = df_s[df_s['asin'].isin(item_s)]
    index_s = df_s[["overall", "reviewerID"]].groupby('reviewerID').count() >= 20
    user_s = set(index_s[index_s['overall'] == True].index)
    df_s = df_s[df_s['reviewerID'].isin(user_s)]
    index_s = df_s[["overall", "asin"]].groupby('asin').count() >= 20
    item_s = set(index_s[index_s['overall'] == True].index)
    df_s = df_s[df_s['asin'].isin(item_s)]
    index_s = df_s[["overall", "reviewerID"]].groupby('reviewerID').count() >= 20
    user_s = set(index_s[index_s['overall'] == True].index)
    df_s = df_s[df_s['reviewerID'].isin(user_s)]
    return df_s


path_s = 'reviews_Books_5.csv'
df_s = get_df(path_s)
print(df_s[["overall", "reviewerID"]].groupby('reviewerID').count())

# #筛选大于20的,s是cd,t是其他的
df_s = filterout(df_s)

df_s = df_s[['reviewerID','asin','overall']]

df_s.to_csv('./books/books_all_data.csv',index = False)
