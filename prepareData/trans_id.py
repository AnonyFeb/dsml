# from utils import date, evaluate_mse, MFDataset, evaluate_top_n, evaluate_precision, split_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.python.ops import data_flow_ops 

data_path= './books/trans_books_all_data.csv'




df = pd.read_csv(data_path, usecols=[0, 1, 2])
df.columns=['user_id','item_id','ratings']
print(df.user_id)
print(df.head())
ratings = df['ratings']

user_id = df['user_id']
# user_id = user_id.apply(lambda x:x - 1)
# user_id.colums=['user_id']

item_id = df['item_id']
# item_id = item_id.apply(lambda x:x - 1)
# item_id.colums=['item_id']

ratings = df['ratings']
ratings = ratings.apply(lambda x:1.0)


result = pd.concat([user_id,item_id,ratings],axis=1)
result.columns=['user_id','item_id','ratings']
print(result.head())
result.to_csv('./books/trans_Implicit_books_all_data.csv',sep=',',header=True,index=False)
# print(result.head())