#this code is used to transform the stringID to int ID,with dict.

import json
import pandas as pd

filename = './books/books_all_data.csv'
with open('./books_itemID.json','r')as p:
    json_item = json.load(p)
with open('./books_userID.json','r')as fp:
    json_user = json.load(fp)
count = 0
score = []
with open(filename, 'r') as f:
    list_data = []
    for line in f:
        if count ==0:
            count = 1
            continue

        words_user = line.strip().split(',')[0]
        words_item = line.strip().split(',')[1]
        words_score = line.strip().split(',')[2]
        score.append(words_score)
        # if words_item not in json_item: #check if all item contain
        #     count = count + 1
        user_id = json_user[words_user] 
        # print(user_id)
        user_item = json_item[words_item]
        data = [user_id,user_item,words_score]
        list_data.append(data)
    
test = pd.DataFrame(data=list_data)
test.to_csv('./books/trans_books_all_data.csv',index=False)