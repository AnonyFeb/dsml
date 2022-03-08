import pandas as pd
import random
from tqdm import tqdm
import csv
from sklearn.model_selection import train_test_split

dataset_name = 'cds'



df = pd.read_csv('./'+dataset_name+'/trans_Implicit_'+dataset_name+'_all_data.csv', usecols=[0, 1, 2])
df.columns = ['userID', 'itemID', 'ratings']  # Rename above columns for convenience

item_count = df['itemID'].value_counts().count()  

all_item = set(df.itemID)#select from all data
all_user = set(df.userID)

for user_id in tqdm(range(len(all_user))):

    reviewer = df[df['userID'].isin([user_id])]
    pos_item_list = set(reviewer.itemID)

    test_pos_item = random.sample(pos_item_list,1)#per user sample one pos item as test item.
    
    train_pos_item_list = pos_item_list - set(test_pos_item)#all train pos_item

    sample_neg_train_item_list = all_item - train_pos_item_list - test_pos_item#all train neg item
    
    train_pos_item_list = list(train_pos_item_list)

    if 5 * len(pos_item_list) > len(sample_neg_train_item_list): #too many pos_item
        sample_neg_train_index = random.sample(sample_neg_train_item_list,len(pos_item_list))
    else:#select 5 neg items
        sample_neg_train_index = random.sample(sample_neg_train_item_list,4 * len(pos_item_list)) #sample 5 negtive item per pos_item

    sample_neg_item_test = all_item - pos_item_list - set(sample_neg_train_index) - set(test_pos_item)#all test neg item
    
    sample_neg_test_index = random.sample(sample_neg_item_test, 100)

    neg_index = 0
    with open('./'+dataset_name+'/angluarloss/trans_'+dataset_name+'_data_trainmf_WithOneNegData_new.csv','a') as csvfile_pos_train:
        writer = csv.writer(csvfile_pos_train)

        for pos_item in train_pos_item_list:
            writer.writerow([int(user_id),int(pos_item),1.0]) #pos item
                    
                    
        for negg in sample_neg_train_index:#neg_item
            writer.writerow([int(user_id),int(negg),0.0])
    
    #only one positive item
    with open('./'+dataset_name+'/angluarloss/trans_'+dataset_name+'_data_testmf_WithOneNegData.csv','a') as csvfile_pos_test:
            writer = csv.writer(csvfile_pos_test)
            writer.writerow([int(user_id),int(test_pos_item[0]),1.0])
    #100 neg item test
    for neg_item_test in sample_neg_test_index:
        with open('./'+dataset_name+'/angluarloss/trans_'+dataset_name+'_data_testmf_WithOneNegData.csv','a') as csvfile_neg_test:
                writer = csv.writer(csvfile_neg_test)
                writer.writerow([int(user_id),int(neg_item_test),0.0])
