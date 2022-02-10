from ast import Lambda
from re import X
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import random
from math import gamma, log2
import numpy as np
import cvxpy as cp


def check_model_ndcg(combine_U,combine_V,valid_data):
    pred_ratings_ns = np.zeros([len(combine_U),len(combine_V)])
    ndcg_topk_mf = [0] * 14
    hr_topk = [0] * 14
    top_k_list = [1,2,3,4,5,6,7,8,9,10,20,30,40,50]

    for user_id in tqdm(range(len(combine_U))):
        reviewer = valid_data[valid_data['userID'].isin([user_id])]
        pos_item_list = list(set(reviewer[reviewer['rating']==1.0].itemID))
        neg_item_list = list(set(reviewer[reviewer['rating']==0.0].itemID))

        for neg_itemID in neg_item_list:
            ratings = np.sum(combine_U[user_id] * combine_V[neg_itemID])
            pred_ratings_ns[user_id,neg_itemID] = ratings

        for pos_itemID in pos_item_list:
            predict_pos_ratings = np.sum(combine_U[user_id] * combine_V[pos_itemID])
            pred_ratings_ns[user_id,pos_itemID] =  predict_pos_ratings
        ndcg_mf_user = []
        hr_user = []
        predict_mf_score = pred_ratings_ns[user_id,:]
        order_mf = list(np.argsort(predict_mf_score))
        order_mf.reverse()
        order_mf = order_mf

        for k in top_k_list:
            top_k = k
            dcg_ns = get_ndcg(order_mf, top_k, pos_item_list)
            hr = get_hr(order_mf, top_k,pos_item_list)
            ndcg_mf_user.append(dcg_ns)
            hr_user.append(hr)
        ndcg_topk_mf = list(map(lambda x :x[0]+x[1] ,zip(ndcg_topk_mf,ndcg_mf_user)))
        hr_topk = list(map(lambda x :x[0]+x[1] ,zip(hr_topk,hr_user)))

    results_ndcg,results_hr = [],[]

    for j in range(len(ndcg_topk_mf)):
        results_ndcg.append(ndcg_topk_mf[j] / len(combine_U))
        results_hr.append(hr_topk[j] / len(combine_U))
    return results_ndcg,results_hr

def get_ndcg(order_mf, top_k, pos_item_list):
    nDCG = 0
    for i in range(top_k):
        if order_mf[i] in pos_item_list:
            nDCG = 1 / log2(i + 2)
    return nDCG

def get_hr(order_mf, top_k, pos_item_list):
    hr = 0
    for i in range(top_k):
        if order_mf[i] in pos_item_list:
            hr = 1 
    return hr


def update_U_ori(pos_item_list,neg_item_list,user_id,U,V):
    n=20
    #j is pos, k is neg
    x = 2*(cp.Variable(n, boolean=True) - 0.5)
    sum_ci = np.zeros((20,20))
    for j in pos_item_list:
        vj = V[j].reshape(20,1)#vj 转化成列向量
        vjvjt = np.matmul(vj,vj.T)#Vj x Vj.T
        trian_neg_list = random.sample(neg_item_list,5)
        # trian_neg_list = [3428, 1177, 962, 3095, 2131]

        for k in trian_neg_list:
            PI_ijk = compute_pi(np.sum(U[user_id] * V[k] - U[user_id] * V[j]))#rik - rij
            vk = V[k].reshape(20,1)#vk 转化成列向量
            vkvkt = np.matmul(vk,vk.T)#(20,1) x (1,20) = 20x20
            vjvkt = np.matmul(vj,vk.T)#(20,1) x (1,20) = 20x20
            sum_jk = vjvjt *PI_ijk + vkvkt * PI_ijk - 2 *  PI_ijk * vjvkt
            sum_ci += sum_jk
    Ci = (sum_ci + sum_ci.T) / 2
    Di = np.array([20.0]*n)
    # print(trian_neg_list)
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, Ci)+ Di.T@x))
    prob.solve(solver=cp.XPRESS, verbose=False,warm_start=True)
    U[user_id] = x.value
    return U


def update_U(df_train,U,V, user_id):
    n=20
    d=20
    gamma = 1
    user_count = df_train['userID'].value_counts().count()
    Lamda = 1
    #j is pos, k is neg
    x = 2*(cp.Variable(n, boolean=True) - 0.5)
    sum_cxi = np.zeros((20,20))
    sum_cyi = np.zeros((20,20))
    sum_Dyi = np.zeros((n,1))
    sum_Dxi = np.zeros((n,1))
    for user_id in tqdm(range(user_count)):
        reviewer = df[df['userID'].isin([user_id])]
        #all pos
        pos_item_list = set(reviewer[reviewer['rating']==1.0].itemID)
        #all neg
        neg_item_list = set(reviewer[reviewer['rating']==0.0].itemID)
        for j in pos_item_list:
            vj = V[j].reshape(20,1)#vj 转化成列向量
            vjvjt = np.matmul(vj,vj.T)#Vj x Vj.T
            trian_neg_list = random.sample(neg_item_list,5)
            # trian_neg_list = [3428, 1177, 962, 3095, 2131]

            for k in trian_neg_list:
                #vk 转化成列向量
                vk = V[k].reshape(20,1)
                PI_fi_ijk = compute_pi(np.sum(U[user_id] * V[k] - U[user_id] * V[j])/(2*d))#rik - rij
                PI_eta_ijk = compute_pi(np.sum(4 * gamma*gamma*(U[user_id] * V[k] + V[j] * V[k]) - 2*(1 + gamma * gamma)*U[user_id] * V[j] ))#4·γ2 (ui·vk + vj·vk ) − 2(1 + γ2 )ui·vj
                
                
                vkvkt = np.matmul(vk,vk.T)#(20,1) x (1,20) = 20x20
                vjvkt = np.matmul(vj,vk.T)#(20,1) x (1,20) = 20x20

                vkvjt = np.matmul(vk,vj.T)#(20,1) x (1,20) = 20x20
                
                cx_i= vjvjt * PI_fi_ijk + vkvkt * PI_fi_ijk - 2 *  PI_fi_ijk * vkvjt
                cy_i =  pow((1 + gamma * gamma),2) * vjvjt * PI_eta_ijk + vkvkt * PI_eta_ijk -4*(1 + gamma*gamma)*vkvjt * PI_eta_ijk
                #sum Cyi and Cxi
                sum_cxi += cx_i
                sum_cyi += cy_i
                #compute dyi
                d_xi = d * (vk - vj)
                d_yi = 32 * pow(gamma,4) * np.matmul(vkvkt , vj) * PI_eta_ijk - 64 *pow(gamma,2)* (1+pow(gamma,2)) * np.matmul(vjvkt, vj) + 2 * gamma * gamma * vk - (1+pow(gamma,2)) * vj 
                sum_Dyi += d_yi
                sum_Dxi += d_xi
        Ci = sum_cxi + 4 * Lamda *pow(gamma,4) * sum_cyi
        # Cxi = (sum_cxi + sum_cxi.T) / 2
        # Cyi = (sum_cyi + sum_cyi.T) / 2
        Di = sum_Dxi + 4 * Lamda * sum_Dyi #np.array([20.0]*n)
        Ci = (Ci + Ci.T) / 2
        # print(trian_neg_list)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, Ci)+ Di.T@x))
        prob.solve(solver=cp.CPLEX, verbose=False,warm_start=True)
        U[user_id] = x.value
    return U




def update_V(df_train,U,V):
    n=20
    d=20
    item_count = df_train['itemID'].value_counts().count()
    gamma = 1
    Lamda = 1
    #j is pos, k is neg
    x = 2*(cp.Variable(n, boolean=True) - 0.5)

    for item_id in tqdm(range(item_count)):
        reviewer = df_train[df_train['itemID'].isin([item_id])]
        #item_id 也就是j
        pos_user_list = set(reviewer[reviewer['rating']==1.0].userID)# 对该item有正反馈的所有user，也就是i
        sum_cxj = np.zeros((20,20))
        sum_cyj = np.zeros((20,20))
        sum_Dyj = np.zeros((n,1))
        sum_Dxj = np.zeros((n,1))
        for pos_user_id in pos_user_list:
            #pos_user_id 也就是i
            # print(pos_user_id)
            reviewer = df_train[df_train['userID'].isin([pos_user_id])]
            #pos user下的所有neg item 
            neg_item_in_pos_user_list = random.sample(set(reviewer[reviewer['rating']==0.0].itemID),5)#pos 用户对应的neg item也就是k

            ui = U[pos_user_id].reshape(20,1)#ui转化成列向量
            uiuit = np.matmul(ui,ui.T)
            for neg_item_id in neg_item_in_pos_user_list:
                #neg_item_id 也就是k
                vk = V[neg_item_id].reshape(20,1)
                PI_fi_ijk = compute_pi(np.sum(U[pos_user_id] * V[neg_item_id] - U[pos_user_id] * V[item_id])/(2*d))
                PI_eta_ijk = compute_pi(np.sum(4 * gamma*gamma*(U[pos_user_id] * V[neg_item_id] + V[item_id] * V[neg_item_id]) \
                             - 2*(1 + gamma * gamma)*U[pos_user_id] * V[item_id] ))#4·γ2 (ui·vk + vj·vk ) − 2(1 + γ2 )ui·vj
                #vk 转化成列向量
                vkvkt = np.matmul(vk,vk.T)#(20,1) x (1,20) = 20x20

                vkuit = np.matmul(vk,ui.T)
                uitvk = np.matmul(ui,vk.T)#(20,1) x (1,20) = 20x20

                c_xj = uiuit * PI_fi_ijk
                c_yj = 16 * pow(gamma,4) * vkvkt * PI_eta_ijk + 4 * pow((1 + pow(gamma,2)),2) * uiuit * PI_eta_ijk - 16*pow(gamma,2) * (1 + pow(gamma,2)) * vkuit * PI_eta_ijk
                
                sum_cxj += c_xj
                sum_cyj += c_yj
                #compute dxj dyj
                d_xj = -2 * np.matmul(uitvk,ui) * PI_fi_ijk - d * ui
                d_yj = 32 * pow(gamma,4) * np.matmul(uitvk,vk) * PI_eta_ijk - 16 * pow(gamma,2) * (1+ pow(gamma,2)) * np.matmul(uitvk,ui) * PI_eta_ijk \
                    + 2 * pow(gamma,2) * vk - (1+ pow(gamma,2)) * ui
                sum_Dyj += d_yj
                sum_Dxj += d_xj
        try:
            Cj = sum_cxj + Lamda * sum_cyj
            # Cxi = (sum_cxi + sum_cxi.T) / 2
            # Cyi = (sum_cyi + sum_cyi.T) / 2
            
            Dj = sum_Dxj + Lamda * sum_Dyj #np.array([20.0]*n)
            Cj = (Cj + Cj.T) / 2
            # print(trian_neg_list)
            prob = cp.Problem(cp.Minimize(cp.quad_form(x, Cj)+ Dj.T@x))
            prob.solve(solver=cp.CPLEX, verbose=False,warm_start=True)
            V[item_id] = x.value
        except Exception:
            continue
    return V



def compute_sigmoid(x):
    return 1/(1 + np.exp(-x))

def compute_pi(x):
    return (1/((2 * x)+0.001)) * (compute_sigmoid(x) - 0.5)


data_path = './trans_movies_data_trainmf_WithOneNegData.csv'
df = pd.read_csv(data_path, usecols=[0, 1, 2])
df.columns = ['userID', 'itemID','rating']  # Rename above columns for convenience

data_path_test = './trans_movies_data_testmf_WithOneNegData.csv'
df_test = pd.read_csv(data_path_test, usecols=[0, 1, 2])
df_test.columns = ['userID', 'itemID','rating']


#get user/item num
df['userID'] = df.groupby(df['userID']).ngroup()
df['itemID'] = df.groupby(df['itemID']).ngroup()
user_count = df['userID'].value_counts().count()  #count user_num
item_count = df['itemID'].value_counts().count()  #count item_num
print("Dataset contains ,",{df.shape[0]} ," records," ,{user_count} ," users and" ,{item_count} ,"items.")

#load init_feature
u_feature_file = open('./result/movielens/movielens_user_3439_feature.data', 'rb')
u_feature = pickle.load(u_feature_file)
u_feature_hash = u_feature.copy()
u_feature_hash[u_feature_hash>0] = 1
u_feature_hash[u_feature_hash<=0] = -1

v_feature_file = open('./result/movielens/movielens_item_3439_feature.data', 'rb')
v_feature = pickle.load(v_feature_file)
v_feature_hash = v_feature.copy()
v_feature_hash[v_feature_hash>0] = 1
v_feature_hash[v_feature_hash<=0] = -1
#train

#v_feature_file = open('./result/cds/cds_item_1801999_feature.data', 'rb')
#v_feature_file = open('./result/movies/top10_result/movies_item_9899_feature.data', 'rb')

#open('./result/movielens/movielens_item_3439_feature.data', 'rb')
user_id = 333#随机选一个

reviewer = df[df['userID'].isin([user_id])]
#all pos
pos_item_list = set(reviewer[reviewer['rating']==1.0].itemID)
#all neg
neg_item_list = set(reviewer[reviewer['rating']==0.0].itemID)

# results,results_hr = check_model_ndcg(u_feature,v_feature,valid_data=df_test)
# print(results)
# print(results_hr)
# for _ in tqdm(range(20)):
#     results,results_hr = check_model_ndcg(u_feature_hash,v_feature_hash,valid_data=df_test)
#     print(results)
#     print(results_hr)
u_feature_hash = update_U(df,u_feature_hash,v_feature_hash, user_id)
    # v_feature_hash = update_V(df,u_feature_hash,v_feature_hash)