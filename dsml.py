import tensorflow as tf
import pandas as pd
import pickle
import random
from math import log2, ceil, gamma
import os
import numpy as np
import cvxpy as cp
import warnings


class angluarLoss:
    def __init__(self, sess, user_count, item_count, k, learning_rate=0.001, batch_size=512, reg_lambda=0.0001):
        self.sess = sess
        self.n, self.m = user_count, item_count
        self.k = k
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.reg_lambda = tf.constant(reg_lambda, dtype=tf.float32)
        self.gamma = 1
        self.loss_lambda = 1
        self.max_epoch = 70
        self.build_graph()

    def build_graph(self):
        self.neg_item_idx = tf.placeholder(tf.int32, [None, 5], name='neg_idx')
        self.pos_item_idx = tf.placeholder(tf.int32, [None, 1], name='pos_idx')
        self.item_feature = tf.placeholder(tf.float32, [None, ])
        self.u_idx = tf.placeholder(tf.int32, [None, 1], name='u_idx')
        self.U = self.weight_variable([self.n, self.k], 'U')
        self.V = self.weight_variable([self.m, self.k], 'V')
        self.U_embed = tf.nn.embedding_lookup(self.U, self.u_idx)
        self.V_neg_embed = tf.nn.embedding_lookup(self.V, self.neg_item_idx)  # 1 * 20
        self.V_pos_embed = tf.nn.embedding_lookup(self.V, self.pos_item_idx)  # m * 20
        self.sq_tan_alpha = self.gamma ** 2
        self.sq_tan_alpha = tf.cast(self.sq_tan_alpha, tf.float32)

        self.XaTXp = tf.reduce_sum(tf.multiply(self.U_embed, self.V_pos_embed), reduction_indices=2)
        self.XaAddXpPXn = tf.reduce_sum(tf.multiply(tf.add(self.U_embed, self.V_pos_embed), self.V_neg_embed),
                                        reduction_indices=2)  # n * 1
        self.four_sq_tan_Xa_Xp_Xn = tf.multiply(4 * self.sq_tan_alpha, self.XaAddXpPXn)
        self.two_sq_tan_Xa_Xp = tf.multiply(2 * (1 + self.sq_tan_alpha), self.XaTXp)
        self.four_tow = tf.reduce_sum(tf.square(2 + self.four_sq_tan_Xa_Xp_Xn - self.two_sq_tan_Xa_Xp),
                                      reduction_indices=1)
        self.loss_angle = tf.reduce_mean(self.four_tow)

        self.XaTXn = tf.reduce_sum(tf.multiply(self.U_embed, self.V_neg_embed), reduction_indices=2)  # n * 1
        self.t = self.XaTXn - self.XaTXp
        self.sum_t = tf.reduce_sum(tf.square(1 + self.t), reduction_indices=1)
        self.npair_loss = tf.reduce_mean(self.sum_t)

        gloabl_steps = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.learning_rate
                                                        , gloabl_steps,
                                                        30000,
                                                        0.99,
                                                        staircase=True)

        self.total_loss = self.loss_lambda * self.loss_angle + self.npair_loss
        self.reg = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U)),
                          tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V)))
        self.reg_loss = tf.add(self.total_loss, self.reg)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_step_u = self.optimizer.minimize(self.reg_loss, global_step=gloabl_steps)
        self.train_step_v = self.optimizer.minimize(self.reg_loss, global_step=gloabl_steps)
        tf.summary.scalar("Reg-Loss", self.reg_loss)
        # add op for merging summary
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, mean=0.0, stddev=0.001)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        b_init = tf.constant_initializer(0.)
        return tf.get_variable(name, shape, initializer=b_init)

    def construct_feeddict(self, u_idx, pos_idx, neg_idx):
        return {self.u_idx: u_idx, self.pos_item_idx: pos_idx, self.neg_item_idx: neg_idx}

    def train_test_validation(self, df_train, df_ratings):
        df_train['userID'] = df_train.groupby(df_train['userID']).ngroup()
        df_train['itemID'] = df_train.groupby(df_train['itemID']).ngroup()
        user_count = df_train['userID'].value_counts().count()  #
        num_batch = ceil(user_count / self.batch_size)
        self.sess.run(tf.global_variables_initializer())
        best_ndcg = 0.1
        set_user_list = list(range(user_count))
        print('start training ...')
        epoch = 0

        while (epoch < self.max_epoch):
            epoch += 1
            for iteration in range(num_batch):
                if len(set_user_list) > self.batch_size:
                    user_idx_list = random.sample(set_user_list, self.batch_size)
                else:  # sample all set_user
                    user_idx_list = set_user_list
                    set_user_list = list(range(user_count))
                user_idx_list = set_user_list
                batch_pos_idx = []
                batch_neg_idx = []
                batch_user_idx = []
                for user_idx in user_idx_list:
                    user_idx = int(user_idx)
                    reviewer = df_train[df_train['userID'].isin([user_idx])]
                    pos_item_list = set(reviewer[reviewer['rating'] == 1.0].itemID)
                    neg_item_list = set(reviewer[reviewer['rating'] == 0.0].itemID)
                    pos_idx = list(random.sample(pos_item_list, 1))  # positive
                    neg_idx = list(random.sample(neg_item_list, 5))  # 5 negtive
                    batch_pos_idx.append(pos_idx)
                    batch_neg_idx.append(neg_idx)
                    batch_user_idx.append([user_idx])

                batch_user_idx = np.array(batch_user_idx)
                batch_pos_idx = np.array(batch_pos_idx)
                batch_neg_idx = np.array(batch_neg_idx)
                feed_dict = self.construct_feeddict(batch_user_idx, batch_pos_idx, batch_neg_idx)
                _, _, total_loss, summary_str, feature_U, feature_V, loss_angle, npair_loss = self.sess.run(
                    [self.train_step_u, self.train_step_v, self.total_loss, self.summary_op, self.U, self.V,
                     self.loss_angle, self.npair_loss], feed_dict=feed_dict)

            results, hr_results = check_model_ndcg(feature_U, feature_V, df_ratings)

            perf_str = 'Epoch %d: pair_loss = %.5f, scale_loss = %.5f, total_loss = %.5f, ndcg@1-10=[%s]' % \
                       (
                       epoch, npair_loss, loss_angle, npair_loss + loss_angle, ', '.join(['%.5f' % r for r in results]))
            print(perf_str)
            if results[-1] > best_ndcg:
                best_ndcg = results[-1]
                best_feature_u = feature_U
                best_feature_v = feature_V
                save_path = './result/Movielens/movielens_user_feature_best.data'
                with open(save_path, 'wb') as fw:
                    pickle.dump(feature_U, fw)

                save_path = './result/Movielens/movielens_item_feature_best.data'
                with open(save_path, 'wb') as fw:
                    pickle.dump(feature_V, fw)

        print('start testing ...')
        results1, hr_results = check_model_ndcg(best_feature_u, best_feature_v, df_ratings)
        ndcg_all = [round(k, 4) for k in results1]
        ndcg = ', '.join(str(kk) for kk in ndcg_all)
        print('Ours-real: NDCG@1~10 = ', ndcg)
        hr_all = [round(k, 4) for k in hr_results]
        hr = ', '.join(str(kk) for kk in hr_all)
        print('Ours-real: HR@1~10 = ', hr)
        return best_feature_u, best_feature_v


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cosSim(x, y):
    tmp = np.sum(x * y)
    non = np.linalg.norm(x) * np.linalg.norm(y)
    return np.round(tmp / float(non), 9)


def compute_sigmoid(x):
    warnings.filterwarnings('ignore')
    return 1 / (1 + np.exp(-x))


def compute_pi(x):
    return (1 / ((2 * x) + 0.001)) * (compute_sigmoid(x) - 0.5)


def update_UV(df_train, U, V):
    n = 20
    d = 20
    gamma = 1
    user_count = df_train['userID'].value_counts().count()
    Lamda = 1
    # j is pos, k is neg

    for user_id in range(user_count):
        reviewer = df_train[df_train['userID'].isin([user_id])]
        pos_item_list = set(reviewer[reviewer['rating'] == 1.0].itemID)
        neg_item_list = set(reviewer[reviewer['rating'] == 0.0].itemID)
        sum_cxi = np.zeros((20, 20))
        sum_cyi = np.zeros((20, 20))
        sum_Dyi = np.zeros((n, 1))
        sum_Dxi = np.zeros((n, 1))
        for j in pos_item_list:
            vj = V[j].reshape(20, 1)
            vjvjt = np.matmul(vj, vj.T)  # Vj x Vj.T
            trian_neg_list = random.sample(neg_item_list, 5)
            for k in trian_neg_list:
                vk = V[k].reshape(20, 1)
                fi_ijk = np.sum(U[user_id] * V[k] - U[user_id] * V[j]) / (2 * d)
                PI_fi_ijk = compute_pi(-fi_ijk)  # rik - rij
                PI_eta_ijk = compute_pi(-np.sum(
                    2 * gamma * gamma * (U[user_id] * V[k] + V[j] * V[k]) - (1 + gamma * gamma) * U[user_id] * V[
                        j]))  # 4·γ2 (ui·vk + vj·vk ) − 2(1 + γ2 )ui·vj
                vkvkt = np.matmul(vk, vk.T)  # (20,1) x (1,20) = 20x20
                vjvkt = np.matmul(vj, vk.T)  # (20,1) x (1,20) = 20x20
                vkvjt = np.matmul(vk, vj.T)  # (20,1) x (1,20) = 20x20
                # vjtvk = np.matmul(vk, vj.T)  # (20,1) x (1,20) = 20x20
                cx_i = vjvjt * PI_fi_ijk + vkvkt * PI_fi_ijk - 2 * PI_fi_ijk * vkvjt
                cy_i = pow((1 + gamma * gamma), 2) * vjvjt * PI_eta_ijk + 4 * pow(gamma, 4) * vkvkt * PI_eta_ijk - 4 * gamma * gamma * (
                            1 + gamma * gamma) * vkvjt * PI_eta_ijk
                # sum Cyi and Cxi
                sum_cxi += cx_i
                sum_cyi += cy_i
                # compute dyi
                d_xi = d * (vk - vj)
                d_yi = 4 * pow(gamma, 4) * vj.T@vk * vk * PI_eta_ijk - 8 * pow(gamma, 4) * (
                            1 + pow(gamma, 2)) * vj.T@vk * vj + gamma * gamma * vk - 0.5 * (
                                   1 + pow(gamma, 2)) * vj
                sum_Dyi += d_yi
                sum_Dxi += d_xi

        Ci = (sum_cxi + 4 * Lamda * d * d * sum_cyi) / (len(pos_item_list) * user_count * 5)
        # Cxi = (sum_cxi + sum_cxi.T) / 2
        # Cyi = (sum_cyi + sum_cyi.T) / 2
        Di = (sum_Dxi + 4 * Lamda * d * d * sum_Dyi) / (len(pos_item_list) * user_count * 5) # np.array([20.0]*n)
        Ci = (Ci + Ci.T) / 2
        Di = Di / 2
        U[user_id] = bqp(Ci, Di, 200)
        # x = cp.Variable(n, boolean=True)
        # prob = cp.Problem(cp.Minimize(cp.quad_form(x, Ci) + Di.T @ x))
        # prob.solve(solver=cp.XPRESS, verbose=False, warm_start=True)
        # x = 2 * (x - 0.5)
        # U[user_id] = np.trunc(x.value).astype(int)
    return U


def update_V(df_train, U, V):
    n = 20
    d = 20
    item_count = df_train['itemID'].value_counts().count()
    gamma = 1
    Lamda = 1
    # j is pos, k is neg
    # x = 2 * (cp.Variable(n, boolean=True) - 0.5)

    for item_id in range(item_count):
        reviewer = df_train[df_train['itemID'].isin([item_id])]
        pos_user_list = set(reviewer[reviewer['rating'] == 1.0].userID)
        sum_cxj = np.zeros((20, 20))
        sum_cyj = np.zeros((20, 20))
        sum_Dyj = np.zeros((n, 1))
        sum_Dxj = np.zeros((n, 1))
        for pos_user_id in pos_user_list:
            reviewer = df_train[df_train['userID'].isin([pos_user_id])]
            neg_item_in_pos_user_list = random.sample(set(reviewer[reviewer['rating'] == 0.0].itemID), 5)  #
            ui = U[pos_user_id].reshape(20, 1)
            uiuit = np.matmul(ui, ui.T)
            for neg_item_id in neg_item_in_pos_user_list:
                vk = V[neg_item_id].reshape(20, 1)
                PI_fi_ijk = compute_pi(-np.sum(U[pos_user_id] * V[neg_item_id] - U[pos_user_id] * V[item_id]) / (2 * d))
                PI_eta_ijk = compute_pi(
                    -np.sum(2 * gamma * gamma * (U[pos_user_id] * V[neg_item_id] + V[item_id] * V[neg_item_id]) \
                           - (1 + gamma * gamma) * U[pos_user_id] * V[
                               item_id]))  # 4·γ2 (ui·vk + vj·vk ) − 2(1 + γ2 )ui·vj

                vkvkt = np.matmul(vk, vk.T)  # (20,1) x (1,20) = 20x20
                vkuit = np.matmul(vk, ui.T)
                uitvk = np.matmul(ui, vk.T)  # (20,1) x (1,20) = 20x20
                c_xj = uiuit
                c_yj = 4 * pow(gamma, 4) * vkvkt * PI_eta_ijk + pow((1 + pow(gamma, 2)), 2) * uiuit * PI_eta_ijk - 4 * pow(gamma,2) * (1 + pow(gamma, 2)) * vkuit * PI_eta_ijk

                sum_cxj += c_xj
                sum_cyj += c_yj
                # compute dxj dyj
                d_xj = -2 * ui.T@vk * ui - 4 * d * ui
                d_yj = 8 * pow(gamma, 4) * ui.T@vk * vk * PI_eta_ijk - 4 * pow(gamma, 2) * (
                            1 + pow(gamma, 2)) * ui.T@vk * ui * PI_eta_ijk \
                       + pow(gamma, 2) * vk - 0.5 * (1 + pow(gamma, 2)) * ui
                sum_Dyj += d_yj
                sum_Dxj += d_xj
    # try:
        Cj = (sum_cxj + 4 * Lamda * d * d * sum_cyj) / (len(pos_user_list) * item_count * 5)
        Dj = (sum_Dxj + 4 * Lamda * d * d * sum_Dyj) / (len(pos_user_list) * item_count * 5)  # np.array([20.0]*n)
        Cj = (Cj + Cj.T) / 2
        Dj = Dj / 2
        V[item_id] = bqp(Cj, Dj, 1000)
        # # print(trian_neg_list)
        # x = cp.Variable(n, boolean=True)
        # prob = cp.Problem(cp.Minimize(cp.quad_form(x, Cj) + Dj.T @ x))
        # prob.solve(solver=cp.XPRESS, verbose=False, warm_start=True)
        # x = 2 * (x - 0.5)
        # V[item_id] = np.trunc(x.value).astype(int)
    # except Exception:
    #     continue
    return V


def bqp(A, b, l):
    k = A.shape[0]
    C =np.r_[np.c_[A, b], np.r_[b, np.zeros([1,1])].T]
    X = cp.Variable((k+1, k+1), PSD=True)
    X = cp.atoms.affine.wraps.psd_wrap(X)
    constraints = [cp.diag(X) == 1]
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), constraints)
    prob.solve(solver=cp.CVXOPT, verbose=False, warm_start=True)
    cov = X.value
    cov = (cov + cov.T) / 2
    S = np.sign(np.random.multivariate_normal(np.zeros(k+1), cov, l).T)  # k+1 * L
    loss = np.sum((C @ S) * S, axis=0)
    min_index = loss.tolist().index(min(loss.tolist()))
    xx = S.T[min_index]
    tag = xx[-1]
    x = xx[:k] * tag
    return x


def train_loss(df_train, U, V):
    gamma = 1
    lamda = 1
    d, user_count = U.shape
    # j is pos, k is neg
    total_loss = np.zeros(1)
    for user_id in range(user_count):
        reviewer = df_train[df_train['userID'].isin([user_id])]
        pos_item_list = set(reviewer[reviewer['rating'] == 1.0].itemID)
        neg_item_list = set(reviewer[reviewer['rating'] == 0.0].itemID)
        # sum_loss = np.zeros(1)
        for j in pos_item_list:
            trian_neg_list = random.sample(neg_item_list, 5)
            # sum_l = np.zeros(1)
            for k in trian_neg_list:
                x_ijk = np.sum(U[user_id] * V[k] - U[user_id] * V[j]) / (2 * d)
                y_ijk = np.sum(2 * gamma * gamma * (U[user_id] * V[k] + V[j] * V[k]) - (1 + gamma * gamma) * U[user_id] * V[j])
                # lu = loss(x_ijk, fi_ijk)
                l_ijk = log2(1+np.exp(x_ijk)) + lamda * log2(1+np.exp(y_ijk))
                total_loss += l_ijk
            # sum_loss += sum_l
        # total_loss += sum_loss
    return total_loss


def check_model_ndcg(combine_U, combine_V, valid_data):
    pred_ratings_ns = np.zeros([len(combine_U), len(combine_V)])
    ndcg_topk_mf = [0] * 10
    hr_topk = [0] * 10
    top_k_list = [i for i in range(1, 11)]
    for user_id in range(len(combine_U)):
        reviewer = valid_data[valid_data['userID'].isin([user_id])]

        pos_item_list = list(set(reviewer[reviewer['rating'] == 1.0].itemID))
        neg_item_list = list(set(reviewer[reviewer['rating'] == 0.0].itemID))

        for neg_itemID in neg_item_list:
            ratings = np.sum(combine_U[user_id] * combine_V[neg_itemID])
            pred_ratings_ns[user_id, neg_itemID] = ratings

        for pos_itemID in pos_item_list:
            predict_pos_ratings = np.sum(combine_U[user_id] * combine_V[pos_itemID])
            pred_ratings_ns[user_id, pos_itemID] = predict_pos_ratings
        ndcg_mf_user = []
        hr_user = []
        predict_mf_score = pred_ratings_ns[user_id, :]
        order_mf = list(np.argsort(predict_mf_score))
        order_mf.reverse()
        order_mf = order_mf

        for k in top_k_list:
            top_k = k
            dcg_ns = get_ndcg(order_mf, top_k, pos_item_list)
            hr = get_hr(order_mf, top_k, pos_item_list)
            ndcg_mf_user.append(dcg_ns)
            hr_user.append(hr)
        ndcg_topk_mf = list(map(lambda x: x[0] + x[1], zip(ndcg_topk_mf, ndcg_mf_user)))
        hr_topk = list(map(lambda x: x[0] + x[1], zip(hr_topk, hr_user)))

    results_ndcg, results_hr = [], []

    for j in range(len(ndcg_topk_mf)):
        results_ndcg.append(ndcg_topk_mf[j] / len(combine_U))
        results_hr.append(hr_topk[j] / len(combine_U))
    return results_ndcg, results_hr


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


if __name__ == "__main__":
    data_path = './Data/Movielens/trans_movielens_data_trainmf_WithOneNegData.csv'
    df = pd.read_csv(data_path, usecols=[0, 1, 2])
    df.columns = ['userID', 'itemID', 'rating']  # Rename above columns for convenience
    data_path_test = './Data/Movielens/trans_movielens_data_testmf_WithOneNegData.csv'
    df_ratings = pd.read_csv(data_path_test, usecols=[0, 1, 2])
    df_ratings.columns = ['userID', 'itemID', 'rating']

    user_count = df['userID'].value_counts().count()  #
    item_count = df['itemID'].value_counts().count()  #
    print("Dataset contains ,", {df.shape[0]}, " records,", {user_count}, " users and", {item_count}, "items.")
    test_item = []
    test_user = []
    test_user = np.array(test_user)
    test_item = np.array(test_item)
    index_user = df[["userID"]].groupby('userID')
    list_reviewer = list(index_user)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    threads = tf.train.start_queue_runners(sess=sess)
    angluarLoss = angluarLoss(sess, user_count, item_count, 20)
    feature_u, feature_v = angluarLoss.train_test_validation(df, df_ratings)

    results, result_hr = check_model_ndcg(feature_u, feature_v, df_ratings)

    u_feature_file = open('./hash_user_feature.data', 'rb')
    u_hash_feature = pickle.load(u_feature_file)
    v_feature_file = open('./hash_item_feature.data', 'rb')
    v_hash_feature = pickle.load(v_feature_file)
    uv_feature = update_UV(df, u_hash_feature, v_hash_feature)
    results, results_hr = check_model_ndcg(u_hash_feature, v_hash_feature, df_ratings)
    ndcg_all = [round(k, 4) for k in results]
    ndcg = ', '.join(str(kk) for kk in ndcg_all)
    print('Ours: NDCG@1~10 = ', ndcg)
    hr_all = [round(k, 4) for k in results_hr]
    hr = ', '.join(str(kk) for kk in hr_all)
    print('Ours: HR@1~10 = ', hr)
