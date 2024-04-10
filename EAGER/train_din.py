import torch
import sys

sys.path.append('../..')
#parametres
data_set_name='MIND'#'gowalla' 'MIND' 'Amazon_All_Beauty'
device='cuda:3'
topk=10
optimizer=lambda params: torch.optim.Adam(params, lr=1e-3, amsgrad=True)
have_processed_data=False

from lib.generate_train_and_test_data import _gen_train_sample, _read, _gen_test_sample

emb_dim=96
sum_pooling=False 
sample_negative_num=60 # 60
feature_groups= [5,4,2,2,1,1,1,1,1,1] #[20,20,10,10,2,2,2,1,1,1]  # [20,20,10,10,2,2,2,1,1,1]
train_sample_seg_cnt=10#the training data is located in the train_sample_seg_cnt datafiles
parall=10
seq_len=20 # se_len-1 is the number of behaviours in all the windows
min_seq_len=5
test_user_num=0# the number of user in test file
raw_data_file='/home///data/recommend/{}/{}.txt'.format(data_set_name,data_set_name)
train_instances_file='/home///data/recommend/{}/train_instances'.format(data_set_name)
test_instances_file='/home///data/recommend/{}/test_instances'.format(data_set_name)
validation_instances_file='/home///data/recommend/{}/validation_instances'.format(data_set_name)
item_num_node_num_file='/home///data/recommend/{}/item_node_num.txt'.format(data_set_name)



test_batch_size=100

batch_number=800000#
if device!='cpu':
    torch.cuda.set_device(device)
    device='cuda'

from lib import generate_train_and_test_data


import numpy as np
if not have_processed_data:
    raw_data_file = '/home//2//recommend/Sports_and_Outdoors_5.json'
    behavior_dict, train_sample, test_sample,validation_sample,user_num,item_num, user_ids= generate_train_and_test_data._read(raw_data_file,
                                                                             test_user_num)  # 20 is the test users
    # write the training instance into different train_sample_seg_cnt files， avoid that a file is too large
    # stat record the click frequency of each item
    # seq_len=20 min that 19 behaviors and one label
    stat = generate_train_and_test_data._gen_train_sample(train_sample, train_instances_file,test_sample=test_sample,
                                                    train_sample_seg_cnt=train_sample_seg_cnt,
                                                    parall=parall, seq_len=seq_len, min_seq_len=min_seq_len, user_ids=user_ids)
    # generate_train_and_test_data._gen_test_sample(test_sample, test_instances_file, seq_len=seq_len,
    #                                         min_seq_len=min_seq_len)
    _gen_test_sample(train_sample, test_instances_file, seq_len=seq_len, min_seq_len=min_seq_len)
    _gen_test_sample(validation_sample, validation_instances_file, seq_len=seq_len, min_seq_len=min_seq_len)
    del behavior_dict
    del train_sample
    del test_sample
    del stat
    np.savetxt(item_num_node_num_file,np.array([user_num,item_num]),fmt='%d',delimiter=',')
else:
    [user_num, item_num] = np.loadtxt(item_num_node_num_file, dtype=np.int32, delimiter=',')

print('user num is {}, item is {}'.format(user_num,item_num))

from lib import DINTrain

train_model=DINTrain(item_num=item_num,
                     sample_negative_num=sample_negative_num,
                     emb_dim=emb_dim,
                     device=device,
                     sum_pooling=sum_pooling,
                     feature_groups=feature_groups,
                     optimizer=optimizer)
print(train_model.DINModel)


from lib.generate_training_batches import Train_instance
train_instances=Train_instance(parall=parall)

his_maxtix = '/home///data/recommend/{}/his_maxtix.pt'.format(data_set_name)
labels = '/home///data/recommend/{}/labels.pt'.format(data_set_name)
training_data,training_labels=train_instances.get_training_data(train_instances_file,train_sample_seg_cnt,item_num, his_maxtix, labels)
item_set = set()
# for i in range(len(training_data)):
#     for j in range(19):
#         item_set.add(training_data[i][j].item())
# print(len(item_set))
# print("?")

test_batch_generator=train_instances.test_batches(test_instances_file,item_num,batchsize=test_batch_size)
validation_batch_generator=train_instances.validation_batches(validation_instances_file,item_num,batchsize=test_batch_size)
test_instances=train_instances.read_test_instances_file(test_instances_file,item_num)

from pandas import DataFrame
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
moving_average = lambda x, **kw: DataFrame({'x':np.asarray(x)}).x.ewm(**kw).mean().values
loss_history,dev_precision_history,dev_recall_history,dev_f_measure_history,dev_novelty_history,dev_ndcg_history,policy_acc=[],[],[],[],[],[],[]
test_precision_history,test_recall_history,test_f_measure_history,test_novelty_history,test_ndcg_history=[],[],[],[],[]
total_precision_history,total_recall_history,total_f_measure_history,total_novelty_history,total_ndcg_history, total_hit_history=[],[],[],[],[], []


def presision(result_list,gt_list,top_k):
    count=0.0
    for r,g in zip(result_list,gt_list):
        count+=len(set(r).intersection(set(g)))
    return count/(top_k*len(result_list))
def recall(result_list,gt_list):
    t=0.0
    for r,g in zip(result_list,gt_list):
        t+=1.0*len(set(r).intersection(set(g)))/len(g)
    return t/len(result_list)
def f_measure(result_list,gt_list,top_k,eps=1.0e-9):
    f=0.0
    for r,g in zip(result_list,gt_list):
        recc=1.0*len(set(r).intersection(set(g)))/len(g)
        pres=1.0*len(set(r).intersection(set(g)))/top_k
        if recc+pres<eps:
            continue
        f+=(2*recc*pres)/(recc+pres)
    return f/len(result_list)

def novelty(result_list,s_u,top_k):
    count=0.0
    for r,g in zip(result_list,s_u):
        count+=len(set(r)-set(g))
    return count/(top_k*len(result_list))

def hit_ratio(result_list,gt_list):
    intersetct_set=[len(set(r)&set(g)) for r,g in zip(result_list,gt_list)]
    return 1.0*sum(intersetct_set)/sum([len(gts) for gts in gt_list])

def NDCG_bug(result_list,gt_list):
    t=0.0
    for re,gt in zip(result_list,gt_list):
        setgt=set(gt)
        indicator=np.asfarray([1 if r in setgt else 0 for r in re])
        sorted_indicator=indicator[indicator.argsort(-1)[::-1]]
        if 1 in indicator:
            t+=np.sum(indicator / np.log2(1.0*np.arange(2,len(indicator)+ 2)))/\
               np.sum(sorted_indicator/np.log2(1.0*np.arange(2,len(indicator)+ 2)))
    return t/len(gt_list)

def NDCG_(result_list,gt_list):
    t=0.0
    for re,gt in zip(result_list,gt_list):
        setgt=set(gt)
        indicator=np.asfarray([1 if r in setgt else 0 for r in re])
        sorted_indicator = np.ones(min(len(setgt), len(re)))
        if 1 in indicator:
            t+=np.sum(indicator / np.log2(1.0*np.arange(2,len(indicator)+ 2)))/\
               np.sum(sorted_indicator/np.log2(1.0*np.arange(2,len(sorted_indicator)+ 2)))
    return t/len(gt_list)

import math
def NDCG_comicrec(result_list,gt_list):
    t=0.0
    for re,gt in zip(result_list,gt_list):
        recall = 0
        dcg = 0.0
        setgt=set(gt)
        for no, iid in enumerate(re):
            if iid in setgt:
                recall += 1
                dcg += 1.0 / math.log(no + 2, 2)
        idcg = 0.0
        for no in range(recall):
            idcg += 1.0 / math.log(no + 2, 2)
        if recall > 0:
            t += dcg / idcg
    return t/len(gt_list)

def MAP(result_list,gt_list,topk):
    t=0.0
    for re,gt in zip(result_list,gt_list):
        setgt=set(gt)
        indicator=np.asfarray([1 if r in setgt else 0 for r in re])
        t+=np.mean([indicator[:i].sum(-1)/i for i in range(1,topk+1)],axis=-1)
    return t/len(gt_list)


train_model.DINModel.train()

train_model=DINTrain(item_num=item_num,
                     sample_negative_num=sample_negative_num,
                     emb_dim=emb_dim,
                     device=device,
                     sum_pooling=sum_pooling,
                     feature_groups=feature_groups,
                     optimizer=optimizer)
# DIN_Model_path='/home//2//recommend//data/{}/DIN_MODEL_60000.pt'.format(data_set_name)
# train_model.DINModel=torch.load(DIN_Model_path,map_location=torch.device(device))

# train_model.DINModel.eval()
# gt_history=train_instances.test_labels
# all_items=torch.arange(item_num,device=device).view(-1,1)
# preference_matrix=torch.full((len(test_instances),item_num),-1.0e9,dtype=torch.float32)
# batch_size=2000
# f_num=test_instances.shape[1]
# #print(item_num,test_batch.shape)
# for i,user in enumerate(test_instances):
#     start_id=0
#     while start_id<item_num:
#         part_labels=all_items[start_id:start_id+batch_size,:]
#         #print(len(part_labels),)
#         with torch.no_grad():
#             preference_matrix[i,start_id:start_id+batch_size]=train_model.calculate_preference(\
#                 user.to(device).expand(len(part_labels),f_num),part_labels).view(1,-1).cpu()
#         start_id=start_id+batch_size
#
# topk = 20
# resutl_history=preference_matrix.argsort(dim=-1)[:,-topk:].numpy()
# resutl_history = resutl_history[:, ::-1]

def NDCG(result_list,gt_list):
    t=0.0
    for re,gt in zip(result_list,gt_list):
        setgt=gt
        indicator=np.asfarray([1 if r in setgt else 0 for r in re])
        sorted_indicator = np.ones(len(gt)) #np.ones(min(len(setgt), len(re)))
        if 1 in indicator:
            t+=np.sum(indicator / np.log2(1.0*np.arange(2,len(indicator)+ 2)))/\
               np.sum(sorted_indicator/np.log2(1.0*np.arange(2,len(sorted_indicator)+ 2)))
    return t/len(gt_list)

# resutl_history = resutl_history[:, ::-1]
# ndcg = NDCG(resutl_history,gt_history)
# P = presision(resutl_history,gt_history,topk)
# R = recall(resutl_history,gt_history)
# F = f_measure(resutl_history,gt_history,topk)
# ndcg_bug = NDCG_bug(resutl_history, gt_history)
# ndcg_ = NDCG_(resutl_history, gt_history)
# print(P)
# print(R)
# print(F)
# print(ndcg_bug)
# print(ndcg_)
#
# exit(0)
#
# DIN_Model_path='/home//2//recommend//data/{}/DIN_MODEL.pt'.format(data_set_name)
# torch.save(train_model.DINModel,DIN_Model_path)
#
# sorted_test_users_path='/home//2//recommend//data/{}/sorted_test_users.txt'.format(data_set_name)
# np.savetxt(sorted_test_users_path,preference_matrix.argsort(dim=-1).numpy(),delimiter=',',fmt='%d')
# #%%
# from lib.generate_training_batches import Train_instance
# train_instances=Train_instance(parall=parall)
# test_instances=train_instances.read_test_instances_file(test_instances_file,item_num)
#
# sorted_test_users_path='/home//2//recommend//data/{}/sorted_test_users.txt'.format(data_set_name)
# gt_history=train_instances.test_labels
# preference_matrix=np.loadtxt(sorted_test_users_path,delimiter=',')
# #%%
# topk=20
# resutl_history=np.array(preference_matrix[:,-1:-topk-1:-1],dtype=np.int32)
# P = presision(resutl_history,gt_history,topk)
# R = recall(resutl_history,gt_history)
# F = f_measure(resutl_history,gt_history,topk)
# N = novelty(resutl_history,test_instances.tolist(),topk)
# hr_ = hit_ratio(resutl_history,gt_history)
# ndcg_ = NDCG(resutl_history,gt_history)
# map_ = MAP(resutl_history,gt_history,topk)
# print("{:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}".format(P,R,F,N,hr_,ndcg_,map_))
#
# print(np.array(preference_matrix,dtype=np.int32))
# exit(0)


for (batch_x, batch_y) in train_instances.generate_training_records(training_data, training_labels, batch_size=128):
    # print(batch_x,batch_y)
    loss = train_model.update_DIN(batch_x, batch_y)
    loss_history.append(loss.item())


    if train_model.batch_num % 1000 == 0:
        # train_model.DINModel.eval()
        # resutl_history=train_model.predict(test_instances,topk=topk).numpy()#40 is N
        # total_precision_history.append(presision(resutl_history,train_instances.test_labels,topk))
        # total_recall_history.append(recall(resutl_history,train_instances.test_labels))
        # total_f_measure_history.append(f_measure(resutl_history,train_instances.test_labels,topk))
        # total_novelty_history.append(novelty(resutl_history,test_instances.tolist(),topk))
        # train_model.DINModel.train()

        # ###start to test
        train_model.DINModel.eval()
        # test_batch, test_index = validation_batch_generator.__next__()
        # gt_history = [train_instances.validation_labels[i.item()] for i in test_index]
        gt_history = train_instances.test_labels

        all_items = torch.arange(item_num, device=device).view(-1, 1)
        preference_matrix = torch.full((len(test_instances), item_num), -1.0e9, dtype=torch.float32)
        batch_size = 2000
        print(test_instances.shape)
        f_num = test_instances.shape[1]
        # print(item_num,test_batch.shape)
        for i, user in enumerate(test_instances):
            start_id = 0
            while start_id < item_num:
                part_labels = all_items[start_id:start_id + batch_size, :]
                # print(len(part_labels),)
                with torch.no_grad():
                    preference_matrix[i, start_id:start_id + batch_size] = train_model.calculate_preference( \
                        user.to(device).expand(len(part_labels), f_num), part_labels).view(1, -1).cpu()
                start_id = start_id + batch_size
        resutl_history = preference_matrix.argsort(dim=-1)[:, -topk:].numpy()
        resutl_history = resutl_history[:, ::-1]
        total_precision_history.append(presision(resutl_history, gt_history, topk))
        total_recall_history.append(recall(resutl_history, gt_history))
        total_f_measure_history.append(f_measure(resutl_history, gt_history, topk))
        total_novelty_history.append(novelty(resutl_history, test_instances.tolist(), topk))
        total_ndcg_history.append(NDCG(resutl_history, gt_history))
        total_hit_history.append(hit_ratio(resutl_history, gt_history))
        print('precision: {}'.format(total_precision_history[-1]))
        print('recall: {}'.format(total_recall_history[-1]))
        print('f-score: {}'.format(total_f_measure_history[-1]))
        print('ndcg: {}'.format(total_ndcg_history[-1]))
        print('hit_rate: {}'.format(total_hit_history[-1]))
        train_model.DINModel.train()
        # #######

    if train_model.batch_num % 1000 == 0:
        DIN_Model_path = '/home//2//recommend//data/{}/DIN_MODEL_{}.pt'.format(
            data_set_name, (train_model.batch_num))
        torch.save(train_model.DINModel, DIN_Model_path)
        print('saved')

    if train_model.batch_num % 100 == 0:
        # ###start to test
        # train_model.DINModel.eval()
        # test_batch,test_index=test_batch_generator.__next__()
        # gt_history=[train_instances.test_labels[i.item()] for i in test_index]
        # resutl_history=train_model.predict(test_batch,topk=topk).numpy()
        # test_precision_history.append(presision(resutl_history,gt_history,topk))
        # test_recall_history.append(recall(resutl_history,gt_history))
        # test_f_measure_history.append(f_measure(resutl_history,gt_history,topk))
        # test_novelty_history.append(novelty(resutl_history,test_batch.tolist(),topk))
        # test_ndcg_history.append(NDCG(resutl_history, gt_history))
        # train_model.DINModel.train()
        # #######

        # clear_output(True)
        # plt.figure(figsize=[18, 12])
        # plt.subplot(2, 3, 1)
        # plt.title('train loss over time');
        # plt.grid();
        # plt.plot(moving_average(loss_history, span=50))
        # plt.scatter(range(len(loss_history)), loss_history, alpha=0.1)
        #
        # plt.subplot(2, 3, 2)
        # plt.title('dev presision over time');
        # plt.grid();
        # # plt.plot(moving_average(test_precision_history, span=50))
        # # plt.scatter(range(len(test_precision_history)), test_precision_history, alpha=0.1)
        # plt.plot(50 * (np.arange(len(total_precision_history)) + 1), total_precision_history, c='r')
        #
        # plt.subplot(2, 3, 3)
        # plt.title('dev recall over time');
        # plt.grid();
        # # plt.plot(moving_average(test_recall_history, span=10))
        # # plt.scatter(range(len(test_recall_history)), test_recall_history, alpha=0.1)
        # plt.plot(50 * (np.arange(len(total_recall_history)) + 1), total_recall_history, c='r')
        #
        # plt.subplot(2, 3, 4)
        # plt.title('dev f-measure over time');
        # plt.grid();
        # # plt.plot(moving_average(test_f_measure_history, span=10))
        # # plt.scatter(range(len(test_f_measure_history)), test_f_measure_history, alpha=0.1)
        # plt.plot(50 * (np.arange(len(total_f_measure_history)) + 1), total_f_measure_history, c='r')
        #
        # plt.subplot(2, 3, 5)
        # plt.title('dev novelty over time');
        # plt.grid();
        # # plt.plot(moving_average(test_novelty_history, span=10))
        # # plt.scatter(range(len(test_novelty_history)), test_novelty_history, alpha=0.1)
        # plt.plot(50 * (np.arange(len(total_novelty_history)) + 1), total_novelty_history, c='r')
        #
        # plt.subplot(2, 3, 6)
        # plt.title('dev ndcg over time');
        # plt.grid();
        # # plt.plot(moving_average(test_ndcg_history, span=10))
        # # plt.scatter(range(len(test_ndcg_history)), test_ndcg_history, alpha=0.1)
        # plt.plot(50 * (np.arange(len(total_ndcg_history)) + 1), total_ndcg_history, c='r')

        # plt.show()

        print("step=%i, mean_loss=%.3f, time=%.3f" %
              (len(loss_history), np.mean(loss_history[-100:]), 1.0))
        # print('_' * 100)

    if train_model.batch_num > batch_number:
        break


# train_model.DINModel.eval()
# resutl_history=train_model.predict(test_instances,topk=topk).numpy()#40 is N
# total_precision_history.append(presision(resutl_history,train_instances.test_labels,topk))
# total_recall_history.append(recall(resutl_history,train_instances.test_labels))
# total_f_measure_history.append(f_measure(resutl_history,train_instances.test_labels,topk))
# total_novelty_history.append(novelty(resutl_history,test_instances.tolist(),topk))
# train_model.DINModel.train()

# train_model=DINTrain(item_num=item_num,
#                      sample_negative_num=sample_negative_num,
#                      emb_dim=emb_dim,
#                      device=device,
#                      sum_pooling=sum_pooling,
#                      feature_groups=feature_groups,
#                      optimizer=optimizer)
# DIN_Model_path='/home//2//recommend//data/{}/DIN_MODEL.pt'.format(data_set_name)
# train_model.DINModel=torch.load(DIN_Model_path,map_location=torch.device(device))
train_model.DINModel.eval()
gt_history=train_instances.test_labels
all_items=torch.arange(item_num,device=device).view(-1,1)
preference_matrix=torch.full((len(test_instances),item_num),-1.0e9,dtype=torch.float32)
batch_size=2000
f_num=test_instances.shape[1]
#print(item_num,test_batch.shape)
for i,user in enumerate(test_instances):
    start_id=0
    while start_id<item_num:
        part_labels=all_items[start_id:start_id+batch_size,:]
        #print(len(part_labels),)
        with torch.no_grad():
            preference_matrix[i,start_id:start_id+batch_size]=train_model.calculate_preference(\
                user.to(device).expand(len(part_labels),f_num),part_labels).view(1,-1).cpu()
        start_id=start_id+batch_size
resutl_history=preference_matrix.argsort(dim=-1)[:,-topk:].numpy()
total_precision_history.append(presision(resutl_history,gt_history,topk))
total_recall_history.append(recall(resutl_history,gt_history))
total_f_measure_history.append(f_measure(resutl_history,gt_history,topk))
# total_novelty_history.append(novelty(resutl_history,test_batch.tolist(),topk))
train_model.DINModel.train()

DIN_Model_path='/home//2//recommend//data/{}/DIN_MODEL.pt'.format(data_set_name)
torch.save(train_model.DINModel,DIN_Model_path)
print(total_precision_history[-1],total_recall_history[-1],total_f_measure_history[-1],total_novelty_history[-1])

sorted_test_users_path='/home//2//recommend//data/{}/sorted_test_users.txt'.format(data_set_name)
np.savetxt(sorted_test_users_path,preference_matrix.argsort(dim=-1).numpy(),delimiter=',',fmt='%d')

def presision(result_list,gt_list,top_k):
    count=0.0
    for r,g in zip(result_list,gt_list):
        count+=len(set(r).intersection(set(g)))
    return count/(top_k*len(result_list))
def recall(result_list,gt_list):
    t=0.0
    for r,g in zip(result_list,gt_list):
        t+=1.0*len(set(r).intersection(set(g)))/len(g)
    return t/len(result_list)
def f_measure(result_list,gt_list,top_k,eps=1.0e-9):
    f=0.0
    for r,g in zip(result_list,gt_list):
        recc=1.0*len(set(r).intersection(set(g)))/len(g)
        pres=1.0*len(set(r).intersection(set(g)))/top_k
        if recc+pres<eps:
            continue
        f+=(2*recc*pres)/(recc+pres)
    return f/len(result_list)
def novelty(result_list,s_u,top_k):
    count=0.0
    for r,g in zip(result_list,s_u):
        count+=len(set(r)-set(g))
    return count/(top_k*len(result_list))
def hit_ratio(result_list,gt_list):
    intersetct_set=[len(set(r)&set(g)) for r,g in zip(result_list,gt_list)]
    return 1.0*sum(intersetct_set)/sum([len(gts) for gts in gt_list])

def NDCG_bug(result_list,gt_list):
    t=0.0
    for re,gt in zip(result_list,gt_list):
        setgt=set(gt)
        indicator=np.asfarray([1 if r in setgt else 0 for r in re])
        sorted_indicator=indicator[indicator.argsort(-1)[::-1]]
        if 1 in indicator:
            t+=np.sum(indicator / np.log2(1.0*np.arange(2,len(indicator)+ 2)))/\
               np.sum(sorted_indicator/np.log2(1.0*np.arange(2,len(indicator)+ 2)))
    return t/len(gt_list)

def MAP(result_list,gt_list,topk):
    t=0.0
    for re,gt in zip(result_list,gt_list):
        setgt=set(gt)
        indicator=np.asfarray([1 if r in setgt else 0 for r in re])
        t+=np.mean([indicator[:i].sum(-1)/i for i in range(1,topk+1)],axis=-1)
    return t/len(gt_list)
#%%
from lib.generate_training_batches import Train_instance
train_instances=Train_instance(parall=parall)
test_instances=train_instances.read_test_instances_file(test_instances_file,item_num)

sorted_test_users_path='/home//2//recommend//data/{}/sorted_test_users.txt'.format(data_set_name)
gt_history=train_instances.test_labels
preference_matrix=np.loadtxt(sorted_test_users_path,delimiter=',')
#%%
topk=40
resutl_history=np.array(preference_matrix[:,-1:-topk-1:-1],dtype=np.int32)
P = presision(resutl_history,gt_history,topk)
R = recall(resutl_history,gt_history)
F = f_measure(resutl_history,gt_history,topk)
N = novelty(resutl_history,test_instances.tolist(),topk)
hr_ = hit_ratio(resutl_history,gt_history)
ndcg_ = NDCG(resutl_history,gt_history)
map_ = MAP(resutl_history,gt_history,topk)
print("{:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}".format(P,R,F,N,hr_,ndcg_,map_))

print(np.array(preference_matrix,dtype=np.int32))
