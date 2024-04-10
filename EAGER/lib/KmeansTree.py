
import time
import collections
#import argparse

#import multiprocessing as mp
import numpy as np
import math
from joblib import Parallel,delayed
import queue as Que
import torch
from tqdm import tqdm

def pairwise_distance_function(data1, data2):
    # #data1 [m,dim],data2 [k,dim]
    bsz = len(data1)

    times = bsz // 4 # 1024 for 256    1296 for 18
    res = []
    for t in range(times):
        cur_data1 = data1[t*4:(t+1)*4]
        res_tmp = ((cur_data1.unsqueeze(dim=1)-data2) ** 2.0).sum(-1)
        res.append(res_tmp)
    res = torch.cat(res, dim=0)
    return res
    # return  ((data1.unsqueeze(dim=1)-data2) ** 2.0).sum(-1)#[m,k]



def initialize(X, num_clusters):
    indices = torch.randperm(X.shape[0], device=X.device)
    #return torch.gather(X, 0, indices.unsqueeze(-1).expand(X.shape)).view(num_clusters, -1, X.shape[-1]).mean(dim=1)#[num_clusters,dim]
    return X[indices].view(num_clusters, -1, X.shape[-1]).mean(dim=1)#[num_clusters,dim]

def kmeans_equal(
        X,
        num_clusters=2,
        cluster_size=10,
        max_iters=100,
        initial_state=None,
        update_centers=True,
        tol=1e-6):
    assert X.shape[0]==num_clusters*cluster_size,'data point size should be the product of num_clusters and cluster_size'
    if initial_state is None:
        # randomly group vectors to clusters (forgy initialization)
        initial_state = initialize(X, num_clusters)##[num_clusters,dim]
    iteration = 0

    final_choice=torch.full((X.shape[0],),-1,dtype=torch.int64,device=X.device)
    left_index=torch.full((X.shape[0],),True,dtype=torch.bool,device=X.device)
    all_ins_ids=torch.arange(X.shape[0],device=X.device)
    while True:
        #choices is [num_sample,num_cluster],remark the cluster rank
        #start_t = time.time()
        choices = torch.argsort(pairwise_distance_function(X, initial_state), dim=-1)
        #print(time.time()-start_t)
        
        initial_state_pre = initial_state.clone()
        left_index[:]=True
        for index in torch.randperm(num_clusters):
            cluster_positions = torch.argmax((choices[left_index] == index).to(torch.long), dim=-1)#cluster_positions is [left_num_sample]

            #choose the most colse cluster_size samples to cluster index,selected_ind is [cluster_size]
            selected_ind=all_ins_ids[left_index].gather(dim=0,index=torch.argsort(cluster_positions, dim=-1)[:cluster_size])
            #print(selected_ind)

            final_choice.scatter_(0, selected_ind, value=index)
            left_index.scatter_(0,selected_ind,value=False)
            # update cluster center

            if update_centers:#initial_state is [num_clusters,dim]
                initial_state[index] = torch.gather(X, 0, index=selected_ind.view(-1,1).expand(cluster_size,X.shape[1])).mean(dim=0)
        center_shift =torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)).sum()

        # increment iteration
        iteration = iteration + 1

        if center_shift ** 2 < tol:
            break
        if iteration >= max_iters:
            break

    return final_choice, initial_state

class ConstructKmeansTree:
    def __init__(self,parall):
        self.timeout=5
        self.parall=parall
    def train(self,data=None,k=4,max_iters=100,feature_ratio=0.8):
        assert data is not None,'provide data please'
        self.k=k
        self.max_iters=max_iters
        self.feature_ratio=feature_ratio
        #id and data are tensors

        self.data = data
        self.ids = torch.arange(data.shape[0],dtype=torch.int64)

        queue=Que.Queue()
        mini_batch_queue=Que.Queue()
        print('get into cluster training process')
        
        while True:
            try:
                #print('start')
                queue.put((0, self.ids),timeout=self.timeout)#parent code and index
                break
            except:
                print('put item into queue error!!')
                pass
        assert queue.qsize()==1
        print('start to cluster')
        while queue.qsize()>0:
            pcode,index=queue.get()
            if len(index)<=1024:
                #self._minbatch(pcode, index, code)
                while True:
                    try:
                        mini_batch_queue.put((pcode,index),timeout=self.timeout)
                        break
                    except:
                        print('1024 mini batch error')
                        pass
            else:
                tstart = time.time()
                result_matrix= self._kmeans(index)
                print("Train iteration done, pcode:{}, "
                            "data size: {}, elapsed time: {}"
                            .format(pcode, len(index), time.time() - tstart))
                self.timeout = int(
                    0.4 * self.timeout + 0.6 * (time.time() - tstart))
                if self.timeout < 5:
                    self.timeout = 5

                for c in range(self.k):
                    while True:
                        try:
                            queue.put((self.k * pcode + c+1, result_matrix[c]),
                                        timeout=self.timeout)  # cluster is from root to leaf node
                            break
                        except:
                            print('err')
                            pass

        print('start to process mini-batch parallel.....................................')
        tstart=time.time()
        qcodes,indice=[],[]
        while mini_batch_queue.qsize()>0:
            pcode,index=mini_batch_queue.get()
            qcodes.append(pcode)
            indice.append(index)
        make_job = delayed(self._minbatch)
        re = Parallel(n_jobs=self.parall)(make_job(pcode,index) for pcode,index in zip(qcodes, indice))
        #re = [self._minbatch(pcode,index) for pcode,index in zip(qcodes, indice)]
        id_code_list=[]#(item_id,code),the code is the code of leaf node
        for r in re:
            id_code_list.extend(r)
        id_code_list.sort(key=lambda x:x[0])
        ids = torch.LongTensor([id for (id, _) in id_code_list])
        codes=torch.LongTensor([code for (_,code) in id_code_list])
        print('cluster all the nodes, cost {} s'.format(time.time()-tstart))
        assert (codes<=0).sum()<=0
        assert queue.qsize()==0
        assert len(ids)==len(data)
        return ids,codes#builder.build(ids, codes, stat=self.stat, kv_file=self.kv_file)


    def _minbatch(self, pcode, index):
            #pocde is paretn code,index is the assinged items' id, code is np.zeros(len(ids)), used tor recoder whetert the item processed as a leaf
            dq = collections.deque()
            dq.append((pcode, index))
            batch_size = len(index)
            id_code_list=[]#(item_id,node_code)
            tstart = time.time()
            while dq:
                pcode, index = dq.popleft()# pop the tuple which is added into the deque early
                if len(index) == self.k:
                    for i in range(self.k):
                        id_code_list.append((index[i].item(), self.k * pcode + 1+i))#(in,code) pair
                elif len(index)>self.k:
                    result_matrix = self._kmeans(index)# divide the index into two nodes
                    if result_matrix.shape[1]>=self.k:
                        for c in range(self.k):
                            dq.append((self.k * pcode +c+ 1, result_matrix[c]))
                    else:
                        assert False,'wrong size of index'
                else:
                    assert False,'wrong size of index'
            print("Minbatch, batch size: {}, elapsed: {}".format(batch_size, time.time() - tstart))
            return id_code_list

    def _kmeans(self,index):
        #index is tensor
        #make sure that index.length is greater or equal to k,data is tensor matrix
        #data is tensor
        assert len(index)%self.k==0
        cluster_size=len(index)//self.k
        with torch.no_grad():
            self.data = self.data.cuda()
            selected_feature=torch.randperm(self.data.shape[1])[:math.ceil(self.feature_ratio*self.data.shape[1])]
            choices, _ = kmeans_equal(self.data[:,selected_feature][index], cluster_size=cluster_size,num_clusters=self.k,max_iters=self.max_iters)
        result_index=torch.full((self.k,cluster_size),-1,dtype=torch.int64,device=index.device)
        for c in range(self.k):
            result_index[c]=index[choices==c]
        self.data = self.data.cpu()
        result_index = result_index.cpu()
        return result_index






