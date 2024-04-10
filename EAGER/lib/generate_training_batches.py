from __future__ import print_function
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def iterate_minibatches(*tensors, batch_size=4096, shuffle=True, cycle=True, **kw):
    while True:
        yield from DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)
        print('cycle')
        # break
        if not cycle:
            break

class Train_instance:
    def __init__(self,parall=4):
        self.parall=parall

    def read_one_instrance_file(self,traing_instaces_path,sec_count_id,item_num):
        file_path=traing_instaces_path+'_{}'.format(sec_count_id)
        historys,labels=[],[]
        with open(file_path) as f:
            for line in f:
                user,history,label=line.split('|')
                historys.append([int(st) for st in history.split(',')])
                labels.append(int(label))
                #line = f.readline()
        one_file_data=torch.LongTensor(historys)
        one_file_data[one_file_data<0]=item_num
        return one_file_data,torch.LongTensor(labels)
    '''
    def read_files(self,traing_instaces_path,sec_count_ids,pipe):
        hs,la=[],[]
        for id in sec_count_ids:
            history,labels=self.read_one_instrance_file(traing_instaces_path,id)
            hs.append(history)
            la.append(labels)
        print('read traning data over at pid {}'.format(os.getpid()))
        pipe.send((torch.cat(hs,dim=0),torch.cat(la,dim=0)))

    def read_all_instances_files(self,traing_instaces_path,seg_couts):
        process = []
        pipes = []
        job_size = int(seg_couts/ self.parall)
        file_couts=list(range(seg_couts))
        if seg_couts % self.parall != 0:
            self.parall += 1
        for i in range(self.parall):
            a, b = mp.Pipe()
            pipes.append(a)
            p = mp.Process(
                target=self.read_files,
                args=(traing_instaces_path,
                      file_couts[i*job_size:(i+1)*job_size],b)
            )
            process.append(p)
            p.start()
        history,lables=[],[]
        for pipe in pipes:
            (his, las) = pipe.recv()
            history.append(his)
            lables.append(las)

        for p in process:
            p.join()
        his=torch.cat(history,dim=0)
        lables=torch.cat(lables,dim=0)
        assert len(his)==len(lables)
        return his,lables

    '''
    def read_all_instances_files(self,traing_instaces_path,seg_couts,item_num, his_maxtix=None,labels=None):
        if False and his_maxtix and labels:
            his_maxtix = torch.load(his_maxtix)
            labels = torch.load(labels)
            assert len(his_maxtix) == len(labels)
            return his_maxtix, labels
        his_maxtix=None
        labels=None
        for i in range(seg_couts):
            print('reading {}-th part'.format(i))
            part_his,part_labels=self.read_one_instrance_file(traing_instaces_path,i,item_num)
            if his_maxtix is not None:
                his_maxtix=torch.cat((his_maxtix,part_his),0)
                labels=torch.cat((labels,part_labels),0)
            else:
                his_maxtix=part_his
                labels=part_labels
        assert len(his_maxtix)==len(labels)
        torch.save(his_maxtix, '/home///data/recommend/Amazon_Beauty/his_maxtix.pt')
        torch.save(labels, '/home///data/recommend/Amazon_Beauty/labels.pt')
        return his_maxtix,labels


    def read_test_instances_file(self,test_instance_path,item_num):
        historys,labels=[],[]
        with open(test_instance_path) as f:
            line=f.readline()
            while line:
                user,history,label=line.split('|')
                historys.append([int(st) for st in history.split(',')])
                labels.append([int(st) for st in label.split(',')])
                line = f.readline()
        self.test_labels=labels
        test_data=torch.LongTensor(historys)
        test_data[test_data<0]=item_num
        return test_data
    
    def read_validation_instances_file(self,validation_instance_path,item_num):
        historys,labels=[],[]
        with open(validation_instance_path) as f:
            line=f.readline()
            while line:
                user,history,label=line.split('|')
                historys.append([int(st) for st in history.split(',')])
                labels.append([int(st) for st in label.split(',')])
                line = f.readline()
        self.validation_labels=labels
        #print(self.validation_labels)
        validation_data=torch.LongTensor(historys)
        validation_data[validation_data<0]=item_num
        return validation_data



    def training_batches(self,traing_instaces_path,seg_couts,item_num,batchsize=300):
        history_matrix,positive_labels=self.read_all_instances_files(traing_instaces_path,seg_couts,item_num)
        tensor_train_instances=TensorDataset(history_matrix,positive_labels)
        train_loader=DataLoader(dataset=tensor_train_instances,batch_size=batchsize,shuffle=True,num_workers=4)
        while True:
            yield from train_loader

    def get_training_data(self,traing_instaces_path,seg_couts,item_num, his_maxtix=None, labels=None):
        history_matrix,positive_labels=self.read_all_instances_files(traing_instaces_path,seg_couts,item_num,his_maxtix, labels)
        '''
        index = np.arange(len(history_matrix))
        np.random.shuffle(index)
        assert len(history_matrix) == len(positive_labels)
        history_matrix = history_matrix[index]
        positive_labels = positive_labels[index]
        '''
        return history_matrix,positive_labels

    def test_batches(self,test_instances_path, item_num,batchsize=100):
        test_instances_matrix=self.read_test_instances_file(test_instances_path,item_num)
        mindex=torch.tensor(np.arange(len(test_instances_matrix)))
        tensor_test_instances=TensorDataset(test_instances_matrix,mindex)
        test_loader=DataLoader(dataset=tensor_test_instances,batch_size=batchsize,shuffle=True,num_workers=4)
        while True:
            yield from test_loader
    
    def validation_batches(self,validation_instances_path, item_num,batchsize=100):
        test_instances_matrix=self.read_validation_instances_file(validation_instances_path,item_num)
        mindex=torch.tensor(np.arange(len(test_instances_matrix)))
        tensor_test_instances=TensorDataset(test_instances_matrix,mindex)
        test_loader=DataLoader(dataset=tensor_test_instances,batch_size=batchsize,shuffle=True,num_workers=4)
        while True:
            yield from test_loader
    
    def generate_training_records(self,trining_instances,trining_labels,batch_size=1024, shuffle=False):
        for batch_x,batch_y in iterate_minibatches(trining_instances,trining_labels,batch_size=batch_size,shuffle=True):
            yield batch_x, batch_y













