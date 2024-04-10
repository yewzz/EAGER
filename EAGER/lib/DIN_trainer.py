
import torch
import numpy as np
from lib.DIN_Model import DeepInterestNetwork
class DINTrain:
    def __init__(self,item_num=100,sample_negative_num=60,emb_dim=96,device='cpu',
                feature_groups=[20,20,10,10,2,2,2,1,1,1],
                sum_pooling=False,
                optimizer=lambda params: torch.optim.Adam(params, lr=1e-3, amsgrad=True)):

        self.item_num=item_num
        self.device=device
        self.N=sample_negative_num

        self.DINModel=DeepInterestNetwork(item_num=item_num,embedding_dim=emb_dim,\
            feature_groups=feature_groups,sum_pooling=sum_pooling).to(self.device)

        # data = torch.from_numpy(
        #     np.load('/home///data/recommend/Amazon_Beauty/t5_fullfeat_norm.np.npy')).cpu()
        # self.DINModel.item_embedding._modules['0'].embed.weight.data[1:,:] = data.cuda()
        # self.DINModel.item_embedding._modules['0'].embed.weight.requires_grad_(False)
        #optimizer
        self.optimizer = optimizer(self.DINModel.parameters())
        self.batch_num=0

    def update_learning_rate(self, t, learning_rate_base=1e-3, warmup_steps=5000,
                             decay_rate=1./3, learning_rate_min=1e-6):
        """ Learning rate with linear warmup and exponential decay """
        lr = learning_rate_base * np.minimum(
            (t + 1.0) / warmup_steps,
            np.exp(decay_rate * ((warmup_steps - t - 1.0) / warmup_steps)),
        )
        lr = np.maximum(lr, learning_rate_min)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr



    def uniform_sampled_softmax(self,batch_users,batch_labels,N):

        batch_size=batch_users.shape[0]
        
        samples = torch.full((batch_size, N+1),-1,device=self.device, dtype=torch.int64)
        log_q_matrix=torch.full(samples.shape,0.0,device=self.device,dtype=torch.float32)
        effective_index=torch.full(samples.shape,True,device=self.device,dtype=torch.bool)
        samples[:,0:1]=batch_labels#positve labels
        samples[:,1:]=torch.randint(low=0,high=self.item_num,size=(batch_size,N),device=self.device)
        effective_index[:,1:]=samples[:,0:1]!=samples[:,1:]
        log_q_matrix[:,1:][effective_index[:,1:]]=\
                            torch.log(effective_index[:,1:].sum(-1).view(batch_size,1)*1.0/(self.item_num-1)).\
                                expand(batch_size,N)[effective_index[:,1:]]
        
        o_pi=torch.full(samples.shape,-1.0e9,device=self.device,dtype=torch.float32)

        user_index = torch.arange(batch_size,device=self.device).view(-1, 1).expand(samples.shape)[effective_index]

        o_pi[effective_index] = self.DINModel(batch_users[user_index],\
                                samples[effective_index].view(-1, 1))[:, 0] - log_q_matrix[effective_index]

        return (torch.logsumexp(o_pi,dim=1)-o_pi[:,0]).mean(-1)

    def update_DIN(self,batch_users,batch_labels):
        self.batch_num+=1
        loss=self.uniform_sampled_softmax(batch_users.to(self.device),batch_labels.to(self.device).view(-1,1),self.N)
        loss.backward()#compute the gradient
        self.optimizer.step()# update the parameters
        self.optimizer.zero_grad()# clean the gradient
        self.update_learning_rate(self.batch_num)
        return loss

    def calculate_preference(self,batch_user,batch_items):
        return self.DINModel(batch_user,batch_items)


        
        