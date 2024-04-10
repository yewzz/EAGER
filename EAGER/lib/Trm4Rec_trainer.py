# from lib2to3.pgen2 import token
# from operator import index
import torch
import torch.nn.functional as F
from lib.HF_Model import HFTransformerModel as TransformerModel
from lib.Tree_Model import Tree
import numpy as np


class Trm4Rec:
    def __init__(self,
                 item_num = 1000,
                 user_seq_len = 69,
                 d_model = 24,
                 d_model2 = 96,
                 nhead = 4,
                 device = 'cuda',
                 optimizer=lambda params: torch.optim.Adam(params, lr=1e-3, amsgrad=True),
                 enc_num_layers = 4,
                 dec_num_layers = 4,
                 k = 2,
                 item_to_code_file=None,
                 code_to_item_file=None,
                 tree_has_generated=False,
                 init_way='random',
                 data=None,
                 max_iters=100,
                 feature_ratio=0.8,
                 parall=4,
                 position_embedding_type='absolute',
                 type=0
                 ):
        self.item_num = item_num
        self.device = device
        self.k = k#k branch on each tree
        self.opti=optimizer
        if tree_has_generated:
                self.tree=Tree(construct=False) 
                self.tree.read_tree(item_to_code_file=item_to_code_file,code_to_item_file=code_to_item_file,k=k)
        else:
            self.tree = Tree(data=data,max_iters=max_iters,feature_ratio=feature_ratio,\
                                item_num=item_num,k=k,init_way=init_way,parall=parall)
            np.save(code_to_item_file,self.tree.code_to_item.cpu().numpy())
            item_to_code_mat=torch.full((item_num,self.tree.tree_height),-1,dtype=torch.int64)
            for item_id,paths in self.tree.item_to_code.items():
                assert len(paths)>0
                item_to_code_mat[item_id]=paths[0]
            self.tree.item_to_code = item_to_code_mat.to(self.device)
            np.save(item_to_code_file,item_to_code_mat.numpy())

        self.item_num = item_num
        self.src_voc_size = item_num * 2 + 1#2 + 4 * 2
        self.tgt_voc_size = (k*k+2+k)   #k + 2
        self.max_src_len = user_seq_len
        self.max_tgt_len = self.tree.tree_height + 1


        self.trm_model = TransformerModel(src_voc_size=self.src_voc_size, 
                                            tgt_voc_size=self.tgt_voc_size, 
                                            max_src_len=self.max_src_len,
                                            max_tgt_len=self.max_tgt_len, 
                                            d_model=d_model,
                                            d_model2=d_model2,
                                            nhead=nhead, 
                                            device=device,
                                            enc_num_layers=enc_num_layers,
                                            dec_num_layers=dec_num_layers,
                                            position_embedding_type=position_embedding_type,
                                            item_num=item_num).to(self.device)
        
        self.batch_num = 0
        self.optimizer = self.opti(self.trm_model.parameters())
        self.type = type

        ori_code = self.tree.item_to_code
        self.tree.item_to_code = torch.zeros(len(self.tree.item_to_code)+1, self.tree.item_to_code.size(-1), dtype=torch.int64).cuda()
        self.tree.item_to_code[:-1] = ori_code

        import math
        end_idx = k #int(math.sqrt(self.tgt_voc_size))
        self.end_idx = end_idx
        self.tree.item_to_code[-1] = torch.tensor([end_idx-1, end_idx-1]).cuda()

    def update_learning_rate(self, t, learning_rate_base=1e-3, warmup_steps=5000,
                             decay_rate=1./3, learning_rate_min=1e-5):
        """ Learning rate with linear warmup and exponential decay """
        lr = learning_rate_base * np.minimum(
            (t + 1.0) / warmup_steps,
            np.exp(decay_rate * ((warmup_steps - t - 1.0) / warmup_steps)),
        )
        lr = np.maximum(lr, learning_rate_min)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
        
    def update_model(self, batch_x, batch_y, data_emb=None, type=0, use_con=False, use_guide=False, guide_feat=None):
        self.batch_num+=1
        #y = self.tree_learner.label_to_code(batch_y)#the encode sequence of the item

        temp_y = self.tree.item_to_code[batch_y]
        seq_x = self.tree.item_to_code[batch_x]
        # temp_y = temp_y + torch.tensor([0, 256]).cuda()

        assert temp_y.shape[0]==batch_x.shape[0] and temp_y.shape[1]==self.tree.tree_height
        #print(temp_y.dtype)

        output, contra_loss, guide_feat_ = self.trm_model(batch_x.to(self.device), seq_x.to(self.device), temp_y.to(self.device), data_emb.to(self.device),
                                batch_y.to(self.device), type=type, use_con=use_con, use_guide=use_guide, guide_feat=guide_feat)
        loss = output.loss #+ contra_loss
        return loss, contra_loss, guide_feat_
        # return loss

    def compute_scores(self, batch_x, batch_y, type=0):
        seq_x = self.tree.item_to_code[batch_x]
        temp_y = self.tree.item_to_code[batch_y]
        # temp_y[:, -1] = batch_y
        # temp_y += torch.tensor([0, 256]).cuda()
        attention_mask=(batch_x != self.trm_model.src_pad)
        #print(temp_y)

        prompt_y = temp_y + 1 - 1
        prompt_y[:, 1] = self.end_idx * prompt_y[:, 0] + prompt_y[:, 1] + self.end_idx
        prompt_y = torch.cat([torch.tensor(self.trm_model.trm.config.decoder_start_token_id).cuda().view(1, 1).expand(temp_y.size(0), -1), prompt_y, torch.tensor(self.trm_model.tgt_cls).cuda().view(1, 1).expand(temp_y.size(0), -1),
                             ], dim=-1)
        tgt_key_padding_mask = (prompt_y != self.trm_model.tgt_pad)
        # input_embeds = self.trm_model.trm.decoder.get_input_embeddings()(batch_y[:, 0] * 256 + batch_y[:, 1])
        identifier_x = seq_x[:, :, 0] * self.end_idx + seq_x[:, :, 1] + self.end_idx
        identifier_x = identifier_x.long()
        # input_embeds = self.trm_model.trm.decoder.get_input_embeddings()(identifier_x.long())
        emb_x = batch_x + 1 - 1
        if type == 1:
            emb_x[emb_x != self.item_num] += (self.item_num + 1)
        input_embeds = self.trm_model.trm.encoder.get_input_embeddings()(emb_x.long())
        output = self.trm_model.trm(inputs_embeds=input_embeds,
                     attention_mask=attention_mask,  # [batch_size,seq_length]
                     decoder_input_ids=prompt_y,
                     decoder_attention_mask=tgt_key_padding_mask,
                     output_hidden_states=True)
        #print(2,time.time()-start_time)
        #start_time = time.time()
        # logits = output.logits[:,:-2,:self.tgt_voc_size-2]
        # loss = F.cross_entropy(logits.transpose(1,2), temp_y, reduction='none')

        final_hidden = output.decoder_hidden_states[-1]  # [bsz, l, dim]
        dec_emb = self.trm_model.trm.decoder.get_input_embeddings().weight[:-3]  # .view(256, 256, -1)
        # dec_emb_l1 = dec_emb[::self.end_idx]
        # dec_emb_l2 = dec_emb.view(self.end_idx, self.end_idx, -1)[temp_y[:, 0]]
        dec_emb_l1 = dec_emb[:self.end_idx]
        dec_emb_l2 = dec_emb[self.end_idx:].view(self.end_idx, self.end_idx, -1)[temp_y[:, 0]]
        logits1 = torch.matmul(final_hidden, dec_emb_l1.unsqueeze(0).transpose(1, 2))
        logits2 = torch.matmul(final_hidden, dec_emb_l2.transpose(1, 2))

        loss1 = F.cross_entropy(logits1[:, 0, :].contiguous().view(temp_y.size(0), -1), temp_y[:, 0], reduction='none')
        loss2 = F.cross_entropy(logits2[:, 1, :].contiguous().view(temp_y.size(0), -1), temp_y[:, 1], reduction='none')
        loss = torch.cat([loss1.view(-1, 1), loss2.view(-1, 1)], dim=-1)
        #print(3,time.time()-start_time)
        #self.trm_model.train()
        return -loss

    def predict(self, batch_x, topk=24, num_beams=100, type=0):

        # batch_x = torch.cat([torch.tensor(self.trm_model.encoder_start_token_id).cuda().view(1, 1).expand(batch_x.size(0), -1), batch_x], dim=-1)

        batch_size=batch_x.shape[0]
        #start_time = time.perf_counter()
        input_ids = torch.zeros((batch_size*num_beams,batch_x.shape[1]),device=self.device,dtype=torch.int64)
        #print('init1',time.perf_counter()-start_time)
        #start_time=time.perf_counter()
        select_index = torch.arange(batch_size).view(-1,1).repeat(1,num_beams).to(self.device)
        #print('init2',time.perf_counter()-start_time)
        #start_time = time.perf_counter()
        x = batch_x

        input_batch_size=x.shape[0]
        #input_ids = x.repeat_interleave(num_beams,dim=0)
        input_ids[0:input_batch_size*num_beams]=x[select_index[0:input_batch_size].view(-1)]
        #input_ids=x.to(self.device)[torch.arange(input_batch_size).view(-1,1).repeat(1,num_beams).view(-1)]
        attention_mask=(input_ids[0:input_batch_size*num_beams] != self.trm_model.src_pad)
        pred_scores = torch.full((input_batch_size,num_beams),-1e9,dtype=torch.float32,device=self.device)
        pred_scores[:,0]=0

        pred = torch.full((input_batch_size * num_beams, 1),self.trm_model.trm.config.decoder_start_token_id,dtype=torch.int64,device=self.device)

        import math
        pred_last_token = torch.arange(0,int(math.sqrt(self.tgt_voc_size)),device=self.device).repeat(input_batch_size*num_beams).view(input_batch_size,-1,1)
        #print(model_kwargs['encoder_hidden_states'].shape)

        #init_time+=time.perf_counter()-start_time


        for j in range(self.max_tgt_len-1):
        #pred,pred_scores = self.trm_model.trm.generate(num_beams=num_beams,**model_kwargs,num_return_sequences=num_beams,do_sample=False, max_length=self.max_tgt_len-1)
        #print(pred.shape)
            #start_time = time.perf_counter()
            with torch.no_grad():
                input_x = input_ids[0:input_batch_size * num_beams]
                seq_x = self.tree.item_to_code[input_x]
                identifier_x = seq_x[:, :, 0] * self.end_idx + seq_x[:, :, 1] + self.end_idx
                identifier_x = identifier_x.long()

                emb_x = input_x.long() + 1 - 1
                if type == 1:
                    emb_x[emb_x != self.item_num] += (self.item_num + 1)
                # input_embeds = self.trm_model.trm.decoder.get_input_embeddings()(identifier_x)

                input_embeds = self.trm_model.trm.encoder.get_input_embeddings()(emb_x.long())
                output = self.trm_model.trm(inputs_embeds=input_embeds,decoder_input_ids=pred,\
                    attention_mask=attention_mask,output_attentions=True,   output_hidden_states=True
                    )
            #print('compute',time.perf_counter()-start_time)
            final_hidden = output.decoder_hidden_states[-1]  # [bsz, l, dim]
            if j == 0:
                dec_emb = self.trm_model.trm.decoder.get_input_embeddings().weight[:-3]  # .view(256, 256, -1)
                cls_emb = dec_emb[:self.end_idx].unsqueeze(0)
            else:
                dec_emb = self.trm_model.trm.decoder.get_input_embeddings().weight[:-3]
                cls_emb = dec_emb[self.end_idx:].view(self.end_idx, self.end_idx, -1)[pred[:, -1]]
            logits = torch.matmul(final_hidden, cls_emb.transpose(1, 2))
            # if j == 0:
            #     last_token_logits = logits[:, -1, :]
            #     last_token_logits[:, 256:] = -9999
            # else:
            #     last_token_logits = logits[:, -1, :]
            #     last_token_logits[:, :256] = -9999
            last_token_logits = logits[:, -1, :]
            last_token_logits = last_token_logits.view(input_batch_size,num_beams,-1)
            last_token_logits = torch.log_softmax(last_token_logits,dim=-1)[:,:,:]
            pred_scores = (pred_scores.view(input_batch_size,num_beams,1)+last_token_logits).view(input_batch_size,-1)
            #start_time = time.perf_counter()
            #pred = pred.repeat_interleave(self.tgt_voc_size-2,dim=1)
            # if j == 0 and type == 1:
            #     pred = pred[:, -1]
            import math
            pred = pred.view(input_batch_size*num_beams,1,-1).repeat(1,int(math.sqrt(self.tgt_voc_size)),1).view(input_batch_size,num_beams*(int(math.sqrt(self.tgt_voc_size))),-1)
            pred = torch.cat([pred,pred_last_token],dim=-1)
            if pred.shape[-1] == self.max_tgt_len:
                pred_scores, index = pred_scores.topk(topk)
                index = index.unsqueeze(-1).expand(-1,-1,pred.shape[-1])
                pred = pred.gather(dim=1,index=index).view(input_batch_size*topk,-1)    
            else:
                # argsort_ = pred_scores.argsort(dim=-1,descending=True)[:,:num_beams]
                # pred_scores = pred_scores.gather(dim=1,index=argsort_)
                # index = argsort_.unsqueeze(-1).expand(-1,-1,pred.shape[-1])
                # pred = pred.gather(dim=1,index=index).view(x.shape[0]*num_beams,-1)
                pred_scores, index = pred_scores.topk(num_beams)
                index = index.unsqueeze(-1).expand(-1,-1,pred.shape[-1])
                pred = pred.gather(dim=1,index=index).view(input_batch_size*num_beams,-1)

            
            #compute_time += time.perf_counter()-start_time
            #torch.cuda.empty_cache()
        #start_time = time.perf_counter()
        #print(pred.shape)
        #all_pred.append(pred)
        #append_time += time.perf_counter() - start_time
        #all_pred = torch.cat(all_pred,dim=0)
        #all_pred = all_pred.view(batch_x.shape[0],topk,self.max_tgt_len)[:,:,1:]

        #start_time = time.perf_counter()
        all_pred = pred.view(batch_x.shape[0],topk,-1)[:,:,1:]
        label = self.decode(all_pred, type)# [batch_size,topk*tree_num]
        #decode_time+=time.perf_counter()-start_time
        #self.trm_model.train()
        #print(compute_time,decode_time,init_time,append_time)
        return label
    
    # def predict_hf(self, batch_x, topk=24, num_beams=100, batch_size=50):
    #     self.trm_model.eval()
    #     num_batch = int(math.ceil(batch_x.shape[0] / batch_size))
    #     all_pred = []
    #     for i in range(num_batch): 
    #         x = batch_x[i*batch_size:(i+1)*batch_size].to(self.device)
    #         model_kwargs = {
    #             "input_ids": x,
    #             "attention_mask": torch.FloatTensor((x != self.trm_model.src_pad).cpu().numpy()).to(self.device)
    #         }
    #         #print(model_kwargs)
    #         with torch.no_grad():
    #             pred,pred_scores = self.trm_model.trm.generate(num_beams=num_beams,**model_kwargs,num_return_sequences=num_beams,do_sample=False, max_length=self.max_tgt_len-1)
    #             last_token_logits = self.trm_model.trm(input_ids=model_kwargs['input_ids'].repeat_interleave(num_beams,dim=0), \
    #                 attention_mask=model_kwargs['attention_mask'].repeat_interleave(num_beams,dim=0), decoder_input_ids=pred).logits[:,-1,0:]
    #             input_batch_size=model_kwargs['input_ids'].shape[0]
    #             pred = pred.view(input_batch_size,num_beams,-1)
    #             pred_scores = pred_scores.view(input_batch_size,num_beams)
    #             last_token_logits = last_token_logits.view(input_batch_size,num_beams,-1)
    #             last_token_logits = torch.log_softmax(last_token_logits,dim=-1)[:,:,:self.tgt_voc_size-2]
    #             pred_scores = pred_scores.view(input_batch_size,num_beams,1)+last_token_logits
    #             pred_last_token = (torch.arange(0,self.tgt_voc_size-2)).repeat(input_batch_size*num_beams).to(self.device)
    #             pred = torch.cat([pred.repeat_interleave(self.tgt_voc_size-2,dim=1),pred_last_token.view(input_batch_size,-1,1)],dim=-1)
    #             pred_scores = pred_scores.view(input_batch_size,-1)
    #             index = pred_scores.argsort(dim=-1,descending=True)[:,:topk].unsqueeze(-1).expand(-1,-1,self.max_tgt_len)
    #             pred = pred.gather(dim=1,index=index).view(input_batch_size*topk,-1)
    #         all_pred.append(pred.cpu())
    #     all_pred = torch.cat(all_pred,dim=0)
    #     all_pred = all_pred.view(batch_x.shape[0],topk,self.max_tgt_len)[:,:,1:]
    #     label = self.decode(all_pred)# [batch_size,topk*tree_num]
    #     self.trm_model.train()
    #     return label

    def decode(self,all_pred, type=0):
        """
        all pred: [batch_size,topk,self.max_tgt_len-1], eliminate the starting symbol
        translate Decimal into tree_num-ary, i.e. find the result on each tree

        return [batch_size,topk*tree_num]
        """
        all_pred = all_pred
        # base_code = torch.tensor([0,256]).cuda()
        # all_pred -= base_code
        # all_pred[all_pred < 0] = 0
        # if type == 1:
        #     all_pred -= base_code
        batch_size,topk,max_len=all_pred.shape[0],all_pred.shape[1],all_pred.shape[-1]
        #start_position=(torch.log(all_pred+1.0)/torch.log(tree_num)).ceil()-1
        return self.tree.path_to_label(all_pred).view(batch_size,topk)
        

        

