import torch
import torch.nn as nn
import math
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel, BertModel, BertForMaskedLM
import numpy as np

from lib.transformer.decoder import TransformerDecoder
from lib.transformer.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding

import torch.nn.functional as F
class HFTransformerModel(nn.Module):
    def __init__(self,
                 src_voc_size = 1000,#item_num+1
                 tgt_voc_size = 4,
                 max_src_len = 69,
                 max_tgt_len = 15,
                 d_model = 24,
                 d_model2 = 96,
                 nhead = 4,
                 device = 'cuda',
                 enc_num_layers = 6,
                 dec_num_layers = 6,
                 intermediate_size=1024,
                 position_embedding_type='absolute',
                 item_num = 1,
                 ):
        super(HFTransformerModel, self).__init__()
        self.src_voc_size = src_voc_size
        # tgt_voc_size *= 2
        self.item_num = item_num
        self.tgt_voc_size = tgt_voc_size
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.device = device
        self.src_pad = src_voc_size - 1
        self.tgt_pad = tgt_voc_size - 1
        self.tgt_cls = tgt_voc_size


        config_encoder = BertConfig(vocab_size=src_voc_size+2, hidden_size=d_model, num_hidden_layers=enc_num_layers, num_attention_heads=nhead,
                            intermediate_size=intermediate_size, pad_token_id=self.src_pad,position_embedding_type=position_embedding_type,max_position_embeddings=max_src_len+1)
        config_decoder = BertConfig(vocab_size=tgt_voc_size+1, hidden_size=d_model, num_hidden_layers=dec_num_layers, num_attention_heads=nhead,
                                    intermediate_size=intermediate_size, pad_token_id=self.tgt_pad, position_embedding_type=position_embedding_type,max_position_embeddings=max_tgt_len+1+1)
        # config_decoder.is_decoder = False # True
        config_decoder.is_decoder = True
        config_decoder.add_cross_attention = True
        config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
        self.trm = EncoderDecoderModel(config=config)

        self.trm.config.decoder_start_token_id = tgt_voc_size - 2
        self.trm.config.pad_token_id = self.tgt_pad

        self.mask_decoder = TransformerDecoder(num_layers=4, d_model=96, num_heads=4)
        # self.trm.config.decoder_start_token_id = tgt_voc_size - 2
        # self.decoder_behav_start_token_id = tgt_voc_size - 4
        self.encoder_start_token_id = src_voc_size + 1
        self.decoder_start_token_id = tgt_voc_size - 1
        # self.decoder_cls = tgt_voc_size - 2
        # self.decoder_mask = tgt_voc_size - 3


        self.loss_fct = nn.CrossEntropyLoss()

        self.d_model2 = d_model2

        self.fc_proj1 = nn.Linear(d_model, d_model2)

        self.word_pos_encoder = SinusoidalPositionalEmbedding(96, 0, 10)

        self.guide_proj = nn.Linear(768,96)
        self.trans_d_rec = TransformerDecoder(num_layers=1,  d_model=96, num_heads=4)
        self.fc_comp = nn.Linear(96, 1)

        from torch.autograd import Variable

        self.start_vec = nn.Parameter(torch.zeros(96, ), requires_grad=True)  # .cuda()
        # # self.mask_vec = torch.zeros(96).float()#.requires_grad_(True)
        self.mask_vec = nn.Parameter(torch.zeros(96, ), requires_grad=True)  # .cuda()

        end_idx = int(math.sqrt(self.tgt_voc_size))
        self.end_idx = end_idx



    def forward(self, batch_x, seq_x, batch_y, data_emb=None, orig_y=None, type=0, use_con=False, use_guide=False, guide_feat=None, src_decoder_emb=None, src_y=None):
        """
        batch_x: [batch_size,seq_lenght]
        """
        x, y = batch_x,batch_y

        # y[:, -1] = orig_y + 256
        start_token = self.trm.config.decoder_start_token_id
        prompt_y = y + 1 - 1
        prompt_y[:, -1] = prompt_y[:, 0] * self.end_idx + prompt_y[:, -1] + self.end_idx
        prompt_y = torch.cat([torch.tensor(start_token).cuda().view(1, 1).expand(y.size(0), -1), prompt_y, torch.tensor(self.tgt_cls).cuda().view(1, 1).expand(y.size(0), -1)], dim=-1)

        src_key_padding_mask = _generate_pad_mask(x, self.src_pad).to(self.device)
        # tgt_key_padding_mask = _generate_pad_mask(y, self.tgt_pad).to(self.device)

        tgt_key_padding_mask = _generate_pad_mask(prompt_y, self.tgt_pad).to(self.device)
        # z = self.trm.encoder(input_ids=x,
        #                 attention_mask=src_key_padding_mask, # [batch_size,seq_length]
        #                 output_hidden_states=True)

        identifier_x = seq_x[:, :, 0] * self.end_idx + seq_x[:, :, 1] + self.end_idx
        identifier_x = identifier_x.long()
        # input_embeds = self.trm.decoder.get_input_embeddings()(identifier_x.long())
        emb_x = x + 1 - 1
        if type == 1:
            emb_x[emb_x != self.item_num] += (self.item_num + 1)
            # emb_x += self.item_num
        input_embeds = self.trm.encoder.get_input_embeddings()(emb_x.long())

        decoder_embeds = self.trm.decoder.get_input_embeddings()(prompt_y.long())

        z = self.trm(inputs_embeds=input_embeds,
                     attention_mask=src_key_padding_mask,  # [batch_size,seq_length]
                     decoder_inputs_embeds=decoder_embeds,
                     decoder_attention_mask=tgt_key_padding_mask,
                     output_hidden_states=True,
                     )
        # y = y.view(-1)

        # loss1 = self.loss_fct(z.logits[:, 0, :256].contiguous().view(y.size(0), -1), y[:, 0])
        # loss2 = self.loss_fct(z.logits[:, 1, 256:-2].contiguous().view(y.size(0), -1), y[:, 1]-256)
        # z.loss = (loss1 + loss2) / 2
        final_hidden = z.decoder_hidden_states[-1]  # [bsz, l, dim]

        if True or type == 0 or src_decoder_emb is not None:
            dec_emb = self.trm.decoder.get_input_embeddings().weight[:-3]#.view(256, 256, -1)
            dec_emb_l1 = dec_emb[:self.end_idx]
            dec_emb_l2 = dec_emb[self.end_idx:].view(self.end_idx, self.end_idx, -1)[y[:, 0]]
        else:
            cur_dec_emb = self.trm.decoder.get_input_embeddings().weight[:-3]  # .view(256, 256, -1)
            dec_emb_l1 = cur_dec_emb[::self.end_idx]
            dec_emb_l2 = src_decoder_emb.view(self.end_idx, self.end_idx, -1)[src_y[:, 0]]

        logits1 = torch.matmul(final_hidden, dec_emb_l1.unsqueeze(0).transpose(1, 2))
        logits2 = torch.matmul(final_hidden, dec_emb_l2.transpose(1, 2))
        loss1 = self.loss_fct(logits1[:, 0, :].contiguous().view(y.size(0), -1), y[:, 0])
        loss2 = self.loss_fct(logits2[:, 1, :].contiguous().view(y.size(0), -1), y[:, 1])
        z.loss = (loss1 + loss2) / 2
        global_feat = z.decoder_hidden_states[-1][:, -1] # [bsz, 96]
        # global_feat2 = z.decoder_hidden_states[-2][:, -1]
        # global_feat3 = z.decoder_hidden_states[-3][:, -1]
        local_feat = z.decoder_hidden_states[-1][:, :]
        local_feat_ = local_feat + 1 - 1
        global_feat_ = global_feat + 1 - 1
        with torch.no_grad():
            data_emb = torch.tensor(data_emb).cuda()
            sample_id = torch.randperm(len(data_emb)).cuda()
            sample_emb = torch.index_select(input=data_emb, index=sample_id, dim=0)[:2048] # sample 2048 items

            positive_emb = torch.index_select(input=data_emb, index=orig_y, dim=0)

        global_feat = F.normalize(global_feat, dim=-1)
        global_feat = self.fc_proj1(global_feat)
        sample_emb = sample_emb.float()
        positive_emb = positive_emb.float()

        # l1 distillation
        # contra_loss = F.l1_loss(global_feat, positive_emb.float())
        # global_feat = F.normalize(global_feat, dim=-1)
        # positive_emb = F.normalize(positive_emb.float(), dim=-1)

        # contra_loss = -(global_feat * positive_emb).sum(dim=-1).mean()
        contra_loss = torch.tensor(0.0).cuda()
        if use_con:
            contra_loss = info_nce(global_feat, positive_emb, sample_emb.view(1, 2048, -1).expand(len(global_feat), -1, -1), temperature=0.3, negative_mode='paired')
            if type == 0:
                z.loss += 1.0 * contra_loss
            else:
                z.loss += 10.0 * contra_loss # larger loss for modal emb
        #
        recon_loss = torch.tensor(0.0).cuda()
        est_loss = torch.tensor(0.0).cuda()
        if use_guide and type == 0 and guide_feat is not None:
            # content-guided reconstruction
            decoder_emb = self.trm.decoder.get_input_embeddings()
            input_feat = decoder_emb(y)
            input_len = torch.ones([len(input_feat)]).cuda() * 2

            guide_feat = guide_feat.detach()
            guide_feat = self.guide_proj(guide_feat).unsqueeze(1)

            bsz, l, dim = input_feat.size()
            x = torch.zeros([bsz, l + 1, dim]).cuda()
            x[:, 0, :] = self.start_vec.clone()  # .cuda()
            x[:, 1:] = input_feat
            words_feat = x
            words_pos = self.word_pos_encoder(words_feat)
            # words_feat, masked_pos, masked_flag = self._mask_words(words_feat, input_len) #+ words_pos
            import random
            first_col = True
            if random.random() > 0.5:
                first_col = False
            mask_token = self.mask_vec.cuda().unsqueeze(0).unsqueeze(0)
            masked_pos = torch.zeros(words_feat.size(0), words_feat.size(1)).cuda()
            if first_col:
                masked_pos[:, 1] = 1
                words_feat[:, 1] = mask_token
            else:
                masked_pos[:, 2] = 1
                words_feat[:, 2] = mask_token
            words_feat += words_pos
            # masked_pos = masked_pos.squeeze(-1)

            h = self.trans_d_rec(guide_feat, None, words_feat, None)
            h = h.masked_select(masked_pos.unsqueeze(-1) == 1).view(-1, 96)
            # logit = self.fc_comp(h)[:,:]
            # batch_y = batch_y.view(-1)
            # recon_loss = F.cross_entropy(logit.contiguous().view(len(batch_y), -1), batch_y.view(-1), reduction='none')
            # recon_loss = recon_loss.view(-1, 2).masked_fill(masked_pos == 0, 0).sum() / masked_pos.sum()

            decoder_emb = self.trm.decoder.get_input_embeddings().weight[:-3]
            if first_col:
                pos_id = y[:, 0]#.masked_select(masked_pos[:, 1:] == 1)
                pos_emb = decoder_emb[pos_id].detach()
                neg_emb = decoder_emb.detach()[:self.end_idx].view(1, -1, 96).expand(len(pos_emb), -1, -1)
                recon_loss = info_nce(h, pos_emb, neg_emb, temperature=1.0, negative_mode='paired', norm=False)
            else:
                pos_id = y[:, 0] * self.end_idx + y[:, 1] + self.end_idx
                pos_emb = decoder_emb[pos_id].detach()
                neg_emb = decoder_emb.detach()[self.end_idx:].view(self.end_idx, self.end_idx, -1)[y[:, 0]]
                recon_loss = info_nce(h, pos_emb, neg_emb, temperature=1.0, negative_mode='paired', norm=False)
            # neg_emb1 = decoder_emb.weight.detach()[:256]
            # # neg_emb2 = decoder_emb.weight.detach()[256:-5]
            # neg_emb2 = decoder_emb.view(self.end_idx, self.end_idx, -1)[y[:, 0]]
            # neg_emb = []
            # for flag in masked_flag:
            #     if flag == 1:
            #         neg_emb.append(neg_emb1.unsqueeze(0))
            #     else:
            #         neg_emb.append(neg_emb2.unsqueeze(0))
            # neg_emb = torch.cat(neg_emb, dim=0).cuda()
            # recon_loss = info_nce(h, pos_emb, neg_emb, temperature=1.0, negative_mode='paired', norm=False)
            z.loss += 1 * recon_loss

            # # content-guided estimation
            import random
            first_col = True
            if random.random() > 0.5:
                first_col = False
            pos_words_feat = x

            if first_col:
                neg_id = np.random.randint(0, self.end_idx, (pos_words_feat.size(0), pos_words_feat.size(1)))
            else:
                neg_id = np.random.randint(self.end_idx, self.end_idx ** 2 + self.end_idx, (pos_words_feat.size(0), pos_words_feat.size(1)))
            neg_id = torch.from_numpy(neg_id).cuda()
            neg_vec = decoder_emb[neg_id]
            neg_words_feat, neg_masked_pos = self._neg_words(pos_words_feat, input_len, first_col, neg_vec)  # + words_pos

            pos_words_feat += words_pos
            neg_words_feat += words_pos

            pos_h = self.trans_d_rec(guide_feat, None, pos_words_feat, None)
            neg_h = self.trans_d_rec(guide_feat, None, neg_words_feat, None)
            pos_logit = self.fc_comp(pos_h)[:, 0]
            neg_logit = self.fc_comp(neg_h)[:, 0]
            logit = torch.cat([pos_logit, neg_logit], dim=0)
            label = torch.cat([torch.ones(len(pos_logit)), torch.zeros(len(neg_logit))], dim=0).view(-1, 1)
            est_loss = F.binary_cross_entropy_with_logits(logit, label.cuda())
            z.loss += 1.0 * est_loss

        return z, recon_loss, global_feat #local_feat_
        # return self.trm(input_ids=x,
        #                 attention_mask=src_key_padding_mask,#[batch_size,seq_length]
        #                 labels=y,
        #                 decoder_attention_mask=tgt_key_padding_mask)
        #return output

    #def generate(self, num_beams=40, topk=24, max_length=7, **kwargs):
    def _mask_words(self, words_feat, words_len, weights=None):
        token = self.mask_vec.cuda().unsqueeze(0).unsqueeze(0)
        # token, _ = self.query_encoder(token, None, None)

        masked_words = []
        masked_flag = []
        for i, l in enumerate(words_len):
            l = int(l)
            # l = min(l, len(p))
            # l = min(l, weights.size(1))
            num_masked_words = max(l // 2, 1)
            masked_words.append(torch.zeros([words_feat.size(1)]).byte().cuda())
            choices = np.random.choice(np.arange(1, l+1), num_masked_words, replace=False)
            if choices == 2:
                masked_flag.append(torch.tensor(2))
            else:
                masked_flag.append(torch.tensor(1))
            masked_words[-1][choices] = 1
        # exit(0)
        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        masked_flag = torch.stack(masked_flag, dim=0).cuda()
        return words_feat1, masked_words, masked_flag

    def _neg_words(self, words_feat, words_len,first_col, neg_vec):
        token = neg_vec
        # token, _ = self.query_encoder(token, None, None)

        masked_words = []
        for i, l in enumerate(words_len):
            num_masked_words = 1
            masked_words.append(torch.zeros([words_feat.size(1)]).byte().cuda())
            if first_col:
                choices = np.random.choice(np.arange(1, 2), num_masked_words, replace=False)
            else:
                choices = np.random.choice(np.arange(2, 3), num_masked_words, replace=False)
            masked_words[-1][choices] = 1
        # exit(0)
        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        return words_feat1, masked_words

        



def _generate_pad_mask(x, pad):
    #mask=(x != pad).float()
    #mask=torch.where(x!=pad,torch.ones(x.shape),torch.zeros(x.shape),device=device,dtype=torch.float32)
    mask = torch.FloatTensor((x != pad).cpu().numpy())
    return mask



def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired',mask=None, norm=True):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    if norm:
        query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys) # [bsz, neg_num]

        elif negative_mode == 'paired':
            query = query.unsqueeze(1) # [bsz, 1, dim]
            negative_logits = query @ transpose(negative_keys) # [bsz, 1, dim] * [bsz, dim, neg_num]
            negative_logits = negative_logits.squeeze(1) # [bsz, neg_num]

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    #a = F.cross_entropy(logits / temperature, labels, reduction=reduction)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]



def cal_nll_loss(logit, idx, mask):
    eps = 0.1
    logit = logit.log_softmax(dim=-1)
    # idx:[48, 25] logit:[48, 25, 11743]
    nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)  # [nb * nw, seq_len]
    smooth_loss = -logit.sum(dim=-1)  # [nb * nw, seq_len]
    nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss

    nll_loss = nll_loss.masked_fill(mask == 0, 0)
    nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)

    return nll_loss.contiguous()


