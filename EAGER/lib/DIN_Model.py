import torch.nn as nn
import torch

class DeepInterestNetwork(nn.Module):
    def __init__(self, item_num=100,embedding_dim=96,
        feature_groups=[20,20,10,10,2,2,2,1,1,1],
        sum_pooling=False,
        ):
        super().__init__()
        self.item_num=item_num
        self.embed_dim=embedding_dim
        self.feature_num=sum(feature_groups)
        self.sum_pooling=sum_pooling
        self.item_embedding=\
            EmbeddingLayer(item_num,96)
            # nn.Linear(768, 256),
            # nn.ReLU(inplace=True),
            # nn.Linear(256, embedding_dim))

        self.attention_unit=LocalActivationUnit(hidden_size=[64, 16], bias=[True, True], \
                                            embedding_dim=embedding_dim, batch_norm=False)
        if sum_pooling:    
            self.fc_layer = FullyConnectedLayer(input_size=2 *embedding_dim,
                                    hidden_size=[200, 80, 1],
                                    bias=[True, True, True],
                                    activation='dice',
                                    sigmoid=False)    
        else:                               
            self.fc_layer = FullyConnectedLayer(input_size=(len(feature_groups)+1) *embedding_dim,
                                                hidden_size=[200, 80, 1],
                                                bias=[True, True, True],
                                                activation='dice',
                                                sigmoid=False)

            #window matrix for each window's weight sum
            window_matrix = torch.zeros(len(feature_groups),sum(feature_groups))
            start_index = 0
            for i, feature in enumerate(feature_groups):
                window_matrix[i, start_index:start_index + feature] = 1.0
                start_index += feature
            self.register_buffer('window_matrix',window_matrix)



    def forward(self, batch_user,batch_label):
        """
        item_num fill with the absence in batch_user
        """
        batch_size,f_num=batch_user.shape
        effective_index=batch_user<self.item_num
        effective_labels=batch_label.expand(batch_user.shape)[effective_index]
        if self.sum_pooling:
            weight_matrix=torch.zeros((batch_user.shape),device=batch_user.device,dtype=torch.float32)
            weight_matrix[effective_index]=\
                self.attention_unit(self.item_embedding(batch_user[effective_index]),\
                    self.item_embedding(effective_labels)).view(-1)
            return self.fc_layer(
                torch.cat((
                    torch.matmul(weight_matrix.view(batch_size,1,f_num),self.item_embedding(batch_user)),\
                                    self.item_embedding(batch_label.view(-1))),dim=1))
        else:
            pre_linear_part_inputs = \
                torch.zeros((batch_size * f_num, self.embed_dim),device=batch_user.device,dtype=torch.float32)
            pre_linear_part_inputs[effective_index.view(-1)]=\
                self.item_embedding(batch_user[effective_index])*\
                    self.attention_unit(self.item_embedding(batch_user[effective_index]),self.item_embedding(effective_labels))
            a = self.window_matrix.size()
            b = pre_linear_part_inputs.view(batch_size,f_num,self.embed_dim).size()

            return self.fc_layer(torch.cat(\
                        (torch.matmul(self.window_matrix,pre_linear_part_inputs.view(batch_size,f_num,self.embed_dim)).view(batch_size,-1),\
                             self.item_embedding(batch_label.view(-1))), -1))







class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, hidden_size, bias, batch_norm=True,\
         dropout_rate=0.1, activation='relu', sigmoid=False, dice_dim=2):
        super(FullyConnectedLayer, self).__init__()
        assert len(hidden_size) >= 1 and len(bias) >= 1
        assert len(bias) == len(hidden_size)
        self.sigmoid = sigmoid

        layers = []
        layers.append(nn.Linear(input_size, hidden_size[0], bias=bias[0]))
        
        for i, h in enumerate(hidden_size[:-1]):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size[i]))
            
            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'dice':
                assert dice_dim
                layers.append(Dice(hidden_size[i], dim=dice_dim))
            elif activation.lower() == 'prelu':
                layers.append(nn.PReLU())
            else:
                raise NotImplementedError
            
            layers.append(nn.Dropout(p=dropout_rate))
            layers.append(nn.Linear(hidden_size[i], hidden_size[i+1], bias=bias[i]))
        
        self.fc = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()
        
        # weight initialization xavier_normal (or glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x) 

        

class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_size=[80, 40], bias=[True, True], embedding_dim=4, batch_norm=False):
        super(LocalActivationUnit, self).__init__()
        self.fc1 = FullyConnectedLayer(input_size=4*embedding_dim,
                                       hidden_size=hidden_size,
                                       bias=bias,
                                       batch_norm=batch_norm,
                                       activation='dice',
                                       dice_dim=2)

        self.fc2 = FullyConnectedLayer(input_size=hidden_size[-1],
                                       hidden_size=[1],
                                       bias=[True],
                                       batch_norm=batch_norm,
                                       activation='dice',
                                       dice_dim=2)
        # TODO: fc_2 initialization

    def forward(self,user_behavior, queries):
        attention_output = self.fc2(self.fc1( 
            torch.cat([queries, user_behavior, queries-user_behavior, queries*user_behavior], dim=-1)))
        return attention_output


class Dice(nn.Module):
    def __init__(self, num_features, dim=2):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3
        self.bn = nn.BatchNorm1d(num_features, eps=1e-9)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim
        
        if self.dim == 3:
            #self.alpha = torch.zeros((num_features, 1)).cuda()
            self.alpha=nn.Parameter(torch.rand((num_features, 1)))
        elif self.dim == 2:
            #self.alpha = torch.zeros((num_features,)).cuda()
            self.alpha=nn.Parameter(torch.rand((num_features,)))
    def forward(self, x):
        if self.dim == 3:
            # x is [batch_size,time_seq_len,hidden_size]
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)
        
        elif self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        
        return out

class EmbeddingLayer(nn.Module):
    def __init__(self, item_num, embedding_dim):
        super(EmbeddingLayer,self).__init__()

        self.embed = nn.Embedding(item_num+1, embedding_dim, padding_idx=item_num)
        
        # normal weight initialization
        self.embed.weight.data.normal_(0., 0.0001)
        # TODO: regularization

    def forward(self, x):
        return self.embed(x)



