
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class WordEmbed(nn.Module):
    
    def __init__(self, args, is_train_embd=False):
        super(WordEmbed, self).__init__()
        self.embedding = nn.Embedding(args.vocab_size, 100)
        self.embedding.weight = nn.Parameter(args.glove_embd, requires_grad=is_train_embd)

    def forward(self, x):
        return self.embedding(x)
    


# In[ ]:

class BiDAF_PrNet(nn.Module):
    
    def __init__(self, args):
        super(BiDAF_PrNet, self).__init__()
        self.input_size = 100
        self.usePointer = args.pointer
        if self.usePointer:
            self.answers = 2
            self.pointer_embd_size = 2*self.input_size
            self.hidden_size = 4*self.input_size
            self.pointer_weight_size = 2*self.input_size
        
        self.word_embd_layer = WordEmbed(args)
        self.context_layer = nn.GRU(self.input_size, self.input_size, bidirectional=True, dropout=0.15, batch_first=True)
        self.W = nn.Linear(6*self.input_size, 1, bias=False)
        self.modeling_layer = nn.GRU(8*self.input_size, self.input_size, num_layers=2, bidirectional=True, dropout=0.15, batch_first=True)
        if self.usePointer:
            self.encode_layer = nn.GRU(self.pointer_embd_size, self.hidden_size, batch_first=True)
            self.decode_layer = nn.GRUCell(self.pointer_embd_size, self.hidden_size)
            self.W1 = nn.Linear(self.hidden_size, self.pointer_weight_size, bias=False) 
            self.W2 = nn.Linear(self.hidden_size, self.pointer_weight_size, bias=False) 
            self.vt = nn.Linear(self.pointer_weight_size, 1, bias=False)

    def to_variable(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)
    
    def build_word_embd(self, x_context):
        word_embd = self.word_embd_layer(x_context)
        output, _vector = self.context_layer(word_embd)
        return output
    
    def forward(self, x_word, x_query):
        batch_size = x_word.size(0)
        T = x_word.size(1)
        L = x_query.size(1)
        #print("T", T)
        #print("L", L)
        embd_context = self.build_word_embd(x_word)
        embd_query = self.build_word_embd(x_query)
        #Attention Layer
        shape = (batch_size, T, L, 2*self.input_size)
        embd_context_expend = embd_context.unsqueeze(2).expand(shape) 
        embd_query_expend = embd_query.unsqueeze(1).expand(shape)            
        State = self.W(torch.cat((embd_context_expend, embd_query_expend, torch.mul(embd_context_expend, embd_query_expend) ), 3)).view(batch_size, T, L)
        c2q = torch.bmm(F.softmax(State, dim=-1), embd_query)  
        q2c = torch.bmm(F.softmax(torch.max(State, 2)[0], dim=-1).unsqueeze(1), embd_context).repeat(1, T, 1) 

        H = torch.cat((embd_context, c2q, embd_context.mul(c2q), embd_context.mul(q2c)), 2)
        M, _vector = self.modeling_layer(H) 
        #Output Layer:
        if self.usePointer:
            encoder_states, _vector = self.encode_layer(M) 
            encoder_states = encoder_states.transpose(1, 0) 
            decoder_input = self.to_variable(torch.zeros(batch_size, self.pointer_embd_size))
            hidden = self.to_variable(torch.zeros([batch_size, self.hidden_size]))
            cell_state = encoder_states[-1]                            
            probs = []
            for i in range(self.answers):
                hidden = self.decode_layer(decoder_input, hidden)
                blend_sum = F.tanh(self.W1(encoder_states) + self.W2(hidden))
                out = self.vt(blend_sum).squeeze()        
                out = F.log_softmax(out.t().contiguous()) 
                probs.append(out)
            probs = torch.stack(probs, dim=1) 
        return probs        

        
        

