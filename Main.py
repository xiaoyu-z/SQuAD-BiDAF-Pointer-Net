
# coding: utf-8

# In[4]:
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from BiDAF_PointerNet import BiDAF_PrNet

# In[5]:

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--evaluate', type=bool, default=False)
parser.add_argument('--n_epoch', type=int, default = 20)
parser.add_argument('--resume_file', type=str, default='./save/',metavar='PATH')
parser.add_argument('--pointer', type=bool, default=True)
parser.add_argument('--cross_entropy', type=bool, default=True)
args = parser.parse_args()


# In[6]:




# In[ ]:

class MyData():
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.length = len(self.dataset1['q']) 
    #self.batch_size = batch_size
        
        
    def get_batch(self, current, batch_size):
        batches = []
        batch = []
        for i in range(batch_size):
            context  = self.dataset2['x'][self.dataset1['*x'][current+i][0]][self.dataset1['*x'][current+i][1]][0]
            query  = self.dataset1['q'][current+i]
            context = [j.lower() for j in context]
            query = [j.lower() for j in query]
            ans  = (self.dataset1['y'][current+i][0][0][1], self.dataset1['y'][current+i][0][1][1])
            #print("Len:", len(context), len(query), len(ans))
            batch.append((context, query, ans))
        batches.append(batch)
        return batches
                
    def get_word_index(self, word_num=5):

        word2vec_dict = self.dataset2['lower_word2vec']
        word_counter = self.dataset2['lower_word_counter']
        char_counter = self.dataset2['char_counter']
        windex = {w: i for i, w in enumerate(w for w, ct in word_counter.items()
                                            if ct > word_num or (w in word2vec_dict))}

        return windex


# In[ ]:

class Model():
    def __init__(self, args):
        self.read_data()
        args.vocab_size = len(self.word2index)
        args.glove_embd = self.glove_embd
        self.args = args
        self.model = BiDAF_PrNet(args)
        if self.args.cross_entropy:
            self.loss = F.cross_entropy
        else:
            self.loss = F.nll_loss
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()))
        if torch.cuda.is_available():
            self.model.cuda()
        self.current = 0
        self.start_epoch = 0
        self.load_model()
            
    def load_dev_data(self, path='./dataset/'):
        test_json = json.load(open(path+'data_dev.json'))
        test_shared_json = json.load(open(path+'shared_dev.json'))
        self.testdata = MyData(test_json, test_shared_json)
    
    def load_train_data(self, path='./dataset/'):
        test_json = json.load(open(path+'data_train.json'))
        test_shared_json = json.load(open(path+'shared_train.json'))
        self.traindata = MyData(test_json, test_shared_json)
    
    def load_glove(self, path='./dataset/'):
        embeddings_index = {}
        with open(path+'glove.6B.100d.txt') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                embeddings_index[word] = vector

        embedding_matrix = np.zeros((len(self.word2index), 100))
        #print('embed_matrix.shape', embedding_matrix.shape)
        for word, i in self.word2index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
        self.glove_embd = torch.from_numpy(embedding_matrix).type(torch.FloatTensor)

    def read_data(self):

        self.load_train_data()
        self.load_dev_data()
        
        word_index_train = self.traindata.get_word_index()
        word_index_test = self.testdata.get_word_index()
        word_vocabulary = sorted(list(set(list(word_index_train.keys()) + list(word_index_test.keys()))))
        self.word2index = {w : i for i, w in enumerate(word_vocabulary, 3)}
        self.word2index["-NULL-"] = 0
        self.word2index["-UNK-"] = 1
        self.word2index["-ENT-"] = 2
        self.load_glove()
    
    def to_variable(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)
        
    def to_vector(self, batch, context_len, query_len):
        c = []
        q = []
        ans = []
        def _make_word_vector(sentence, w2i, seq_len):
            index_vec = [w2i[w] if w in w2i else w2i["-UNK-"] for w in sentence]
            pad_len = max(0, seq_len - len(index_vec))
            index_vec += [w2i["-NULL-"]] * pad_len
            index_vec = index_vec[:seq_len]
            return index_vec
        for d in batch:
            c.append(_make_word_vector(d[0], self.word2index, context_len))
            q.append(_make_word_vector(d[1], self.word2index, query_len))
            ans.append(d[-1])
        c = self.to_variable(torch.LongTensor(c))
        q = self.to_variable(torch.LongTensor(q))
        a = self.to_variable(torch.LongTensor(ans))
        return c, q, a

    def save_model(self, state, path):
        torch.save(state, path)
    
    def load_model(self):
        if os.path.isfile(self.args.resume_file):
            checkpoint = torch.load(args.resume)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train(self):
    #print('----Train---')
        self.model.train()
        batch_size = self.args.batch_size
        for epoch in range(self.start_epoch, self.args.n_epoch):
            print('Epoch:', epoch)
            for j in range(int(self.traindata.length/batch_size)):
            
                batch = self.traindata.get_batch(self.current, batch_size)[0]
                self.current += batch_size
                #print(batch)
                contex_len = max([len(d[0]) for d in batch])
                #print('contlen',contex_len)
                query_len = max([len(d[1]) for d in batch])
                #print('query_len', query_len)
                context, query, ans = self.to_vector(batch, contex_len, query_len)
                ans_beg = ans[:, 0]
                ans_end = ans[:, 1] - 1
                probs= self.model(context, query)
                T = context.size(1)
                outputs = probs.view(-1, T)
                ans_beg = torch.unsqueeze(ans_beg, 1)
                ans_end = torch.unsqueeze(ans_end, 1)
                Y = torch.cat((ans_beg, ans_end), 1).view(batch_size, 2)
                y_1 = Y.contiguous().view(-1)
                #print('out',outputs)
                #print('y',y_1)
                loss = self.loss(outputs, y_1)
                    #if self.args.cross_entropy:
                    #loss = F.cross_entropy(outputs, y_1)
                    #else:
                    #load = F.nll_loss(outputs, y_1)
                if(j%50) == 0:
                    EM = 0
                    F1 = 0
                    prob, indices = torch.max(probs, 2)
                    for k in range(len(indices)):
                        y = Y[k]
                        pred = indices[k]
                        if torch.equal(y,pred):
                            EM+=1
                            overlap = 0
                            answer_length = y.data[1]-y.data[0] + 1
                            if y.data[0]<pred.data[0] and y.data[1] > pred.data[1]:
                                overlap = (pred.data[1]-pred.data[0]+1)/answer_length
                            elif y.data[0]>=pred.data[0] and y.data[1] <= pred.data[1]:
                                overlap = 1
                            elif pred.data[0]>=y.data[0] and pred.data[0]<=y.data[1] and pred.data[1]>y.data[1]:
                                overlap =  (y.data[1]-pred.data[0]+1)/answer_length
                            elif pred.data[0] < y.data[0] and pred.data[1] <= y.data[1] and pred.data[1] >= y.data[0]:
                                overlap = (pred.data[1] - y.data[0]+1)/answer_length
                    
                            F1 += overlap
                    print('EM: {:.2f}'.format(EM/batch_size))
                    print('F1: {:.2f}'.format(F1/batch_size))
                    print('Current Batch Loss: {:.5f}'.format(loss.data[0]))
            
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                            
            self.save_checkpoint({'epoch': epoch,'state_dict': self.model.state_dict(),'optimizer' : self.optimizer.state_dict(),}, './save/biDAF_PrNet'+str(epoch)+'.model')
        
    

    
    def evaluate(self):
        if not os.path.isfile(self.args.resume_file):
            print('No Model to Evaluates')
        batch_size = self.args.batch_size
            #for epoch in range(self.start_epoch, self.args.n_epoch):
            #print('Epoch:', epoch)
        EM = 0
        F1 = 0
        for j in range(int(self.testdata.length/batch_size)):
            batch = self.testdata.get_batch(self.current, batch_size)[0]
            self.current += batch_size
            contex_len = max([len(d[0]) for d in batch])
            query_len = max([len(d[1]) for d in batch])
            context, query, ans = self.to_vector(batch, contex_len, query_len)
            ans_beg = ans[:, 0]
            ans_end = ans[:, 1] - 1
            probs= self.model(context, query)
            T = context.size(1)
            outputs = probs.view(-1, T)
            ans_beg = torch.unsqueeze(ans_beg, 1)
            ans_end = torch.unsqueeze(ans_end, 1)
            Y = torch.cat((ans_beg, ans_end), 1).view(batch_size, 2)
            #y_1 = Y.contiguous().view(-1)
            #loss = F.cross_entropy(outputs, y_1)
            prob, indices = torch.max(probs, 2)
            for k in range(len(indices)):
                y = Y[k]
                pred = indices[k]
                if torch.equal(y,pred):
                    EM+=1
                overlap = 0
                answer_length = y.data[1]-y.data[0] + 1
                if y.data[0]<pred.data[0] and y.data[1] > pred.data[1]:
                    overlap = (pred.data[1]-pred.data[0]+1)/answer_length
                elif y.data[0]>=pred.data[0] and y.data[1] <= pred.data[1]:
                    overlap = 1
                elif pred.data[0]>=y.data[0] and pred.data[0]<=y.data[1] and pred.data[1]>y.data[1]:
                    overlap =  (y.data[1]-pred.data[0]+1)/answer_length
                elif pred.data[0] < y.data[0] and pred.data[1] <= y.data[1] and pred.data[1] >= y.data[0]:
                    overlap = (pred.data[1] - y.data[0]+1)/answer_length
                F1 += overlap
        print('EM: {:.2f}'.format(EM/self.testdata.length))
        print('F1: {:.2f}'.format(F1/self.testdata.length))


model = Model(args)
if args.evaluate:
    model.evaluate()
else:
    model.train()

        

