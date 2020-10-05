#!/usr/bin/env python
# coding: utf-8

# In[6]:


import json
import numpy as np


# In[1]:


yelp_size = 6685900
np.random.seed(233)
dataset_choice = np.random.choice(yelp_size, 500000+2000+2000, replace=False)

datas = []
i = 0
with open('review.json', 'r') as f:
    tot = 0
    review = f.readline()
    while review:
        data = json.loads(review)
        datas.append(data)
        tot += 1
        review = f.readline()


# In[22]:


dataset = [datas[x] for x in dataset_choice]



# In[25]:


trn_set = dataset[:500000]
val_set = dataset[500000:502000]
tst_set = dataset[502000:]


# In[27]:


import numpy as np
from os import listdir 
from vocab import Vocab
from nltk.tokenize import word_tokenize, sent_tokenize
import os
import pickle
from tqdm import tqdm_notebook as tqdm


# In[29]:


from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
glove_file = get_tmpfile('/srv/home/wbx/datablaze3/data/glove/glove.6B.100d.txt.word2vec')
model = KeyedVectors.load_word2vec_format(glove_file)


from gensim.corpora import Dictionary
dct = Dictionary([[]])  # initialize a Dictionary
for data in tqdm(tst):
    dct.add_documents([vocab.convertToLabels(data['text'], -1)])
for data in tqdm(val):
    dct.add_documents([vocab.convertToLabels(data['text'], -1)])
for data in tqdm(trn):
    dct.add_documents([vocab.convertToLabels(data['text'], -1)])



import copy
dct1 = copy.deepcopy(dct)
dct1.filter_extremes(no_below=3)
len(dct1.token2id)


# In[101]:


words = list(dct1.token2id.keys())
dst_path = 'new_vocab.txt'

with open(dst_path, 'w') as f:
    for w in words:
        f.write(w + '\n')


# In[102]:


PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
EOS_WORD = '<eos>'
SOS_WORD = '<sos>'
vocab = Vocab(filename=dst_path, data=[PAD_WORD, UNK_WORD, EOS_WORD, SOS_WORD])


# In[103]:


emb_fn = 'new_emb.pth'
embedding_size = 100
embedding = torch.zeros(vocab.size(), 100, dtype=torch.float)
embedding[vocab.getIndex(UNK_WORD)].fill_(1e-2)
embedding[vocab.getIndex(PAD_WORD)].fill_(1e-3)
embedding[vocab.getIndex(EOS_WORD)].fill_(1e-4)
embedding[vocab.getIndex(SOS_WORD)].fill_(1e-5)
for word, index in vocab.labelToIdx.items():
    try:
        embedding[index] = torch.FloatTensor(model[word])
    except:
        print(word)
torch.save(embedding, emb_fn)


# In[104]:


embedding.size()


# In[107]:


def preprocess(datas):
    datas = joblib.load(datas)
    for data in tqdm(datas):
        data['text'] = vocab.convertToIdx([word.lower() for sent in sent_tokenize(
                data['raw_text']) for word in word_tokenize(sent)], UNK_WORD)
        data['split_text'] = [vocab.convertToIdx([word for word in word_tokenize(sent)], UNK_WORD) for sent in sent_tokenize(data['raw_text'].lower())]
        data['label'] = int(data['stars'] - 1)
    return datas



from nltk.parse.corenlp import CoreNLPDependencyParser

def parse_sentence_without_compare(sentence):
    """
    :param sentence: str
    :return: triples, words
    """
    parser = CoreNLPDependencyParser(url='http://localhost:9000')

    parse = parser.raw_parse(sentence)

    parse_sents = list(parse)
    assert len(parse_sents) == 1, "More than 1 sentence extracted"
    parse_graph = list(parse_sents[0].nodes.values())

    parse_graph.sort(key=lambda x: x["address"])
    parse_graph = parse_graph[1:]
    words = [x["word"] for x in parse_graph]

    triples = []
    for k in parse_graph:
        if k["head"] is None:
            continue
        elif k["head"] == 0:
            triples.append((("ROOT", k["head"] - 1), (words[k["address"] - 1], k["address"] - 1), k["rel"]))
        else:
            triples.append(
                ((words[k["head"] - 1], k["head"] - 1), (words[k["address"] - 1], k["address"] - 1),
                 k["rel"]))

    return triples, words



# In[129]:


def get_trees(datas):
    trees = []
    for i, data in enumerate(tqdm(datas)):
        sentences = sent_tokenize(data['raw_text'])
        paragraph_trees = {"tree": [], "words":[]}
        for sentence in sentences:
            try:
                triples, words = parse_sentence_without_compare(sentence)
            except:
                words = word_tokenize(sentence)
                triples = None
            words = vocab.convertToIdx(words, UNK_WORD)
            paragraph_trees["tree"].append(triples)
            paragraph_trees["words"].append(words)
        trees.append(paragraph_trees)
    return trees


# In[144]:


from joblib import Parallel, delayed
vocab.lower = True

def multi_thread_get_trees(datas):
    def get_tree(data):
        sentences = sent_tokenize(data['raw_text'])
        paragraph_trees = {"tree": [], "words":[]}
        for sentence in sentences:
            try:
                triples, words = parse_sentence_without_compare(sentence)
            except:
                words = word_tokenize(sentence)
                triples = None
            words = vocab.convertToIdx(words, UNK_WORD)
            paragraph_trees["tree"].append(triples)
            paragraph_trees["words"].append(words)
        return paragraph_trees
    
    return Parallel(n_jobs=24, verbose=1)(delayed(get_tree)(data) for data in datas)



from gensim.corpora import Dictionary
dct = Dictionary([[]])  # initialize a Dictionary
for data in tqdm(tst):
    dct.add_documents([[word.lower() for sent in sent_tokenize(
                data['raw_text']) for word in word_tokenize(sent)]])
for data in tqdm(val):
    dct.add_documents([[word.lower() for sent in sent_tokenize(
                data['raw_text']) for word in word_tokenize(sent)]])
for data in tqdm(trn):
    dct.add_documents([[word.lower() for sent in sent_tokenize(
                data['raw_text']) for word in word_tokenize(sent)]])


# In[191]:


print(len(dct.token2id))
dct_bak = copy.deepcopy(dct)
dct.filter_extremes(no_below=3, no_above=100)
print(len(dct.token2id))


# In[199]:


words = []
for word in dct.token2id.keys():
    if word in model.wv:
        words += word,
print(len(words))


# In[200]:


dst_path = 'full_vocab.txt'

with open(dst_path, 'w') as f:
    for w in words:
        f.write(w + '\n')


# In[201]:


PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
EOS_WORD = '<eos>'
SOS_WORD = '<sos>'
vocab = Vocab(filename=dst_path, data=[PAD_WORD, UNK_WORD, EOS_WORD, SOS_WORD])
vocab.lower = True


# In[202]:


emb_fn = 'full_emb.pth'
embedding_size = 100
embedding = torch.zeros(vocab.size(), 100, dtype=torch.float)
embedding[vocab.getIndex(UNK_WORD)].fill_(1e-2)
embedding[vocab.getIndex(PAD_WORD)].fill_(1e-3)
embedding[vocab.getIndex(EOS_WORD)].fill_(1e-4)
embedding[vocab.getIndex(SOS_WORD)].fill_(1e-5)
for word, index in vocab.labelToIdx.items():
    try:
        embedding[index] = torch.FloatTensor(model[word])
    except:
        print(word)
torch.save(embedding, emb_fn)


# In[203]:


def preprocess(datas):
    for data in tqdm(datas):
        data['text'] = vocab.convertToIdx([word.lower() for sent in sent_tokenize(
                data['raw_text']) for word in word_tokenize(sent)], UNK_WORD)
        data['split_text'] = [vocab.convertToIdx([word for word in word_tokenize(sent)], UNK_WORD) for sent in sent_tokenize(data['raw_text'])]
    return datas


# In[211]:


tst = joblib.load('tst-processed.pkl')
vocab.lower = True
tst = preprocess(tst)


# In[213]:


print(vocab.convertToLabels(tst[0]['text'], -1))
print(vocab.convertToLabels(tst[0]['split_text'][0], -1))
print(tst[0]['raw_text'])


# In[214]:


val = preprocess(val)
trn = preprocess(trn)


# In[215]:


import joblib
joblib.dump(trn, 'full-trn-processed.pkl')
joblib.dump(val, 'full-val-processed.pkl')
joblib.dump(tst, 'full-tst-processed.pkl')


# In[216]:


trn_trees = multi_thread_get_trees(trn)
tst_trees = multi_thread_get_trees(tst)
val_trees = multi_thread_get_trees(val)


# In[217]:


import joblib
joblib.dump(trn_trees, 'full-trn-trees.pkl')
joblib.dump(val_trees, 'full-val-trees.pkl')
joblib.dump(tst_trees, 'full-tst-trees.pkl')

