import json
import numpy as np
import torch
from os import listdir 
from vocab import Vocab
from nltk.tokenize import word_tokenize, sent_tokenize
import os
import pickle
from tqdm import tqdm
import copy
import joblib
from joblib import Parallel, delayed
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.corpora import Dictionary
from nltk.parse.corenlp import CoreNLPDependencyParser


glove_dir = '/srv/home/wbx/datablaze3/data/glove/glove.6B.100d.txt.word2vec'  # /glove.6B.100d.txt.word2vec
glove_file = get_tmpfile(glove_dir)
model = KeyedVectors.load_word2vec_format(glove_file)

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


dataset = [datas[x] for x in dataset_choice]

trn_set = dataset[:500000]
val_set = dataset[500000:502000]
tst_set = dataset[502000:]

tst = tst_set
trn = trn_set
val = val_set

dct = Dictionary([[]])  # initialize a Dictionary
for data in tqdm(tst):
    dct.add_documents([[word.lower() for sent in sent_tokenize(
                data['text']) for word in word_tokenize(sent)]])
for data in tqdm(val):
    dct.add_documents([[word.lower() for sent in sent_tokenize(
                data['text']) for word in word_tokenize(sent)]])
for data in tqdm(trn):
    dct.add_documents([[word.lower() for sent in sent_tokenize(
                data['text']) for word in word_tokenize(sent)]])


print(len(dct.token2id))
dct_bak = copy.deepcopy(dct)
dct.filter_extremes(no_below=3, no_above=100)
print(len(dct.token2id))


words = []
for word in dct.token2id.keys():
    if word in model.wv:
        words += word,
print(len(words))

dst_path = 'full_vocab.txt'

with open(dst_path, 'w') as f:
    for w in words:
        f.write(w + '\n')

import torch
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
EOS_WORD = '<eos>'
SOS_WORD = '<sos>'
vocab = Vocab(filename=dst_path, data=[PAD_WORD, UNK_WORD, EOS_WORD, SOS_WORD])
vocab.lower = True


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


def preprocess(datas):
    for data in tqdm(datas):
        data['label'] = int(data['stars'] - 1)
        data['raw_text'] = data['text']
        data['text'] = vocab.convertToIdx([word.lower() for sent in sent_tokenize(
                data['raw_text']) for word in word_tokenize(sent)], UNK_WORD)
        data['split_text'] = [vocab.convertToIdx([word for word in word_tokenize(sent)], UNK_WORD) for sent in sent_tokenize(data['raw_text'])]
    return datas


tst = preprocess(tst)
# sanity check
print(vocab.convertToLabels(tst[0]['text'], -1))
print(vocab.convertToLabels(tst[0]['split_text'][0], -1))
print(tst[0]['raw_text'])

val = preprocess(val)
trn = preprocess(trn)

joblib.dump(trn, 'full-trn-processed.pkl')
joblib.dump(val, 'full-val-processed.pkl')
joblib.dump(tst, 'full-tst-processed.pkl')

trn_trees = multi_thread_get_trees(trn)
tst_trees = multi_thread_get_trees(tst)
val_trees = multi_thread_get_trees(val)

joblib.dump(trn_trees, 'full-trn-trees.pkl')
joblib.dump(val_trees, 'full-val-trees.pkl')
joblib.dump(tst_trees, 'full-tst-trees.pkl')

