import logging
import time

import os

import torch.utils.data as data

import util
from treeLSTM.treeNode import TreeNode
from vocab import Vocab

logger = logging.getLogger(__name__)
info = logger.info


class YelpDataset(data.Dataset):
    def __init__(self, path, vocab, tree_path=None):
        super(YelpDataset, self).__init__()
        self.path = path
        self.tree_path = tree_path
        start = time.time()
        self.data = self.read_data(path)
        print("Load data done: ", time.time() - start, "s")
        self.vocab = vocab

        if self.tree_path is not None:
            start = time.time()
            self.tree_data = self.read_data(tree_path)
            print("Load trees done: ", time.time() - start, "s")
            self.rel_vocab = self.build_rel_vocab()
            self.rel_emb_size = 50
            # self.trees = self.read_trees()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def read_data(path):
        print("start loading dataset")
        import joblib
        return joblib.load(path)


    def build_rel_vocab(self):
        rel_vocab = "rel_vocab.txt"
        if not os.path.isfile(rel_vocab):
            vocab = set()
            for paragraph in self.tree_data:
                for sent in paragraph['tree']:
                    if sent is not None:
                        for rel in sent:
                            vocab.add(rel[2])
            with open(rel_vocab, 'w') as f:
                for w in sorted(vocab):
                    f.write(w + '\n')
        return Vocab(filename=rel_vocab,
                     data=[util.UNK_WORD])

    def get_tree(self, tri_case):
        """
        :param tri_case: one sentence tri case
        :return: root node
        """
        tri_case.sort(key=lambda x: x[1][1])
        Nodes = dict()
        root = None
        for i in range(len(tri_case)):
            # if i not in Nodes.keys() and tri_case[i][0][1] != -1:
            if i not in Nodes.keys():
                idx = i
                prev = None
                rel = None
                while True:
                    tree = TreeNode()
                    Nodes[idx] = tree
                    tree.idx = idx
                    if prev is not None:
                        tree.add_child(
                            prev, self.rel_vocab.getIndex(rel, util.UNK_WORD))
                    parent = tri_case[idx][0][1]
                    parent_rel = tri_case[idx][2]
                    if parent in Nodes.keys():
                        Nodes[parent].add_child(
                            tree, self.rel_vocab.getIndex(parent_rel, util.UNK_WORD))
                        break
                    elif parent == -1:
                        root = TreeNode()
                        root.idx = -1
                        Nodes[-1] = root
                        root.add_child(
                            tree, self.rel_vocab.getIndex(parent_rel, util.UNK_WORD))
                        break
                    else:
                        prev = tree
                        rel = tri_case[idx][2]
                        idx = parent
        if root is None:
            print(tri_case)
        return root

    def read_trees(self):
        articles = []
        for paragraph in self.tree_data:
            trees = []
            for sent in paragraph['tree']:
                if sent is None:
                    trees.append(None)
                else:
                    root = self.get_tree(sent)
                    trees.append(root)
            articles.append(trees)
        return articles

#
# class CommonDataset(data.Dataset):
#     def __len__(self):
#         return len(self.sentences)
#
#     def __getitem__(self, index):
#         article_ind, paragraph_ind = index
#         sents = deepcopy(self.sentences[article_ind][paragraph_ind])
#         trees = deepcopy(self.trees[article_ind][paragraph_ind])
#         targets = deepcopy(self.data[article_ind][paragraph_ind]['targets'])
#         masks = deepcopy(self.data[article_ind][paragraph_ind]['masks'])
#         return sents, trees, targets, masks
#
#     def __init__(self, path, device, vocab=None):
#         super(CommonDataset, self).__init__()
#         self.path = path
#         self.data = self.read_data(path)
#         if not vocab:
#             self.vocab = self.build_vocab()
#         else:
#             self.vocab = vocab
#
#         self.rel_vocab = self.build_rel_vocab()
#         self.rel_emb_size = 50
#
#         self.device = device
#
#         self.sentences = self.read_sentences()
#         self.trees = self.read_trees()
#         print("sentence", len(self.sentences))
#         print("trees", len(self.trees))
#
#     def get_target(self):
#         for data_case in self.data:
#             targets = []
#             id_paths = []
#             for path_case in data_case['paths']:
#                 target = []
#                 id_path = []
#                 for path in path_case:
#                     path.insert(0, '<s>')
#                     # path.append('</s>')
#                     indices = self.vocab.convertToIdx(path, Constants.UNK_WORD)
#                     # print('INDICES', indices)
#                     id_path.append(indices)
#                     end_id = indices[-1]
#                     target.append(end_id)
#                 # one_hot_target = np.zeros([self.vocab.size(), len(path_case)])
#                 # one_hot_target[target, np.arange(len(path_case))] = 1
#                 id_paths.append(id_path)
#                 # targets.append(one_hot_target)
#                 targets.append(target)
#             data_case['targets'] = targets
#             data_case['paths'] = id_paths
#
#     @staticmethod
#     def get_target_and_path(paths, vocab):
#         targets = []
#         id_paths = []
#         for path_case in paths:
#             target = []
#             id_path = []
#             for path in path_case:
#                 path.insert(0, '<s>')
#                 # path.append('</s>')
#                 indices = vocab.convertToIdx(path, Constants.UNK_WORD)
#                 # print('INDICES', indices)
#                 id_path.append(indices)
#                 end_id = indices[-1]
#                 target.append(end_id)
#             # one_hot_target = np.zeros([self.vocab.size(), len(path_case)])
#             # one_hot_target[target, np.arange(len(path_case))] = 1
#             id_paths.append(id_path)
#             targets.append(target)
#         return targets, id_paths
#
#     def build_vocab(self):
#         if not os.path.isfile(QAConfig.vocab):
#             utils.build_vocab([self.path], QAConfig.vocab)
#         return Vocab(filename=QAConfig.vocab,
#                      data=[
#                          Constants.PAD_WORD, Constants.UNK_WORD])
#         # Constants.BOS_WORD, Constants.EOS_WORD])
#
#     def build_rel_vocab(self):
#         if not os.path.isfile(QAConfig.rel_vocab):
#             utils.build_rel_vocab([self.path], QAConfig.rel_vocab)
#         return Vocab(filename=QAConfig.rel_vocab,
#                      data=[Constants.UNK_WORD])
#
#     def build_embedding(self):
#         if os.path.isfile(QAConfig.embed):
#             emb = torch.load(QAConfig.embed)
#         else:
#             # load glove embeddings and vocab
#             glove_vocab, glove_emb = utils.load_word_vectors(QAConfig.glove)
#             logger.debug('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
#             emb = torch.zeros(self.vocab.size(), glove_emb.size(1), dtype=torch.float, device=self.device)
#             emb.normal_(0, 0.05)
#             # zero out the embeddings for padding and other special words if they are absent in vocab
#             for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD]):
#                 # Constants.BOS_WORD, Constants.EOS_WORD]):
#                 if idx == 0:
#                     emb[idx].fill_(1e-3)
#                 if idx == 1:
#                     emb[idx].fill_(1e-2)
#                 if idx == 2:
#                     emb[idx].fill_(1e-1)
#                 if idx == 3:
#                     emb[idx].fill_(1)
#             for word in self.vocab.labelToIdx.keys():
#                 if glove_vocab.getIndex(word):
#                     emb[self.vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
#             torch.save(emb, QAConfig.embed)
#         return emb
#
#     def build_rel_embedding(self):
#         if os.path.isfile(QAConfig.rel_embed):
#             emb = torch.load(QAConfig.rel_embed)
#         else:
#             emb = torch.randn(self.rel_vocab.size(), self.rel_emb_size, requires_grad=True)
#             emb[0].fill_(1e-2)
#             torch.save(emb, QAConfig.rel_embed)
#         return emb
#
#     # sentences is an article
#     def read_sentences(self):
#         articles = []
#         for article in self.data:
#             sentences = []
#             for paragraph in article:
#                 sentence_list = []
#                 targets_list = []
#                 for sent_case in paragraph['words']:
#                     # indices = self.vocab.convertToIdx(['<s>'] + sent_case + ['</s>'], Constants.UNK_WORD)
#                     indices = self.vocab.convertToIdx(sent_case, Constants.UNK_WORD)
#                     try:
#                         indices = torch.tensor(indices, dtype=torch.long, device=self.device)
#                     except:
#                         print(sent_case)
#                     # targets_indices = torch.tensor(indices[1:], dtype=torch.long, device=self.device)
#                     sentence_list.append(indices)
#                     targets_list.append(indices)
#                 sentences.append(sentence_list)
#                 paragraph['targets'] = targets_list
#             articles.append(sentences)
#             # with an end notation
#         # with open(filename, 'r') as f:
#         #     sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
#         return articles
#
#     @staticmethod
#     def get_sentences_id(paragraph, vocab, device):
#         sentences = []
#         for data_case in paragraph:
#             sentence_list = []
#             for sent_case in data_case['words']:
#                 indices = vocab.convertToIdx(sent_case, Constants.UNK_WORD)
#                 torch_indices = torch.tensor(indices, dtype=torch.long, device=device)
#                 sentence_list.append(torch_indices)
#             sentences.append(sentence_list)
#         # with open(filename, 'r') as f:
#         #     sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
#         return sentences
#
#     def get_tree(self, tri_case):
#         """
#         :param tri_case: one sentence tri case
#         :return: root node
#         """
#         tri_case.sort(key=lambda x: x[1][1])
#         # print('Tri sorted:', tri_case)
#         Nodes = dict()
#         root = None
#         for i in range(len(tri_case)):
#             # if i not in Nodes.keys() and tri_case[i][0][1] != -1:
#             if i not in Nodes.keys():
#                 idx = i
#                 prev = None
#                 rel = None
#                 while True:
#                     tree = TreeNode()
#                     Nodes[idx] = tree
#                     tree.idx = idx
#                     if prev is not None:
#                         tree.add_child(
#                             prev, self.rel_vocab.convertToIdx([rel], Constants.UNK_WORD)[0])
#                     parent = tri_case[idx][0][1]
#                     parent_rel = tri_case[idx][2]
#                     if parent in Nodes.keys():
#                         Nodes[parent].add_child(
#                             tree, self.rel_vocab.convertToIdx([parent_rel], Constants.UNK_WORD)[0])
#                         break
#                     elif parent == -1:
#                         root = TreeNode()
#                         root.idx = -1
#                         Nodes[-1] = root
#                         root.add_child(
#                             tree, self.rel_vocab.convertToIdx([parent_rel], Constants.UNK_WORD)[0])
#                         break
#                     else:
#                         prev = tree
#                         rel = tri_case[idx][2]
#                         idx = parent
#         if root is None:
#             print(tri_case)
#         return root
#
#     def read_trees(self):
#         articles = []
#         for article in self.data:
#             trees = []
#             for paragraph in article:
#                 tree_list = []
#                 if paragraph['triples'] is None:
#                     tree_list.append(None)
#                 else:
#                     for tri_case in paragraph['triples']:
#                         if tri_case is None:
#                             tree_list.append(None)
#                             continue
#                         root = self.get_tree(tri_case)
#
#                         debug_dict = {}
#
#                         def debug(trees, p=False):
#                             debug_dict[trees.idx] = True
#                             if p:
#                                 print(trees.idx)
#                             for i in range(trees.num_children):
#                                 child = trees.children[i]
#                                 debug(child, p)
#
#                         debug(root)
#                         for i in range(-1, len(tri_case)):
#                             try:
#                                 if not debug_dict[i]:
#                                     print(tri_case)
#                             except KeyError:
#                                 print(tri_case)
#                                 debug(root, True)
#                                 exit(0)
#                         tree_list.append(root)
#                 trees.append(tree_list)
#             articles.append(trees)
#         return articles
#
#     def get_trees(self, triples):
#         """
#         :param triples: list(list(triples)): a paragraph of triples
#         :return:
#         """
#         tree_list = []
#         for tri_case in triples:
#             root = self.get_tree(tri_case)
#             tree_list.append(root)
#         return tree_list
#
#     @staticmethod
#     def read_data(filename):
#         with open(filename, 'r') as load_f:
#             data = json.load(load_f)
#         return data
