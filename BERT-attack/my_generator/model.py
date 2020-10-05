import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import util
from seqbackLSTM.tree import TreeDecoder
from treeLSTM import treeNode
from treeLSTM.model import TreeLSTM
from treeLSTM.treeNode import TreeNode
from util import get_args
from vocab import Vocab

args = get_args()

logger = logging.getLogger(__name__)
info = logger.info


class WrappedSeqback(nn.Module):
    """
        Take a batch of paragraphs as input?
    """
    def __init__(self, embed, device, attack=False, seqback_model=None, vocab=None, transfer_emb=None):
        super().__init__()
        # a batch of parapgraph of sentences
        self.device = device
        self.embed = torch.nn.Embedding(embed.shape[0], embed.shape[1]).to(self.device)
        self.embed.weight.data.copy_(embed)
        self.attack = attack
        self.seqback_model = seqback_model
        self.vocab = vocab
        self.teacher_forcing_ratio = args.tr
        self.null_token = self.embed(torch.LongTensor([self.vocab.getIndex(util.SOS_WORD)]).to(self.device))

        self.sentences = None  # embedding of input sentences
        self.batch_add_sent = None
        self.batch_trees = None
        self.adv_sent = None
        self.start = None
        self.end = None
        self.temp = None
        self.transfer_emb = transfer_emb
        self.target_start = None
        self.target_end = None
        self.target = None


    def forward(self, hiddens):
        result = self.sentences.clone().detach().requires_grad_(False)
        if not self.attack:
            return self.sentences

        for batch_idx in range(self.sentences.shape[0]):
            # every batch
            tree = self.batch_trees[batch_idx]
            hidden = hiddens[batch_idx]

            output = {}
            tree.hidden = torch.cat([hidden for _ in range(self.seqback_model.nlayers)]).view(
                self.seqback_model.nlayers, 1, self.seqback_model.hidden_dim)
            tree.hidden = (tree.hidden, tree.hidden)
            if args.decode_word:
                sentence = self.seqback_model.emb(torch.LongTensor(self.batch_add_sent[batch_idx]).cuda())
                self.traverse(tree, output, sentence)
            else:
                self.traverse(tree, output)

            try:
                output = torch.cat([output[j] for j in range(len(output))])
            except KeyError:
                print(self.seqback_model.sentences)
                self.debug(tree)
            output = F.softmax(output / self.temp, dim=1)
            # output = F.gumbel_softmax(output, hard=True, eps=1e-10)
            # debug
            if self.vocab is not None and output is not None:
                val, res = torch.max(output, 1)
                if self.target is not None:
                    res[self.target_start:self.target_end+1] = self.target
                output_label = self.vocab.tensorConvertToLabels(res, -100)
                output_emb = self.transfer_emb(res)
                self.adv_sent.append(output_label)
                if args.debugging:
                    # print("probability:", val)
                    print("seqback result:", output_label)
                        # print("seqback result:", output_label, file=open("tree_attack/adv_sent.txt", "a"))

            start = self.start[batch_idx]
            end = self.end[batch_idx]
            result[batch_idx, start:end] = torch.matmul(output[:end - start], self.transfer_emb.weight.data)
            result.data[batch_idx, start:end] = output_emb[:end - start]
        return result

    def traverse(self, node: treeNode, output, sentence=None):
        relations = torch.tensor(node.relation, device=self.device, dtype=torch.long)
        prev = node
        for idx in range(node.num_children):
            child = node.children[idx]
            if sentence is not None:
                is_teacher = random.random() < self.teacher_forcing_ratio
                if prev.idx == -1:
                    word_emb = self.null_token.clone().detach()
                else:
                    predicted_word = self.embed(torch.argmax(output[prev.idx]))
                    assert predicted_word.shape == sentence[prev.idx].shape
                    word_emb = sentence[prev.idx] if is_teacher else predicted_word
                output[child.idx], child.hidden = self.seqback_model(relations[idx], node.hidden,
                                                                     word=word_emb)
            else:
                output[child.idx], child.hidden = self.seqback_model(relations[idx], node.hidden)
            prev = child
            self.traverse(child, output, sentence)


class Generator(nn.Module):
    """
        Take a paragraph as input
    """

    def forward(self, sents, trees, masks=None):
        hiddens, prediction = self.tree_model(sents, trees, masks)
        outputs = []
        for i in range(len(trees)):
            output = {}
            trees[i].hidden = torch.cat([hiddens[i] for _ in range(self.seqback_model.nlayers)]).view(
                self.seqback_model.nlayers, 1, self.seqback_model.hidden_dim)
            trees[i].hidden = (trees[i].hidden, trees[i].hidden)
            if args.decode_word:
                sentence = self.seqback_model.emb(sents[i])
                self.traverse(trees[i], output, sentence)
            else:
                self.traverse(trees[i], output)

            try:
                output = torch.cat([output[i] for i in range(len(output))])
            except KeyError:
                print(self.seqback_model.sentences)
                self.debug(trees[i])
            outputs.append(output)
        return outputs

    def debug(self, trees):
        print(trees.idx)
        for i in range(trees.num_children):
            child = trees.children[i]
            self.debug(child)

    def traverse(self, node: treeNode, output, sentence=None):
        # print(node.relation)
        # print(node.num_children)
        relations = torch.tensor(node.relation, device=self.device, dtype=torch.long)
        prev = node
        for idx in range(node.num_children):
            child = node.children[idx]
            # print("idx: {}".format(idx))
            # print("child {}".format(child.idx))
            if sentence is not None:
                is_teacher = random.random() < self.teacher_forcing_ratio
                if prev.idx == -1:
                    word_emb = self.null_token.clone().detach()
                else:
                    predicted_word = self.tree_model.emb(torch.argmax(output[prev.idx]))
                    assert predicted_word.shape == sentence[prev.idx].shape
                    word_emb = sentence[prev.idx] if is_teacher else predicted_word
                output[child.idx], child.hidden = self.seqback_model(relations[idx],
                                                                     node.hidden, word=word_emb)
            else:
                output[child.idx], child.hidden = self.seqback_model(relations[idx], node.hidden)
            prev = child
            self.traverse(child, output, sentence)

    def __init__(self, path, vocab=None, embed=None, data_set=None):
        super().__init__()
        # GPU select
        args.cuda = args.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if args.cuda else "cpu")

        self.data_set = data_set
        if embed is not None:
            self.embed = embed
        else:
            self.embed = self.data_set.build_embedding()
        # todo: torch save dataset

        if args.sparse and args.wd != 0:
            logger.error('Sparsity and weight decay are incompatible, pick one!')
            exit()

        # debugging args
        logger.debug(args)
        # set seed for embedding metrics
        torch.manual_seed(args.seed)
        random.seed(args.seed)

        self.vocab = vocab
        rel_vocab = "rel_vocab.txt"
        self.rel_emb_size = 50
        self.rel_vocab = Vocab(filename=rel_vocab, data=[util.UNK_WORD])
        self.rel_emb = torch.nn.Embedding(self.rel_vocab.size(), self.rel_emb_size).to(self.device)
        # initialize tree_model, criterion/loss_function, optimizer
        # assume mem_dim = hidden_dim
        if args.encode_rel:
            self.tree_model = TreeLSTM(
                self.vocab.size(),
                self.embed.shape[1],
                args.hidden_dim,
                args.hidden_dim,
                args.sparse,
                device=self.device,
                rel_dim=self.rel_emb_size,
                rel_emb=self.rel_emb)
        else:
            self.tree_model = TreeLSTM(
                self.vocab.size(),
                self.embed.shape[1],
                args.mem_dim,
                args.hidden_dim,
                args.num_classes,
                args.sparse,
                device=self.device)

        # self.tree_criterion = nn.KLDivLoss()
        self.tree_model.to(self.device)
        # plug these into embedding matrix inside tree_model
        self.tree_model.emb.weight.data.copy_(self.embed)

        if args.decode_word:
            self.seqback_model = TreeDecoder(
                self.vocab, self.device, self.rel_emb,
                embedding_dim=self.embed.shape[1], hidden_dim=args.hidden_dim)
        else:
            self.seqback_model = TreeDecoder(
                self.vocab, self.device, self.rel_emb, hidden_dim=args.hidden_dim,
                embedding_dim=0)
        self.seqback_criterion = nn.CrossEntropyLoss()
        self.seqback_model.to(self.device), self.seqback_criterion.to(self.device)

        self.teacher_forcing_ratio = args.tr
        self.null_token = self.tree_model.emb(torch.LongTensor([self.vocab.getIndex(util.SOS_WORD)]).to(self.device))

        if args.decode_word:
            self.seqback_model.emb.weight.data.copy_(self.embed)
        if args.optim == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                               self.parameters()), lr=args.lr, weight_decay=args.wd)
        elif args.optim == 'adagrad':
            self.optimizer = optim.Adagrad(filter(lambda p: p.requires_grad,
                                                  self.parameters()), lr=args.lr, weight_decay=args.wd)
        elif args.optim == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                              self.parameters()), lr=args.lr, weight_decay=args.wd)

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
