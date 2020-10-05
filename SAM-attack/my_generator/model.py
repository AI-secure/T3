import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import Constants
from seqbackLSTM.tree import TreeDecoder
from treeLSTM import treeNode
from treeLSTM.model import TreeLSTM
from util import get_args

args = get_args()

logger = logging.getLogger(__name__)
info = logger.info


class WrappedGenerator(nn.Module):
    def forward(self):
        pass

    def __init__(self, seq_back_model):
        super().__init__()

        self.sentence = None
        self.seq_back_model = seq_back_model

    """
    输入： sentences、trees 和 masks
    输出： 变成了  embedding 的 sentences，直接连入网络
    
    需要：   embedding matrix、masks 来判断哪些句子需要走 Generator 模型那些不需要
    问题： CW 怎么加： 需要一个模型 以那个为输入
    
    
    方案2：
    输入： embedding，是 seqback 模型的包装
    网络外部合并 adv_seqback 和 normal sentences embedding
    并送到 f 网络
    """


class WrappedSeqback(nn.Module):
    """
        Take a batch of paragraphs as input?
    """

    def forward(self, hiddens):
        result = self.transfer_emb(self.sentences)
        if not self.attack:
            return result

        for batch_idx in range(self.sentences.shape[1]):
            # every batch
            trees = self.batch_trees[batch_idx]
            hidden = hiddens[batch_idx]
            outputs = []
            for i in range(len(trees)):
                if self.batch_masks[batch_idx][i] is False:
                    outputs.append(None)
                    # if self.vocab is not None and args.debugging:
                    #     print('seqback result: None')
                    continue
                output = {}
                trees[i].hidden = torch.cat([hidden for _ in range(self.seqback_model.nlayers)]).view(
                    self.seqback_model.nlayers, 1, self.seqback_model.hidden_dim)
                trees[i].hidden = (trees[i].hidden, trees[i].hidden)
                if args.decode_word:
                    sentence = self.seqback_model.emb(self.batch_splitted_sentences[batch_idx][i])
                    self.traverse(trees[i], output, sentence)
                else:
                    self.traverse(trees[i], output)

                try:
                    output = torch.cat([output[j] for j in range(len(output))])
                except KeyError:
                    print(self.seqback_model.sentences)
                    self.debug(trees[i])
                output = F.softmax(output / self.temp, dim=1)
                # output = F.gumbel_softmax(output, hard=True, eps=1e-10)

                outputs.append(output)
                # debug
                if self.vocab is not None and output is not None:
                    val, res = torch.max(output, 1)
                    output_label = self.vocab.tensorConvertToLabels(res, -100)
                    output_emb = self.transfer_emb(res)
                    self.adv_sent.append(res)
                    if args.debugging:
                        # print("probability:", val)
                        print("seqback result:", output_label)
                        # print("seqback result:", output_label, file=open("tree_attack/adv_sent.txt", "a"))

            start = self.start[batch_idx]
            end = self.end[batch_idx]
            result[start:end, batch_idx] = torch.matmul(output[:end - start], self.transfer_emb.weight.data)
            result.data[start:end, batch_idx] = output_emb[:end - start]
        return result

            # for i, output in enumerate(outputs):
            #     if output is not None:
            #         start = 0
            #         for j in range(i):
            #             start += self.batch_splitted_sentences[batch_idx][j].shape[0]
            #         end = start + self.batch_splitted_sentences[batch_idx][i].shape[0]
            #         result[batch_idx, 0, start:end] = torch.matmul(output, self.embed.weight.data)

            # print(output.shape)
            # [12, 100] * [100, embedding_size]
            # print(self.embed.data.)

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
        self.null_token = self.embed(torch.LongTensor([self.vocab.getIndex(Constants.SOS_WORD)]).to(self.device))

        self.sentences = None
        self.batch_splitted_sentences = None
        self.batch_trees = None
        self.batch_masks = None
        self.adv_sent = None
        self.start = None
        self.end = None
        self.temp = None
        self.transfer_emb = transfer_emb
        # self.linear1 = nn.Linear(100, 100)
        # self.linear2 = nn.Linear(100, 100)
        # self.linear3 = nn.Linear(100, 100)
        # self.lstm = nn.LSTM(100, 100, 3, batch_first=True)


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
        self.rel_emb = torch.nn.Embedding(self.data_set.rel_vocab.size(), self.data_set.rel_emb_size).to(self.device)
        # initialize tree_model, criterion/loss_function, optimizer
        # assume mem_dim = hidden_dim
        if args.encode_rel:
            self.tree_model = TreeLSTM(
                self.data_set.vocab.size(),
                self.embed.shape[1],
                args.hidden_dim,
                args.hidden_dim,
                args.sparse,
                device=self.device,
                rel_dim=self.data_set.rel_emb_size,
                rel_emb=self.rel_emb)
        else:
            self.tree_model = TreeLSTM(
                self.data_set.vocab.size(),
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
                self.data_set.vocab, self.device, self.rel_emb,
                embedding_dim=self.embed.shape[1], hidden_dim=args.hidden_dim)
        else:
            self.seqback_model = TreeDecoder(
                self.data_set.vocab, self.device, self.rel_emb, hidden_dim=args.hidden_dim,
                embedding_dim=0)
        self.seqback_criterion = nn.CrossEntropyLoss()
        self.seqback_model.to(self.device), self.seqback_criterion.to(self.device)

        self.teacher_forcing_ratio = args.tr
        self.null_token = self.tree_model.emb(torch.LongTensor([self.vocab.getIndex(Constants.SOS_WORD)]).to(self.device))

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

        logger.debug('==> Size of train data   : %d ' % len(self.data_set))
