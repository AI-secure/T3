import numpy as np

import torch
import torch.nn as nn

from torch.autograd import Variable



class chainLSTM(nn.Module):
    """Container module with a recurrent module, and a decoder."""

    def __init__(self, vocab_size, in_dim=300, hidden_dim=300, nlayers=2, dropout=0.5):
        super(chainLSTM, self).__init__()

        self.lstm = nn.LSTM(in_dim, hidden_dim, nlayers, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # todo: what is this range means
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, root, length):
        # todo: change the hardcoded dimension
        self.hidden = (root.expand(2, length, 300).contiguous(), root.expand(2, length, 300).contiguous())

    def forward(self, input):
        # print('INPUT:', input)
        output, hidden = self.lstm(input, self.hidden)
        # output = self.drop(output)
        # print('OUTPUT:', output.size())
        output, output_len = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = output.contiguous()
        # print('OUTPUT:', output)
        # print('OUTPUT LEN:', output_len)
        output_words = []
        for i, position in enumerate(output_len):
            # print('POSITION:', position.numpy())
            # print('I:', i)
            output_word = output[i, position - 1, :]
            output_word = output_word.unsqueeze(0)
            # print('OUTPUT_WORD_SIZE', output_word.size())
            # output_words.append(torch.transpose(output_word))
            output_words.append(output_word)
        # print('OUTPUT_WORDS', output_words)
        output_words = torch.cat(output_words, 0)
        # print('OUTPUT_WORDS_CAT', output_words)
        # print('OUTPUT_WORDS_CAT', output_words.size())

        decoded = self.decoder(output_words)
        # print('DECODED:', decoded.size())
        return decoded


class SeqbackLSTM(nn.Module):
    # todo: embedding_dim set by arg
    """
        Take a paragraph as input
    """
    def __init__(self, vocab, device, embedding_dim=300, freeze=True):
        super(SeqbackLSTM, self).__init__()

        self.vocab = vocab
        self.device = device
        # self.emb = nn.Embedding(vocab.size(), embedding_dim, padding_idx=Constants.PAD)
        self.emb = nn.Embedding(vocab.size(), embedding_dim)
        if freeze:
            self.emb.weight.requires_grad = False
        # path_set should be set every paragraph
        self.path_set = None
        self.chainLSTM = chainLSTM(vocab.size(), in_dim=embedding_dim)

    def forward(self, roots):
        paths_output = []
        for i in range(len(self.path_set)):
            paths = self.path_set[i]
            root = roots[i]
            length = np.array([len(path) for path in paths])
            path_input = np.zeros((len(length), np.amax(length)))
            for i, p in enumerate(paths):
                path_input[i, 0:len(p)] = np.array(p)
            path_input = Variable(torch.from_numpy(path_input).long()).to(self.device)
            path_input = self.emb(path_input)

            length, idx_sort = np.sort(length)[::-1], np.argsort(-length)
            idx_unsort = np.argsort(idx_sort)
            idx_sort = torch.from_numpy(idx_sort).to(self.device)
            path_input = path_input.index_select(0, Variable(idx_sort))
            embed_packed_path_input = torch.nn.utils.rnn.pack_padded_sequence(path_input, length, batch_first=True)
            # print('SHAPE', embed_packed_path_input.batch_sizes)
            self.chainLSTM.init_hidden(root.to(self.device), embed_packed_path_input.batch_sizes[0].int())
            output = self.chainLSTM(embed_packed_path_input)
            # output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]
            idx_unsort = torch.from_numpy(idx_unsort).to(self.device)
            path_output = output.index_select(0, Variable(idx_unsort))
            paths_output.append(path_output)
        return paths_output

    # def forward(self, path_set, root):
    #     # todo: use simply path set
    #     # print('PATH_SET:', path_set)
    #     length = np.array([len(path) for path in path_set])
    #     # print('LENGTH', length)
    #     path_input = np.zeros((len(length), np.amax(length)))
    #     for i, p in enumerate(path_set):
    #         path_input[i, 0:len(p)] = np.array(p)
    #     path_input = Variable(torch.from_numpy(path_input).long()).to(self.device)
    #     # indices = []
    #     # for path in path_set:
    #     #     length.append(len(path))
    #     #     indice = self.vocab.convertToIdx(path, Constants.UNK_WORD)
    #     #     indices.append(indice)
    #     #     path = torch.tensor(indice, dtype=torch.long, device=self.device)
    #     #     path = self.emb(path)
    #     path_input = self.emb(path_input)
    #
    #     length, idx_sort = np.sort(length)[::-1], np.argsort(-length)
    #     idx_unsort = np.argsort(idx_sort)
    #     idx_sort = torch.from_numpy(idx_sort).to(self.device)
    #     path_input = path_input.index_select(0, Variable(idx_sort))
    #     embed_packed_path_input = torch.nn.utils.rnn.pack_padded_sequence(path_input, length, batch_first=True)
    #     # print('SHAPE', embed_packed_path_input.batch_sizes)
    #     self.chainLSTM.init_hidden(root.to(self.device), embed_packed_path_input.batch_sizes[0].int())
    #     output = self.chainLSTM(embed_packed_path_input)
    #     # output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]
    #     idx_unsort = torch.from_numpy(idx_unsort).to(self.device)
    #     path_output = output.index_select(0, Variable(idx_unsort))
    #
    #     return path_output
