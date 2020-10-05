import torch

import torch.nn as nn
import torch.nn.functional as F


class TreeDecoder(nn.Module):
    def forward(self, input, hidden, word=None):
        output = self.relation_emb(input).view(1, -1, self.relation_dim)
        if word is not None:
            word = word.view(1, -1, self.embedding_dim)
            output = torch.cat((output, word), dim=2)
        output = F.relu(output)

        output, hidden = self.lstm(output, hidden)
        output = self.decoder(output.view(-1, self.hidden_dim))
        return output, hidden

    def __init__(self, vocab, device, relation_emb, hidden_dim=300, embedding_dim=300, freeze=True,
                 relation_dim=50, nlayers=2, dropout=0.5):
        super().__init__()

        self.vocab = vocab
        self.device = device
        if embedding_dim > 0:
            # self.emb = nn.Embedding(vocab.size(), embedding_dim, padding_idx=Constants.PAD)
            self.emb = nn.Embedding(vocab.size(), embedding_dim)
            if freeze:
                self.emb.weight.requires_grad = False
        self.relation_emb = relation_emb
        # sentences should be set every paragraph
        self.sentences = None
        self.nlayers = nlayers
        self.embedding_dim = embedding_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(relation_dim + embedding_dim, hidden_dim, nlayers, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_dim, vocab.size())

        init_range = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)
