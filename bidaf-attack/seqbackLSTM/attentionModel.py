import torch

import torch.nn as nn

from common import Constants


class LSTMDecoder(nn.Module):
    """
        Take a paragraph of tree LSTM embedding as input
    """

    def forward(self, inputs):
        outputs = []
        assert len(self.sentences) == len(inputs), str(len(self.sentences)) + " != " + str(len(inputs)) + "???"
        for i in range(len(self.sentences)):
            sentence_emb = self.emb(self.sentences[i]).view(1, -1, self.embedding_dim)
            lstm_hidden = inputs[i].view(1, -1, self.hidden_dim)
            lstm_hidden = torch.cat([lstm_hidden for _ in range(self.nlayers)])
            output, _ = self.lstm(sentence_emb, (lstm_hidden, lstm_hidden))
            outputs.append(self.decoder(output.view(-1, self.hidden_dim)))
        return outputs

    def __init__(self, vocab, device, hidden_dim=300, embedding_dim=300, freeze=True,
                 nlayers=2, dropout=0.5):
        super().__init__()

        self.vocab = vocab
        self.device = device
        # self.emb = nn.Embedding(vocab.size(), embedding_dim, padding_idx=Constants.PAD)
        self.emb = nn.Embedding(vocab.size(), embedding_dim)
        if freeze:
            self.emb.weight.requires_grad = False
        # sentences should be set every paragraph
        self.sentences = None
        self.nlayers = nlayers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, nlayers, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_dim, vocab.size())

        init_range = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)
