import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from torch.autograd import Variable

from common import Constants
from util import get_args

args = get_args()


class EncoderRNN(nn.Module):
    def __init__(self, vocab, embed_size, hidden_size, device, n_layers=1):
        # vocab already contains <eos> and <sos>
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.device = device
        # vocab.size() == input vocab size
        self.embedding = nn.Embedding(vocab.size(), embed_size)
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, bidirectional=True)

    def forward(self, src, hidden=None):
        # **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
        #           of the input sequence.
        embedded = self.embedding(src)
        outputs, hidden = self.lstm(embedded, hidden)
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=2, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.embed.weight.requires_grad = False
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(hidden_size + embed_size, hidden_size,
                            n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[0][-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.lstm(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights


class Seq2SeqGenerator(nn.Module):
    """
        Take a paragraph as input
    """

    def __init__(self, encoder: EncoderRNN, decoder: Decoder, embed, dataset=None):
        super(Seq2SeqGenerator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed = embed
        self.dataset = dataset
        self.device = self.encoder.device
        self.encoder.embedding.weight.data.copy_(self.embed)
        self.decoder.embed.weight.data.copy_(self.embed)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        encoder_output, hidden = self.encoder(src)
        # hidden = hidden[:self.decoder.n_layers] # not suitable for lstm cell
        output = Variable(trg.data[0, :])  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(trg.data[t] if is_teacher else top1).cuda()
        return outputs


# use the parameters in seq2seq generator without changing the model itself
class WrappedSeqDecoder(nn.Module):
    """
        Take the hidden states as input
        Output QA system's embedding input
    """

    def __init__(self, decoder, vocab):
        super(WrappedSeqDecoder, self).__init__()
        self.decoder = decoder
        self.vocab = vocab

        self.encoder_output = None
        self.trg = None
        self.sentences = None  # paragraph of sentences
        self.start = None  # pos of perturbed sentence
        self.end = None
        self.adv_sent = None

        self.softmax = nn.Softmax(dim=1)
        self.temp = args.temp

    """
        the cw can only input one parameter (we let it to be hiddens)
    """

    def forward(self, hidden):
        # parameters needed to be passed from member variable
        trg = self.trg  # targeted sentence
        encoder_output = self.encoder_output
        # from (batch, 1, num_layers * num_directions * 2, hidden_size)
        hidden = hidden.view(-1, 2, 2, self.decoder.hidden_size).transpose_(0, 2)
        hidden = (hidden[0].contiguous(), hidden[1].contiguous())

        batch_size = trg.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()
        results = self.decoder.embed(self.sentences)

        output = Variable(trg.data[0, :])  # sos

        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                output, hidden, encoder_output)
            outputs[t] = output
            # output shape is batch * vocab_size
            top1 = output.data.max(1)[1]
            output = Variable(top1).cuda()

        final_outputs = Variable(torch.zeros(max_len-1, batch_size, vocab_size)).cuda()
        nll_tensor = torch.zeros(vocab_size).cuda()
        eos_idx = self.vocab.getIndex(Constants.EOS_WORD)
        sos_idx = self.vocab.getIndex(Constants.SOS_WORD)
        nll_idx = self.vocab.getIndex(Constants.PAD_WORD)
        nll_tensor[nll_idx] = 1

        for t in range(1, max_len):
            output = outputs[t]
            top1 = output.data.max(1)[1]
            for i, top in enumerate(top1):
                if top == eos_idx or top == sos_idx:
                    final_outputs[t - 1, i] = nll_tensor
                else:
                    final_outputs[t - 1, i] = output[i]
        final_outputs = F.softmax(final_outputs / self.temp, dim=2)
        # put final outputs into results
        for batch_idx in range(results.size(1)):
            start = self.start[batch_idx]
            end = self.end[batch_idx]
            output = final_outputs[:, batch_idx]
            # print(results[start:end, batch_idx].shape)
            # print(output[:end-start].shape)
            try:
                results[start:end, batch_idx] = torch.matmul(output[:end - start],
                                                                    self.decoder.embed.weight.data)
            except:
                continue
            res = torch.argmax(output, 1)
            self.adv_sent.append(self.vocab.tensorConvertToLabels(res, stop=nll_idx))
            if args.debugging:
                val, res = torch.max(output, 1)
                print("temp", self.temp)
                print("probability:", val)
        return results
