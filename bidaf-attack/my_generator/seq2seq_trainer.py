import logging
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torchnlp.metrics import get_token_accuracy
from tqdm import tqdm

from my_generator.dataset import YelpDataset
from my_generator.seq2seq_model import Seq2Seq, Encoder, Decoder
from util import args, logger
from vocab import Vocab


class BaseSeqTrainer:
    def __init__(self, model: Seq2Seq, dataset, device, args):
        self.model = model
        self.dataset = dataset
        for name, param in self.model.named_parameters():
            # print(type(param.data), param.size())
            print(name, type(param.data), param.size(), param.requires_grad)
        # self.optimizer = optim.Adam(filter(lambda p: p.requires_grad,
        #                                    model.parameters()), lr=args.lr)
        if args.optim == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=args.lr, weight_decay=args.wd)
        elif args.optim == 'adagrad':
            self.optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()),
                                           lr=args.lr, weight_decay=args.wd)
        elif args.optim == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                       lr=args.lr, weight_decay=args.wd)
        elif args.optim == 'adadelta':
            self.optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()),
                                            lr=args.lr, weight_decay=args.wd)
        self.batch = args.batch_size
        self.args = args
        self.device = device
        self.grad_clip = 10
        self.epoch = args.epochs
        self.lr = args.lr
        self.writer = SummaryWriter()

    def adjust_learning_rate(self):
        """Sets the learning rate to the initial LR decayed by 4 when eval result drops"""
        self.lr = self.lr * 0.2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def get_batch(self, dataset=None):
        batch_sentences = []
        batch_targets = []
        extreme = 0
        tot = 0
        if dataset is None:
            dataset = self.dataset
        for j in tqdm(range(len(dataset))):
            sentences = dataset[j]['split_text']
            for k in range(len(sentences)):
                tot += 1
                sentence = torch.tensor(sentences[k], device=self.device)
                # threshold the sentence length
                if len(sentence) > 100:
                    self.writer.add_scalar("extreme_sentence_len", len(sentence), extreme)
                    extreme += 1
                    sentence = sentence[:100]
                sos = torch.tensor([self.dataset.vocab.getIndex(SOS_WORD)], dtype=torch.long,
                                   device=self.device)
                eos = torch.tensor([self.dataset.vocab.getIndex(EOS_WORD)], dtype=torch.long,
                                   device=self.device)
                sentence = torch.cat((sos, sentence, eos), 0)
                batch_sentences.append(sentence)

                target = torch.tensor(sentence)
                batch_targets.append(target)

                if len(batch_targets) == self.batch:
                    # pad the input and target
                    batch_sentences = pad_sequence(batch_sentences, padding_value=PAD)
                    batch_targets = pad_sequence(batch_targets, padding_value=PAD)
                    yield batch_sentences, batch_targets
                    batch_targets = []
                    batch_sentences = []
        print("extreme / tot: {0} / {1}".format(extreme, tot))
        if len(batch_sentences) > 0:
            batch_sentences = pad_sequence(batch_sentences, padding_value=PAD)
            batch_targets = pad_sequence(batch_targets, padding_value=PAD)
            yield batch_sentences, batch_targets

    def evaluate(self, test_dataset=None, saved_model=None):
        total_loss = 0
        overall_correct = 0
        overall_tot = 0
        step = 0

        if not test_dataset:
            dataset = self.dataset
        else:
            dataset = test_dataset

        if saved_model:
            self.model.load_state_dict(torch.load(saved_model))

        if self.args.save_ans:
            with open(self.args.save_ans, 'w') as f:
                f.write('start dump \n')

        self.model.eval()
        with torch.no_grad():
            for idx_b, batch in enumerate(self.get_batch(dataset)):
                src, trg = batch
                output = self.model(src, trg, teacher_forcing_ratio=0.0)
                loss = F.nll_loss(output[1:].view(-1, self.dataset.vocab.size()),
                                  trg[1:].contiguous().view(-1), ignore_index=PAD)
                # total_loss += loss.data[0]
                total_loss += loss.item()
                res = torch.argmax(output, 2)
                for i in range(trg.size(1)):
                    logger.info(("target: ", self.dataset.vocab.tensorConvertToLabels(trg[1:, i], PAD)))
                    logger.info(("output: ", self.dataset.vocab.tensorConvertToLabels(res[1:, i], PAD)))
                    # print("target: ", self.dataset.vocab.tensorConvertToLabels(trg[1:, i], Constants.PAD),
                    #       file=open(self.args.save_ans, 'a'))
                    # print("output: ", self.dataset.vocab.tensorConvertToLabels(res[1:, i], Constants.PAD),
                    #       file=open(self.args.save_ans, 'a'))
                accuracy, n_correct, n_total = get_token_accuracy(trg[1:], res[1:], ignore_index=PAD)
                overall_correct += n_correct.item()
                overall_tot += n_total.item()
                step = idx_b
        return total_loss / step, overall_correct / overall_tot, step

    def train(self, test_dataset=None, saved_model=None):
        total_loss = 0
        best = np.Inf
        if saved_model:
            self.model.load_state_dict(torch.load(saved_model))
        for i in range(self.epoch):
            self.model.train()
            self.model.zero_grad()
            for idx_b, batch in enumerate(self.get_batch()):
                src, trg = batch
                output = self.model(src, trg, teacher_forcing_ratio=self.args.tr)
                loss = F.nll_loss(output[1:].view(-1, self.dataset.vocab.size()),
                                  trg[1:].contiguous().view(-1), ignore_index=PAD)
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                total_loss += loss.item()

                if idx_b % 10 == 0 and idx_b != 0:
                    total_loss = total_loss / 10
                    self.writer.add_scalar('train_loss_epoch_{}'.format(i), total_loss, idx_b)
                    total_loss = 0

            if test_dataset is not None:
                start = time.time()
                eval_loss, all_acc, num_sentence = self.evaluate(test_dataset)
                elapsed = time.time() - start
                logger.warning('-' * 10 + 'test set' + '-' * 88 + '\n' +
                               '| epoch {} | {:5d} / {:5d} dataset size | lr {:05.5f} | time {:5.2f} | '
                               'loss {:5.2f} | all_acc {:5.2f}'.format(
                                   i, len(test_dataset), num_sentence, self.lr,
                                   elapsed / 60, eval_loss, all_acc) +
                               '\n' + '-' * 98)

                if eval_loss < best:
                    best = eval_loss
                    torch.save(self.model.state_dict(), self.args.model + '/' + self.args.task
                               + '_loss_' + "%0.2f" % eval_loss + '_epoch_' + str(i)
                               + '_batch_' + str(self.batch) + '.pt')
                else:
                    self.adjust_learning_rate()

            if self.args.test_train:
                start = time.time()
                eval_loss, all_acc, num_sentence = self.evaluate()
                elapsed = time.time() - start
                logger.warning('-' * 10 + 'train set' + '-' * 88 + '\n' +
                               '| epoch {} | {:5d} / {:5d} dataset size | lr {:05.5f} | time {:5.2f} | '
                               'loss {:5.2f} | all_acc {:5.2f}'.format(
                                   i, len(self.dataset), num_sentence, self.lr,
                                   elapsed / 60, eval_loss, all_acc) +
                               '\n' + '-' * 98)


def main(generator):
    trn = generator.dataset
    tst = YelpDataset(args.test_data, vocab)

    trainer = BaseSeqTrainer(generator, trn, generator.device, args)
    if args.save_ans:
        if args.load:
            print(trainer.evaluate(tst, args.load))
        else:
            logger.error("No model parameter configured!")
            exit(0)
    else:
        trainer.train(tst, args.load)


def init_weights(m):
    for name, param in m.named_parameters():
        torch.nn.init.normal_(param.data, mean=0, std=0.01)

if __name__ == '__main__':
    PAD_WORD = '<pad>'
    UNK_WORD = '<unk>'
    EOS_WORD = '<eos>'
    SOS_WORD = '<sos>'
    vocab = Vocab(filename=args.dictionary, data=[PAD_WORD, UNK_WORD, EOS_WORD, SOS_WORD])
    PAD = vocab.getIndex(PAD_WORD)

    device = torch.device("cuda:0" if args.cuda else "cpu")
    print('Loading word vectors from', args.word_vector)
    embed = torch.load(args.word_vector)

    trn = YelpDataset(args.train_data, vocab)
    encoder = Encoder(vocab.size(), embed.size(1), args.nhid, args.dropout)
    decoder = Decoder(vocab.size(), embed.size(1), args.nhid, args.dropout)
    generator = Seq2Seq(encoder, decoder, embed=embed, device=device, dataset=trn).to(device)
    generator.apply(init_weights)
    main(generator)
