import time
import logging
from tqdm import tqdm
from my_generator.dataset import YelpDataset
from my_generator.model import Generator
import numpy as np
import torch
import torch.nn.functional as F

from util import get_args
from vocab import Vocab

logger = logging.getLogger(__name__)


class GeneratorTrainer:
    def __init__(self, model: Generator, dataset: YelpDataset, device, args):
        self.model = model
        self.dataset = dataset
        self.epoch = args.epochs
        self.device = device
        self.criterion = self.model.seqback_criterion
        self.lr = args.lr
        self.optimizer = self.model.optimizer
        self.batch = args.batch_size
        self.args = args

        for name, param in self.model.named_parameters():
            # print(type(param.data), param.size())
            print(name, type(param.data), param.size())

    def adjust_learning_rate(self):
        """Sets the learning rate to the initial LR decayed by 4 when eval result drops"""
        self.lr = self.lr / 4
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def get_words(self, idxs):
        res = []
        idxs = idxs
        for i in idxs:
            res.append(self.dataset.vocab.convertToLabels(i, -100))
        return res

    def get_batch(self, dataset=None):
        batch_sentences = []
        batch_targets = []
        batch_trees = []
        if dataset is None:
            # train time
            dataset = self.dataset
            dataset_idx = np.random.choice(len(dataset), len(dataset)//20, replace=False)
        else:
            # test time
            dataset_idx = range(len(dataset))
        for j in tqdm(dataset_idx):
            sentences = dataset.tree_data[j]['words']
            raw_trees = dataset.tree_data[j]['tree']
            trees = []
            for sent in raw_trees:
                if sent is None:
                    trees.append(None)
                else:
                    root = self.dataset.get_tree(sent)
                    trees.append(root)
            # trained in sentence level
            for sentence, tree in zip(sentences, trees):
                if not tree:
                    # no tree data
                    continue
                if len(sentence) > 100:
                    continue
                sentence = torch.tensor(sentence, dtype=torch.long, device=self.device)
                target = sentence

                batch_sentences.append(sentence)
                batch_targets.append(target)
                batch_trees.append(tree)

                if len(batch_targets) == self.batch:
                    yield batch_sentences, batch_targets, batch_trees
                    batch_targets = []
                    batch_sentences = []
                    batch_trees = []
        if len(batch_sentences) > 0:
            yield batch_sentences, batch_targets, batch_trees

    def evaluate(self, test_dataset=None, saved_model=None):
        self.model.eval()
        self.model.teacher_forcing_ratio = 0
        loss = 0
        acc = []
        overall_correct = 0
        overall_tot = 0
        step = 0

        if not test_dataset:
            dataset = self.dataset
        else:
            dataset = test_dataset

        if saved_model:
            self.model.load_state_dict(torch.load(saved_model))

        # overwrites the file
        if self.args.save_ans:
            with open(self.args.save_ans, 'w') as f:
                f.write('start dump \n')

        with torch.no_grad():
            for idx_b, batch in enumerate(self.get_batch(dataset)):
                batch_sentences, batch_targets, batch_trees = batch
                self.model.seqback_model.sentences = batch_sentences
                outputs = self.model(batch_sentences, batch_trees)
                for i in range(len(outputs)):
                    step += 1
                    output = outputs[i]
                    target = batch_targets[i]
                    # target = np.array(target)
                    # target = torch.from_numpy(target).long().to(self.device)
                    loss += self.criterion(output, target).item()
                    res = torch.argmax(F.softmax(output, dim=1), 1)

                    # if self.args.save_ans is not None:
                    #     with open(self.args.save_ans, 'a') as f:
                    output_label = self.dataset.vocab.tensorConvertToLabels(res, -100)
                    target_label = self.dataset.vocab.tensorConvertToLabels(target, -100)
                    print("output: ", output_label)
                    print("target: ", target_label)

                    # f.write("output: " + str(output_label) + "\n")
                    # f.write("target: " + str(target_label) + "\n")
                    # f.write("output: " + str(res) + "\n")
                    # f.write("target: " + str(target) + "\n")
                    correct = (res == target).sum().item()
                    overall_correct += correct
                    overall_tot += res.size(0)
                    acc.append(correct / res.size(0))
        print(loss / step, np.mean(acc), overall_correct / overall_tot, step)
        return loss / step, np.mean(acc), overall_correct / overall_tot, step

    # data refers to a sentence and corresponding properties
    def train(self, test_dataset=None, saved_model=None):
        step = 0
        loss = 0
        best = np.Inf
        if saved_model:
            self.model.load_state_dict(torch.load(saved_model))
        for i in range(self.epoch):
            self.model.train()
            self.model.zero_grad()
            self.model.teacher_forcing_ratio = args.tr
            for idx_b, batch in enumerate(self.get_batch()):
                batch_sentences, batch_targets, batch_trees = batch
                self.model.seqback_model.sentences = batch_sentences
                outputs = self.model(batch_sentences, batch_trees)

                for j in range(len(outputs)):
                    step += 1
                    output = outputs[j]
                    target = batch_targets[j]

                    loss += self.criterion(output, target)

                loss.backward()
                logger.info("LOSS " + str(loss))
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss = 0

            if test_dataset is not None:
                start = time.time()
                eval_loss, avg_acc, all_acc, num_sentence = self.evaluate(test_dataset)
                elapsed = time.time() - start
                logger.warning('-' * 10 + 'test set' + '-' * 88 + '\n' +
                               '| epoch {} | {:5d} / {:5d} dataset size | lr {:05.5f} | time {:5.2f} | '
                               'loss {:5.2f} | avg_acc {:5.2f} | all_acc {:5.2f}'.format(
                                   i, len(test_dataset), num_sentence, self.lr,
                                   elapsed / 60, eval_loss, avg_acc, all_acc) +
                               '\n' + '-' * 98)

                if eval_loss < best:
                    best = eval_loss
                    torch.save(self.model.state_dict(), self.args.model + '/' + self.args.save
                               + '_loss_' + "%0.2f" % eval_loss + '_epoch_' + str(i)
                               + '_batch_' + str(self.batch) + '.pt')

                else:
                    self.adjust_learning_rate()
                if (i + 1) % 10 == 0:
                    self.adjust_learning_rate()

            # if self.args.test_train:
            #     start = time.time()
            #     eval_loss, avg_acc, all_acc, num_sentence = self.evaluate()
            #     elapsed = time.time() - start
            #     logger.warning('-' * 10 + 'train set' + '-' * 88 + '\n' +
            #                    '| epoch {} | {:5d} / {:5d} dataset size | lr {:05.5f} | time {:5.2f} | '
            #                    'loss {:5.2f} | avg_acc {:5.2f} | all_acc {:5.2f}'.format(
            #                        i, len(self.dataset), num_sentence, self.lr,
            #                        elapsed / 60, eval_loss, avg_acc, all_acc) +
            #                    '\n' + '-' * 98)


def main(generator):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("{0}/{1}.log".format("log", args.logger)),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info('logger start')
    g = generator
    dataset = g.data_set
    test_dataset = YelpDataset(args.test_data, dataset.vocab, args.test_tree)
    logger.info("start network")
    trainer = GeneratorTrainer(g, dataset, g.device, args)
    if args.save_ans:
        if args.load:
            trainer.evaluate(test_dataset, args.load)
        else:
            logger.error("No model parameter configured!")
            exit(0)
    else:
        trainer.train(test_dataset, args.load)


if __name__ == '__main__':
    args = get_args()

    PAD_WORD = '<pad>'
    UNK_WORD = '<unk>'
    EOS_WORD = '<eos>'
    SOS_WORD = '<sos>'
    vocab = Vocab(filename=args.dictionary, data=[PAD_WORD, UNK_WORD, EOS_WORD, SOS_WORD])

    PAD = vocab.getIndex(PAD_WORD)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    print('Loading word vectors from', args.word_vector)
    embed = torch.load(args.word_vector)

    trn = YelpDataset(args.train_data, vocab, args.train_tree)
    generator = Generator(args.train_data, vocab, embed, trn)
    main(generator)
