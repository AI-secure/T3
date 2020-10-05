from __future__ import print_function

import sys

from gensim.models import KeyedVectors

from CW_attack import CarliniL2
from CW_attack_random import CarliniL2_random
from models import *
from my_generator.dataset import YelpDataset
from my_generator.model import Generator, WrappedSeqback
from my_generator.sequential_model import EncoderRNN, Decoder, Seq2SeqGenerator, WrappedSeqDecoder

from util import logger, get_args, root_dir

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import os
from tqdm import tqdm
from vocab import Vocab


def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 1, keepdim=True), 2, keepdim=True).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')


def package(texts, targets):
    """Package data for training / evaluation."""
    maxlen = 0
    for item in texts:
        maxlen = max(maxlen, len(item))
    maxlen = min(maxlen, 500)
    for i in range(len(texts)):
        if maxlen < len(texts[i]):
            texts[i] = texts[i][:maxlen]
        else:
            for j in range(maxlen - len(texts[i])):
                texts[i].append(vocab.getIndex('<pad>'))
    texts = torch.LongTensor(texts)
    targets = torch.LongTensor(targets)
    return texts.t(), targets


def get_next_trade_day(date: str, prices):
    import datetime as dt
    from datetime import datetime
    date = datetime.strptime(date, "%m%d%Y")
    max_date = datetime.strptime('10012018', "%m%d%Y")
    while date not in prices:
        date = date + dt.timedelta(days=1)
        if date >= max_date:
            return None

        dstr = datetime.strftime(date, "%m%d%Y")
        if dstr in prices:
            return dstr


def evaluate(data_val):
    """evaluate the model while training"""
    model.eval()  # turn on the eval() switch to disable dropout
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch, i in enumerate(range(0, len(data_val), args.batch_size)):
            batch_data = data_val[i:i + args.batch_size]
            text = [x['text'] for x in batch_data]
            label = [x['label'] for x in batch_data]
            data, targets = package(text, label)
            if args.cuda:
                data = data.cuda()
                targets = targets.cuda()
            hidden = model.init_hidden(data.size(1))
            output, attention = model.forward(data, hidden)
            output_flat = output.view(data.size(1), -1)
            total_loss += criterion(output_flat, targets).data
            prediction = torch.max(output_flat, 1)[1]
            total_correct += torch.sum((prediction == targets).float())
    return total_loss.item() / (len(data_val) // args.batch_size), total_correct.item() / len(data_val)


def init_attack():
    for param in model.parameters():
        param.requires_grad = False


def get_batch(data_val, has_tree=False):
    for batch, i in enumerate(tqdm(range(0, len(data_val), args.batch_size))):
        batch_data = data_val[i:min(i + args.batch_size, len(data_val))]
        text = [x['text'] for x in batch_data]
        split_text = [x['split_text'] for x in batch_data]
        label = [x['label'] for x in batch_data]
        data, targets = package(text, label)

        if args.cuda:
            data = data.cuda()
            targets = targets.cuda()

        # find the most attention sentence
        model.encoder.raw_inp = None
        model.encoder.bilstm.attack_mode = False
        hidden = model.init_hidden(data.size(1))
        output, attention = model.forward(data, hidden)
        output_flat = output.view(data.size(1), -1)
        prediction = torch.max(output_flat, 1)[1]
        orig_correct = torch.sum((prediction == targets).float())
        attention = torch.sum(attention, 1)

        # append it to the front of the paragraph
        batch_add_start = []
        batch_add_end = []
        batch_add_words = []
        batch_add_start_sent = []
        batch_sentence_id = batch_find_best_attention(split_text, attention)

        if has_tree:
            batch_tree_data = data_val.tree_data[i:min(i + args.batch_size, len(data_val))]
            batch_trees = []
            batch_masks = []
            for tree_data in batch_tree_data:
                trees = [None for _ in tree_data['tree']]
                masks = [False for _ in tree_data['tree']]
                batch_trees.append(trees)
                batch_masks.append(masks)
            split_text = [x['words'] for x in batch_tree_data]

        for bi, (sentence_id, paragraph) in enumerate(zip(batch_sentence_id, text)):
            # mid = len(split_text[bi]) // 2
            mid = 0
            # mid = len(split_text[bi])
            batch_add_start_sent.append(mid)
            add_start = 0
            for sent_id in range(mid):
                add_start += len(split_text[bi][sent_id])
            add_start = min(add_start, 500)
            add_end = min(add_start + len(split_text[bi][sentence_id]), 500)
            if has_tree:
                tree = batch_tree_data[bi]['tree'][sentence_id]
                while tree is None:
                    sentence_id = (sentence_id + 1) % len(split_text)
                    tree = batch_tree_data[bi]['tree'][sentence_id]
                root = data_val.get_tree(tree)
                batch_trees[bi].insert(mid, root)
                batch_masks[bi].insert(mid, True)
            batch_add_words.append(split_text[bi][sentence_id])
            batch_add_start.append(add_start)
            text[bi] = split_text[bi][sentence_id] + paragraph
            batch_add_end.append(add_end)
            split_text[bi].insert(mid, split_text[bi][sentence_id])

        attack_targets = [4 if x < 2 else 0 for x in label]
        data, attack_targets = package(text, attack_targets)
        if args.cuda:
            data = data.cuda()
            attack_targets = attack_targets.cuda()

        output, attention = model.forward(data, hidden)
        output_flat = output.view(data.size(1), -1)
        prediction = torch.max(output_flat, 1)[1]
        orig_append_correct = torch.sum((prediction == targets).float())

        result = {'data': data, 'targets': targets, 'add_start': batch_add_start, 'add_end': batch_add_end,
                  'sentence_ids': batch_sentence_id, 'text': text, 'split_text': split_text, 'label': label,
                  'add_words': batch_add_words, 'orig_correct': orig_correct,
                  'orig_append_correct': orig_append_correct,
                  'attack_targets': attack_targets, 'mid': batch_add_start_sent}
        if has_tree:
            result['tree'] = batch_trees
            result['mask'] = batch_masks
        yield result


def untargeted_success_rate(predictions, labels):
    tot = 0
    for prediction, label in zip(predictions, labels):
        if prediction.item() < 2 <= label or label < 2 <= prediction.item():
            tot += 1
    return tot


def cw_word_attack(data_val):
    init_attack()
    # fname = "/home/wbx/yelp/vectors.kv"
    fname = "full-vectors.kv"
    if not os.path.isfile(fname):
        embed = model.encoder.bilstm.encoder.weight
        print(len(vocab.idxToLabel), embed.shape[1], file=open(fname, "a"))
        for k, v in vocab.idxToLabel.items():
            vectors = embed[k].cpu().numpy()
            vector = ""
            for x in vectors:
                vector += " " + str(x)
            print(v, vector[1:], file=open(fname, "a"))
    device = torch.device("cuda:0" if args.cuda else "cpu")
    adv_correct = 0
    targeted_success = 0
    untargeted_success = 0
    orig_correct = 0
    orig_append_correct = 0
    tot = 0
    adv_pickle = []

    cw = CarliniL2(debug=args.debugging)
    for batch in get_batch(data_val):
        data = batch['data']
        attack_targets = batch['attack_targets']
        batch_add_start = batch['add_start']
        batch_add_end = batch['add_end']
        text = batch['text']
        split_text = batch['split_text']
        label = batch['label']
        # convert text into embedding and attack in the embedding space
        model.encoder.raw_inp = data
        model.init_hidden(data.size(1))
        model.encoder.bilstm.attack_mode = True
        input_embedding = model.encoder.bilstm.encoder(data)

        cw_mask = np.zeros(input_embedding.shape).astype(np.float32)
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            cw_mask[add_start:add_end, bi] = 1
        cw_mask = torch.from_numpy(cw_mask).float()
        if args.cuda:
            cw_mask = cw_mask.cuda()
            cw.batch_info = batch
            cw.wv = model.encoder.bilstm.encoder.weight

        if args.baseline:
            modifier = torch.randn_like(data, device=device)
            modifier = F.normalize(modifier, p=2, dim=2) * 10
            adv_data = input_embedding + modifier * cw_mask
            adv_data = adv_data.cpu().detach().numpy()
        else:
            cw.mask = cw_mask
            adv_data = cw.run(model, input_embedding, attack_targets)
            # adv_hidden = torch.tensor(adv_data).to(device)

            adv_seq = torch.tensor(data).to(device)
            for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
                if bi in cw.o_best_sent:
                    adv_seq.data[add_start:add_end, bi] = torch.LongTensor(cw.o_best_sent[bi])

            for i in range(len(split_text)):
                adv_pickle.append({
                    'raw_text': vocab.tensorConvertToLabels(adv_seq[:, i]),
                    'label': label[i]
                })
                try:
                    logger.info(("orig:", vocab.convertToLabels(split_text[i][0])))
                    logger.info(("adv:", vocab.convertToLabels(cw.o_best_sent[i])))
                except:
                    continue

            model.encoder.raw_inp = None
            model.encoder.bilstm.attack_mode = False
            output, attention = model(adv_seq)
            output_flat = output.view(data.size(1), -1)
            prediction = torch.max(output_flat, 1)[1]

            targets = batch['targets']
            orig_correct += batch['orig_correct'].item()
            orig_append_correct += batch['orig_append_correct'].item()
            adv_correct += torch.sum((prediction == targets).float()).item()
            targeted_success += torch.sum((prediction == attack_targets).float()).item()
            untargeted_success += untargeted_success_rate(prediction, label)
            tot += len(label)

            logger.info(("orig_correct:", orig_correct))
            logger.info(("orig_append_correct:", orig_append_correct))
            logger.info(("adv_correct:", adv_correct))
            logger.info(("targeted successful rate:", targeted_success))
            logger.info(("untargetd successful rate:", untargeted_success))
            logger.info(("tot:", tot))
            joblib.dump(adv_pickle, root_dir + '/adv_text.pkl')
    logger.info(("orig_correct:", orig_correct / tot))
    logger.info(("adv_correct:", adv_correct / tot))
    logger.info(("orig_append_correct:", orig_append_correct / tot))
    logger.info(("targeted successful rate:", targeted_success / tot))
    logger.info(("untargetd successful rate:", untargeted_success / tot))


# for random token
def add_random_tokens(origin_text, random_num=10):
    """
    inpurt:
        origin_text: list[int] list of idx
        random_num: number of insert tokens
    output:
        new_seq: list[int] list of idx after inserting idxes
        new_seq_len: int
        allow_idx: indexes in new_seq which are allowed to modify
    """
    insert_num = min(len(origin_text) // 2 + 1, random_num)
    insert_idx = sorted(random.sample(range(min(500, len(origin_text))), insert_num))
    insert_idx.append(len(origin_text))
    allow_idx = []
    new_seq = origin_text[:insert_idx[0]]
    for i in range(insert_num):
        if len(new_seq) >= 500:
            break
        allow_idx.append(len(new_seq))
        new_seq.append(vocab.getIndex("the"))
        new_seq.extend(origin_text[insert_idx[i]:insert_idx[i + 1]])
    new_seq = new_seq[:500]
    new_seq_len = len(new_seq)
    return new_seq, new_seq_len, allow_idx

def get_random_word_batch(data_val, has_tree=False):
    for batch, i in enumerate(tqdm(range(0, len(data_val), args.batch_size))):
        batch_data = data_val[i:min(i + args.batch_size, len(data_val))]
        text = [x['text'] for x in batch_data]
        split_text = [x['split_text'] for x in batch_data]
        label = [x['label'] for x in batch_data]
        data, targets = package(text, label)

        if args.cuda:
            data = data.cuda()
            targets = targets.cuda()

        # find the most attention sentence
        model.encoder.raw_inp = None
        model.encoder.bilstm.attack_mode = False
        hidden = model.init_hidden(data.size(1))
        output, attention = model.forward(data, hidden)
        output_flat = output.view(data.size(1), -1)
        prediction = torch.max(output_flat, 1)[1]
        orig_correct = torch.sum((prediction == targets).float())

        # append it to the front of the paragraph
        batch_add_start = []
        batch_add_end = []
        batch_add_words = []
        batch_add_start_sent = []
        batch_sentence_id = []

        new_text = []
        new_text_len = []
        allow_idx = []

        random_texts = [add_random_tokens(x) for x in text]
        for random_text in random_texts:
            new_text.append(random_text[0])
            new_text_len.append(random_text[1])
            allow_idx.append(random_text[2])

        attack_targets = [4 if x < 2 else 0 for x in label]
        data, attack_targets = package(new_text, attack_targets)
        if args.cuda:
            data = data.cuda()
            attack_targets = attack_targets.cuda()

        output, attention = model.forward(data, hidden)
        output_flat = output.view(data.size(1), -1)
        prediction = torch.max(output_flat, 1)[1]
        orig_append_correct = torch.sum((prediction == targets).float())

        result = {'data': data, 'targets': targets, 'add_start': batch_add_start, 'add_end': batch_add_end,
                  'sentence_ids': batch_sentence_id, 'text': new_text, 'split_text': split_text, 'label': label,
                  'add_words': batch_add_words, 'orig_correct': orig_correct, 'new_text_len': new_text_len,
                  'orig_append_correct': orig_append_correct, 'allow_idx': allow_idx,
                  'attack_targets': attack_targets, 'mid': batch_add_start_sent}
        yield result


def cw_rand_words_attack(data_val):
    init_attack()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    adv_correct = 0
    targeted_success = 0
    untargeted_success = 0
    orig_correct = 0
    orig_append_correct = 0
    tot = 0
    adv_pickle = []

    cw = CarliniL2_random(debug=args.debugging)
    for batch in get_random_word_batch(data_val):
        data = batch['data']
        attack_targets = batch['attack_targets']
        batch_add_start = batch['add_start']
        batch_add_end = batch['add_end']
        text = batch['text']
        allow_idxs = batch['allow_idx']
        split_text = batch['split_text']
        label = batch['label']

        # convert text into embedding and attack in the embedding space
        model.encoder.raw_inp = data
        model.init_hidden(data.size(1))
        model.encoder.bilstm.attack_mode = True
        input_embedding = model.encoder.bilstm.encoder(data)

        cw_mask = np.zeros(input_embedding.shape).astype(np.float32)
        for bi, allow_idx in enumerate(allow_idxs):
            cw_mask[np.array(allow_idx), bi] = 1
        cw_mask = torch.from_numpy(cw_mask).float()
        if args.cuda:
            cw_mask = cw_mask.cuda()
            cw.batch_info = batch
            cw.wv = model.encoder.bilstm.encoder.weight

        if args.baseline:
            modifier = torch.randn_like(data, device=device)
            modifier = F.normalize(modifier, p=2, dim=2) * 10
            adv_data = input_embedding + modifier * cw_mask
            adv_data = adv_data.cpu().detach().numpy()
        else:
            cw.mask = cw_mask
            adv_data = cw.run(model, input_embedding, attack_targets)

            adv_seq = torch.tensor(data).to(device)

            for bi, allow_idx in enumerate(allow_idxs):
                if bi in cw.o_best_sent:
                    for i, idx in enumerate(allow_idx):
                        adv_seq.data[idx, bi] = cw.o_best_sent[bi][i]

            for i in range(len(split_text)):
                adv_pickle.append({
                    'raw_text': vocab.tensorConvertToLabels(adv_seq[:, i]),
                    'label': label[i]
                })
                try:
                    logger.info(("orig:", vocab.convertToLabels(split_text[i][0])))
                    logger.info(("adv:", vocab.convertToLabels(cw.o_best_sent[i])))
                except:
                    continue

            model.encoder.raw_inp = None
            model.encoder.bilstm.attack_mode = False
            output, attention = model(adv_seq)
            output_flat = output.view(data.size(1), -1)
            prediction = torch.max(output_flat, 1)[1]

            targets = batch['targets']
            orig_correct += batch['orig_correct'].item()
            orig_append_correct += batch['orig_append_correct'].item()
            adv_correct += torch.sum((prediction == targets).float()).item()
            targeted_success += torch.sum((prediction == attack_targets).float()).item()
            untargeted_success += untargeted_success_rate(prediction, label)
            tot += len(label)

            logger.info(("orig_correct:", orig_correct))
            logger.info(("orig_append_correct:", orig_append_correct))
            logger.info(("adv_correct:", adv_correct))
            logger.info(("targeted successful rate:", targeted_success))
            logger.info(("untargetd successful rate:", untargeted_success))
            logger.info(("tot:", tot))
            joblib.dump(adv_pickle, root_dir + '/adv_text.pkl')
    logger.info(("orig_correct:", orig_correct / tot))
    logger.info(("adv_correct:", adv_correct / tot))
    logger.info(("orig_append_correct:", orig_append_correct / tot))
    logger.info(("targeted successful rate:", targeted_success / tot))
    logger.info(("untargetd successful rate:", untargeted_success / tot))


def cw_seq_attack(data_val):
    init_attack()
    cw = CarliniL2()
    embed = torch.load(args.word_vector)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    encoder = EncoderRNN(vocab, embed.size(1), args.hidden_dim, device)
    decoder = Decoder(embed.size(1), args.hidden_dim, vocab.size(), dropout=0.0)
    generator = Seq2SeqGenerator(encoder, decoder, embed=embed).to(device)
    seqback = WrappedSeqDecoder(decoder, vocab)
    generator.load_state_dict(torch.load(args.load_ae))

    adv_correct = 0
    targeted_success = 0
    untargeted_success = 0
    orig_correct = 0
    tot = 0

    def get_seq_hidden(batch_add_words):
        # get lstm hidden embedding
        encoder_output, hidden = encoder(batch_add_words)
        hidden = torch.stack(hidden).transpose_(0, 2).detach()
        encoder_output = encoder_output.detach()
        return encoder_output, hidden

    class SeqModel(nn.Module):
        def __init__(self):
            super(SeqModel, self).__init__()

        def forward(self, hidden):
            embedding = seqback(hidden)
            return model(embedding)

    seq_model = SeqModel()
    for batch in get_batch(data_val):
        batch_add_words = batch['add_words']
        sos = torch.tensor([vocab.getIndex(SOS_WORD)], dtype=torch.long)
        eos = torch.tensor([vocab.getIndex(EOS_WORD)], dtype=torch.long)
        for i, sentence in enumerate(batch_add_words):
            sentence = torch.tensor(sentence)
            sentence = torch.cat((sos, sentence, eos), 0)
            sentence = sentence.to(device)
            batch_add_words[i] = sentence
        from torch.nn.utils.rnn import pad_sequence
        batch_add_words = pad_sequence(batch_add_words, padding_value=vocab.getIndex(PAD_WORD))
        encoder_output, hidden = get_seq_hidden(batch_add_words)

        seqback.trg = batch_add_words
        seqback.encoder_output = encoder_output
        seqback.start = batch['add_start']
        seqback.end = batch['add_end']
        seqback.sentences = batch['data']
        seqback.adv_sent = []

        data = batch['data']
        model.encoder.raw_inp = batch['data']
        model.init_hidden(data.size(1))
        model.encoder.bilstm.attack_mode = True

        if args.baseline:
            modifier = torch.randn_like(hidden, device=device)
            modifier = F.normalize(modifier, p=2, dim=3) * 1e2
            adv_hidden = hidden + modifier
        else:
            adv_hidden = cw.run(seq_model, hidden, batch['attack_targets'], batch_size=hidden.shape[0])
            adv_hidden = torch.tensor(adv_hidden).to(device)

        seqback.adv_sent = []
        output, attention = seq_model(adv_hidden)

        output_flat = output.view(data.size(1), -1)
        prediction = torch.max(output_flat, 1)[1]

        orig_correct += batch['orig_correct'].item()
        adv_correct += torch.sum((prediction == batch['targets']).float()).item()
        targeted_success += torch.sum((prediction == batch['attack_targets']).float()).item()
        untargeted_success += untargeted_success_rate(prediction, batch['label'])
        tot += len(batch['label'])

        for adv, orig in zip(seqback.adv_sent, batch['add_words']):
            print("orig:", vocab.tensorConvertToLabels(orig[1:], vocab.getIndex(PAD_WORD))[:-1], file=adv_sent_file)
            print("adv:", adv[:-1], file=adv_sent_file)

        print("orig_correct:", orig_correct)
        print("adv_correct:", adv_correct)
        print("targeted successful rate:", targeted_success)
        print("untargetd successful rate:", untargeted_success)
        print("tot:", tot)

    print("orig_correct:", orig_correct / tot)
    print("adv_correct:", adv_correct / tot)
    print("targeted successful rate:", targeted_success / tot)
    print("untargetd successful rate:", untargeted_success / tot)


def cw_tree_attack(data_val):
    init_attack()
    cw = CarliniL2()
    embed = torch.load(args.word_vector)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    generator = Generator(args.test_data, vocab=vocab, embed=embed, data_set=data_val)
    seqback = WrappedSeqback(embed, device, attack=True, seqback_model=generator.seqback_model, vocab=vocab,
                             transfer_emb=model.encoder.bilstm.encoder)
    treelstm = generator.tree_model
    generator.load_state_dict(torch.load(args.load_ae))

    adv_correct = 0
    targeted_success = 0
    untargeted_success = 0
    orig_append_correct = 0
    orig_correct = 0
    tot = 0
    adv_pickle = []

    class TreeModel(nn.Module):
        def __init__(self):
            super(TreeModel, self).__init__()

        def forward(self, hidden):
            self.embedding = seqback(hidden)
            return model(self.embedding)

        def set_temp(self, temp):
            seqback.temp = temp

        def get_embedding(self):
            return self.embedding

        def get_seqback(self):
            return seqback

    tree_model = TreeModel()
    for batch in get_batch(data_val, has_tree=True):
        seqback.sentences = batch['data']
        seqback.batch_trees = batch['tree']
        seqback.batch_masks = batch['mask']
        seqback.batch_splitted_sentences = batch['split_text']
        seqback.start = batch['add_start']
        seqback.end = batch['add_end']
        batch_add_start = batch['add_start']
        batch_add_end = batch['add_end']
        seqback.adv_sent = []
        batch_tree_embedding = []

        for bi, split_text in enumerate(batch['split_text']):
            # todo: default use the embedding of front???
            batch['split_text'][bi] = [torch.tensor(x, dtype=torch.long, device=device) for x in split_text]
            sentences = [batch['split_text'][bi][0]]
            trees = [batch['tree'][bi][0]]
            masks = [batch['mask'][bi][0]]
            tree_embedding = treelstm(sentences, trees, masks)[0][0].detach()
            batch_tree_embedding.append(tree_embedding)

        hidden = torch.cat(batch_tree_embedding, dim=0)
        data = batch['data']
        model.encoder.raw_inp = batch['data']
        model.init_hidden(data.size(1))
        model.encoder.bilstm.attack_mode = True
        input_embedding = model.encoder.bilstm.encoder(data)

        # np.save('tree_attack/input.npy', input_token.cpu().numpy())

        if args.baseline:
            modifier = torch.randn_like(hidden, device=device)
            modifier = F.normalize(modifier, p=2, dim=1) * 1e2
            adv_hidden = hidden + modifier
        else:
            with torch.autograd.detect_anomaly():
                adv_hidden = cw.run(tree_model, hidden, batch['attack_targets'], batch_size=hidden.shape[0],
                                    input_token=input_embedding)

        seqback.adv_sent = []

        adv_seq = torch.tensor(data).to(device)
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            if bi in cw.o_best_sent:
                adv_seq[add_start:add_end, bi] = cw.o_best_sent[bi]

        for i in range(len(batch['label'])):
            adv_pickle.append({
                'raw_text': vocab.tensorConvertToLabels(adv_seq[:, i]),
                'label': batch['label'][i]
            })
            try:
                logger.info(("orig:", vocab.convertToLabels(batch['add_words'][i])))
                logger.info(("adv:", vocab.tensorConvertToLabels(cw.o_best_sent[i])))
            except:
                continue

        model.encoder.raw_inp = None
        model.encoder.bilstm.attack_mode = False
        output, attention = model(adv_seq)
        output_flat = output.view(data.size(1), -1)
        prediction = torch.max(output_flat, 1)[1]

        orig_correct += batch['orig_correct'].item()
        orig_append_correct += batch['orig_append_correct'].item()
        adv_correct += torch.sum((prediction == batch['targets']).float()).item()
        targeted_success += torch.sum((prediction == batch['attack_targets']).float()).item()
        untargeted_success += untargeted_success_rate(prediction, batch['label'])
        tot += len(batch['label'])

        logger.info(("orig_correct:", orig_correct))
        logger.info(("orig_append_correct:", orig_append_correct))
        logger.info(("adv_correct:", adv_correct))
        logger.info(("targeted successful rate:", targeted_success))
        logger.info(("untargetd successful rate:", untargeted_success))
        logger.info(("tot:", tot))
        joblib.dump(adv_pickle, root_dir + '/adv_text.pkl')
    logger.info(("orig_correct:", orig_correct / tot))
    logger.info(("orig_append_correct:", orig_append_correct / tot))
    logger.info(("adv_correct:", adv_correct / tot))
    logger.info(("targeted successful rate:", targeted_success / tot))
    logger.info(("untargetd successful rate:", untargeted_success / tot))


def toHex(n):
    n = int(n)
    n = 255 - n
    return hex(n)[2:].zfill(2).upper()


def visualize(data_val):
    """evaluate the model while training"""
    model.eval()  # turn on the eval() switch to disable dropout
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch, i in enumerate(range(0, len(data_val), args.batch_size)):
            batch_data = data_val[i:i + args.batch_size]
            text = [x['text'] for x in batch_data]
            label = [x['label'] for x in batch_data]
            data, targets = package(text, label)
            if args.cuda:
                data = data.cuda()
                targets = targets.cuda()
            hidden = model.init_hidden(data.size(1))
            output, attention = model.forward(data, hidden)
            output_flat = output.view(data.size(1), -1)
            total_loss += criterion(output_flat, targets).data
            prediction = torch.max(output_flat, 1)[1]
            total_correct += torch.sum((prediction == targets).float())

            attention = torch.sum(attention, 1)
            html = str()
            for bi in range(attention.size(0)):
                att = attention[bi]
                att_max = torch.max(att)
                att_min = torch.min(att)
                norm_att = (att - att_min) / (att_max - att_min).tolist()
                print(norm_att)
                html += """Stars: %i </br>""" % (targets[bi].item() + 1)
                sentence = vocab.convertToLabels(text[bi], -1)
                for si, word in enumerate(sentence):
                    html += """<span style="background-color: #FF%s%s">%s </span>""" % (
                        toHex(norm_att[si] * 255), toHex(norm_att[si] * 255), word)
                html += """</br></br>"""
            with open('viz.html', 'w') as fout:
                fout.write(html)
            exit(0)
    return total_loss.item() / (len(data_val) // args.batch_size), total_correct.item() / len(data_val)


def batch_find_best_attention(batch_split_text, batch_attention):
    """
    :param batch_split_text: batch split text
    :param batch_attention: batch attention
    :return:
    """
    batch_idx = []
    for i, split_text in enumerate(batch_split_text):
        att = batch_attention[i]
        att_max = torch.max(att)
        att_min = torch.min(att)
        norm_att = (att - att_min) / (att_max - att_min).tolist()
        batch_idx.append(find_best_attention(split_text, norm_att))
    return batch_idx


def find_best_attention(split_text, attention):
    """
    :param split_text: a paragraph of split text
    :param attention: normized attention
    :return:
    """
    idx = 0
    sentence_attention = []
    for sentence in split_text:
        att = 0
        for _ in sentence:
            if idx < 500:
                att += attention[idx]
                idx += 1
        # does not consider sentences less than 2 words
        if len(sentence) > 2:
            att /= len(sentence)
        else:
            att = 0
        sentence_attention.append(att)
    return np.argmax(sentence_attention)


if __name__ == '__main__':
    # parse the arguments
    args = get_args()
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    # Load Dictionary
    assert os.path.exists(args.train_data)
    assert os.path.exists(args.val_data)
    print('Begin to load the dictionary.')

    PAD_WORD = '<pad>'
    UNK_WORD = '<unk>'
    EOS_WORD = '<eos>'
    SOS_WORD = '<sos>'
    vocab = Vocab(filename=args.dictionary, data=[PAD_WORD, UNK_WORD, EOS_WORD, SOS_WORD])

    best_val_loss = None
    best_acc = None

    n_token = vocab.size()
    model = Classifier({
        'dropout': args.dropout,
        'ntoken': n_token,
        'nlayers': args.nlayers,
        'nhid': args.nhid,
        'ninp': args.emsize,
        'pooling': 'all',
        'attention-unit': args.attention_unit,
        'attention-hops': args.attention_hops,
        'nfc': args.nfc,
        'vocab': vocab,
        'word-vector': args.word_vector,
        'class-number': args.class_number
    })
    if args.cuda:
        model = model.cuda()

    print(args)
    I = torch.zeros(args.batch_size, args.attention_hops, args.attention_hops)
    for i in range(args.batch_size):
        for j in range(args.attention_hops):
            I.data[i][j][j] = 1
    if args.cuda:
        I = I.cuda()

    criterion = nn.CrossEntropyLoss()
    print('Begin to load data.')
    import joblib

    if args.load:
        model.load_state_dict(torch.load(args.load))
        if args.cuda:
            model = model.cuda()

        print('-' * 89)
        if args.model == 'tree':
            data_val = YelpDataset(args.test_data, vocab, args.test_tree)
            cw_tree_attack(data_val)
        elif args.model == 'random_attack':
            data_val = YelpDataset(args.test_data, vocab)
            print("random attack!")
            cw_rand_words_attack(data_val)
        else:
            data_val = YelpDataset(args.test_data, vocab)
            cw_word_attack(data_val)
