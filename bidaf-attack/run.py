import argparse
import copy, json, os
import random

import joblib
import nltk
import numpy as np
import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
from time import gmtime, strftime

from tqdm import tqdm

from CW_QA_attack import CarliniL2_qa
from model.model import BiDAF
from model.data import SQuAD
from model.ema import EMA
import evaluate
import errno, logging, sys
import time
import pickle

from my_generator.model import Generator, WrappedSeqback
from util import args, logger, root_dir, UNK_WORD, PAD_WORD, EOS_WORD, SOS_WORD
from vocab import Vocab


def train(options, data):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, lr=options.learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    loss, last_epoch = 0, -1
    max_dev_exact, max_dev_f1 = -1, -1

    iterator = data.train_iter
    print(len(iterator))
    for epoch in range(options.epoch):
        logger.info('epoch: {}'.format(epoch + 1))
        t1 = time.time()
        loss = 0
        model.train()
        for i, batch in enumerate(iterator):
            model.zero_grad()
            p1, p2 = model(batch)
            batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
            loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()

            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema.update(name, param.data)

        t2 = time.time() - t1
        dev_loss, dev_exact, dev_f1 = test(model, ema, options, data)
        c = (i + 1) // options.print_freq
        logger.info('{}: train loss: {:.3f} dev loss: {:.3f} dev EM: {:.3f} dev F1: {:.3f}'.format(t2, loss, dev_loss,
                                                                                                   dev_exact, dev_f1))

        if dev_f1 > max_dev_f1:
            max_dev_f1 = dev_f1
            max_dev_exact = dev_exact
            torch.save(model.state_dict(), best_model_file_name)
            with open(best_ema, "wb") as f:
                pickle.dump(ema, f)

            logger.info("new best score")

    logger.info('max dev EM: {:.3f} / max dev F1: {:.3f}'.format(max_dev_exact, max_dev_f1))
    if options.epoch > 0:
        model.load_state_dict(torch.load(best_model_file_name, map_location="cuda:{}".format(options.gpu)))
    dev_loss, dev_exact, dev_f1 = test(model, ema, options, data)
    logger.info('final: dev loss: {:.3f} dev EM: {:.3f} dev F1: {:.3f}'.format(dev_loss, dev_exact, dev_f1))


def test(model, ema, options, data):
    device = torch.device("cuda:{}".format(options.gpu) if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    loss = 0
    answers = dict()
    model.eval()

    backup_params = EMA(0)
    for name, param in model.named_parameters():
        if param.requires_grad:
            backup_params.register(name, param.data)
            param.data.copy_(ema.get(name))
    tot = 0
    with torch.no_grad():
        for batch in tqdm(iter(data.dev_iter)):

            p1, p2 = model(batch)
            batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
            loss += batch_loss.item()
            batch_size, c_len = p1.size()
            ls = nn.LogSoftmax(dim=1)
            mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1,
                                                                                                      -1)
            score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
            score, s_idx = score.max(dim=1)
            score, e_idx = score.max(dim=1)
            s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze(-1)

            for i in range(batch_size):
                id = batch.id[i]
                tot += 1
                answer = batch.c_word[0][i][s_idx[i]:e_idx[i] + 1]
                answer = ' '.join([data.WORD.vocab.itos[idx] for idx in answer])
                answers[id] = answer
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(backup_params.get(name))

    with open(options.prediction_file, 'w', encoding='utf-8') as f:
        print(json.dumps(answers), file=f)

    results = evaluate.main(options)
    print(tot)
    return loss, results['exact_match'], results['f1']


def parse_sentence(sentence):
    """
    :param sentence: str
    :return: triples, words
    """
    parser = nltk.CoreNLPDependencyParser(url='http://localhost:9000')

    parse = parser.raw_parse(sentence, properties={
        'tokenize.options': 'ptb3Escaping=false, normalizeFractions=false'})

    parse_sents = list(parse)
    assert len(parse_sents) == 1, "More than 1 sentence extracted"
    parse_graph = list(parse_sents[0].nodes.values())

    parse_graph.sort(key=lambda x: x["address"])
    parse_graph = parse_graph[1:]
    words = [x["word"] for x in parse_graph]

    triples = []
    for k in parse_graph:
        if k["head"] is None:
            continue
        elif k["head"] == 0:
            triples.append((("ROOT", k["head"] - 1), (words[k["address"] - 1], k["address"] - 1), k["rel"]))
        else:
            triples.append(
                ((words[k["head"] - 1], k["head"] - 1), (words[k["address"] - 1], k["address"] - 1),
                 k["rel"]))
    return triples, words


def generate_tree_info(append_info):
    sent = append_info['append_sent']
    triples, words = parse_sentence(' '.join(sent))
    append_info['append_sent'] = words
    append_info['tree'] = triples
    return append_info


def bidaf_convert_to_idx(sent):
    words = []
    for word in sent:
        words.append(data.WORD.vocab.stoi[word.lower()])
    return words


def transform(idxs):
    words = []
    for idx in idxs:
        words.append(data.WORD.vocab.itos[idx])
    return words


def append_input(batch, vocab=None):
    """
    :param batch: batch data
    :param vocab: give ae vocab if have one
    :return:
    """
    append_input = torch.tensor(batch.c_word[0])
    length = append_input.size(1)
    assert append_input.size(0) == 1, "Only support batch size = 1"
    example_id = batch.id[0]
    ans_append = answer_append_sentences[example_id]
    qas_append = question_append_sentences[example_id]

    append_info = qas_append.copy() if qas_append['append_sent'] is not None else ans_append.copy()
    if vocab is not None:
        append_info = generate_tree_info(append_info)
        append_info['ae_sent'] = vocab.convertToIdx(append_info['append_sent'], UNK_WORD)
        append_info['ae_sent'] = [append_info['ae_sent']]
    try:
        append_info['append_sent'] = bidaf_convert_to_idx(append_info['append_sent'])
    except Exception as e:
        print(e)
        print(append_info['append_sent'])
        print(example_id)
        print(ans_append)
        print(qas_append)
    x = append_info['append_sent']
    append_info['add_start'] = length
    append_info['add_end'] = length + len(x)

    concat_input = torch.unsqueeze(torch.LongTensor(x).cuda(), dim=0)
    append_input = torch.cat((append_input, concat_input), dim=1)
    batch_length = torch.tensor(batch.c_word[1])
    batch_length[0] += len(x)
    batch.c_word = (append_input, batch_length)
    # for append to end
    append_info['tot_length'] = append_info['add_end']
    append_info['add_start'] = [append_info['add_start']]
    append_info['add_end'] = [append_info['add_end']]
    append_info['target_start'] = [append_info['target_start'] + length]
    append_info['target_end'] = [append_info['target_end'] + length]
    append_info['append_sent'] = [append_info['append_sent']]
    return append_info


def compare(start_output, start_target, end_output, end_target, targeted=True):
    if targeted:
        return start_output == start_target and end_output == end_target
    else:
        # non overlepping
        return (start_output > start_target and end_output > start_target) or \
               (start_output < end_target and end_target < end_target)


def compare_untargeted(start_output, start_target, end_output, end_target, targeted=True):
    if targeted:
        return abs(start_output - start_target) < 10 or abs(end_output - end_target) < 10
    else:
        return start_output != start_target and end_output != end_target


def write_to_ans(p1, p2, batch, answers):
    # (batch, c_len, c_len)
    batch_size, c_len = p1.size()
    ls = nn.LogSoftmax(dim=1)
    mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1,
                                                                                              -1)
    score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
    score, s_idx = score.max(dim=1)
    score, e_idx = score.max(dim=1)
    s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze(-1)

    for i in range(batch_size):
        id = batch.id[i]
        answer = batch.c_word[0][i][s_idx[i]:e_idx[i] + 1]
        answer = ' '.join([data.WORD.vocab.itos[idx] for idx in answer])
        answers[id] = answer
        # work only for batch size = 1
        return answer, s_idx[i], e_idx[i]


def cw_word_attack():
    cw = CarliniL2_qa(debug=args.debugging)
    criterion = nn.CrossEntropyLoss()
    loss = 0
    adv_loss = 0
    targeted_success = 0
    untargeted_success = 0
    adv_text = []
    answers = dict()
    adv_answers = dict()

    backup_params = EMA(0)
    for name, param in model.named_parameters():
        if param.requires_grad:
            backup_params.register(name, param.data)
            param.data.copy_(ema.get(name))
    tot = 0
    for batch in tqdm(iter(data.dev_iter), total=1000):
        p1, p2 = model(batch)
        orig_answer, orig_s_idx, orig_e_idx = write_to_ans(p1, p2, batch, answers)
        batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
        loss += batch_loss.item()

        append_info = append_input(batch)
        batch_add_start = append_info['add_start']
        batch_add_end = append_info['add_end']
        batch_start_target = torch.LongTensor(append_info['target_start']).to(device)
        batch_end_target = torch.LongTensor(append_info['target_end']).to(device)
        add_sents = append_info['append_sent']

        input_embedding = model.word_emb(batch.c_word[0])
        cw_mask = np.zeros(input_embedding.shape).astype(np.float32)
        cw_mask = torch.from_numpy(cw_mask).float().to(device)
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            cw_mask[bi, add_start:add_end] = 1
        cw.wv = model.word_emb.weight
        cw.inputs = batch
        cw.mask = cw_mask
        cw.batch_info = append_info
        print(cw.wv.size())
        cw.num_classes = append_info['tot_length']
        cw.run(model, input_embedding, (batch_start_target, batch_end_target))

        # re-test
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            if bi in cw.o_best_sent:
                batch.c_word[0].data[bi, add_start:add_end] = torch.LongTensor(cw.o_best_sent[bi])
        p1, p2 = model(batch)
        adv_answer, adv_s_idx, adv_e_idx = write_to_ans(p1, p2, batch, adv_answers)
        batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
        adv_loss += batch_loss.item()

        for bi, (start_target, end_target) in enumerate(zip(batch_start_target, batch_end_target)):
            start_output = adv_s_idx
            end_output = adv_e_idx
            targeted_success += int(compare(start_output, start_target.item(), end_output, end_target.item()))
            untargeted_success += int(
                compare_untargeted(start_output, start_target.item(), end_output, end_target.item()))

        for i in range(len(add_sents)):
            logger.info(("orig:", transform(add_sents[i])))
            try:
                logger.info(("adv:", transform(cw.o_best_sent[i])))
                adv_text.append({'adv_text': transform(cw.o_best_sent[i]),
                                 'qas_id': batch.id[i],
                                 'adv_predict': (orig_s_idx, orig_e_idx),
                                 'orig_predict': (adv_s_idx, adv_e_idx),
                                 'Orig answer:': orig_answer,
                                 'Adv answer:': adv_answer
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')
            except:
                adv_text.append({'adv_text': transform(add_sents[i]),
                                 'qas_id': batch.id[i],
                                 'adv_predict': (orig_s_idx, orig_e_idx),
                                 'orig_predict': (adv_s_idx, adv_e_idx),
                                 'Orig answer:': orig_answer,
                                 'Adv answer:': adv_answer
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')
                continue
        # for batch size = 1
        tot += 1
        logger.info(("orig predict", (orig_s_idx, orig_e_idx)))
        logger.info(("adv append predict", (adv_s_idx, adv_e_idx)))
        logger.info(("targeted successful rate:", targeted_success))
        logger.info(("untargetd successful rate:", untargeted_success))
        logger.info(("Orig answer:", orig_answer))
        logger.info(("Adv answer:", adv_answer))
        logger.info(("tot:", tot))

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data.copy_(backup_params.get(name))

    with open(options.prediction_file, 'w', encoding='utf-8') as f:
        print(json.dumps(answers), file=f)
    with open(options.prediction_file + '_adv.json', 'w', encoding='utf-8') as f:
        print(json.dumps(adv_answers), file=f)
    results = evaluate.main(options)
    logger.info(tot)
    logger.info(("adv loss, results['exact_match'], results['f1']", loss, results['exact_match'], results['f1']))
    return loss, results['exact_match'], results['f1']


def cw_word_attack_target():
    cw = CarliniL2_qa(debug=args.debugging)
    criterion = nn.CrossEntropyLoss()
    loss = 0
    adv_loss = 0
    targeted_success = 0
    untargeted_success = 0
    adv_text = []
    answers = dict()
    adv_answers = dict()

    backup_params = EMA(0)
    for name, param in model.named_parameters():
        if param.requires_grad:
            backup_params.register(name, param.data)
            param.data.copy_(ema.get(name))
    tot = 0
    for batch in tqdm(iter(data.dev_iter), total=1000):
        p1, p2 = model(batch)
        orig_answer, orig_s_idx, orig_e_idx = write_to_ans(p1, p2, batch, answers)
        batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
        loss += batch_loss.item()

        append_info = append_input(batch)
        batch_add_start = append_info['add_start']
        batch_add_end = append_info['add_end']
        batch_start_target = torch.LongTensor(append_info['target_start']).to(device)
        batch_end_target = torch.LongTensor(append_info['target_end']).to(device)
        add_sents = append_info['append_sent']

        input_embedding = model.word_emb(batch.c_word[0])
        cw_mask = np.zeros(input_embedding.shape).astype(np.float32)
        cw_mask = torch.from_numpy(cw_mask).float().to(device)
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            cw_mask[bi, add_start:add_end] = 1

        for bi, (add_start, add_end) in enumerate(zip(append_info['target_start'], append_info['target_end'])):
            # unmask the target
            cw_mask[bi, add_start:add_end + 1] = 0

        cw.wv = model.word_emb.weight
        cw.inputs = batch
        cw.mask = cw_mask
        cw.batch_info = append_info
        print(cw.wv.size())
        cw.num_classes = append_info['tot_length']
        cw.run(model, input_embedding, (batch_start_target, batch_end_target))

        # re-test
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            if bi in cw.o_best_sent:
                batch.c_word[0].data[bi, add_start:add_end] = torch.LongTensor(cw.o_best_sent[bi])
        p1, p2 = model(batch)
        adv_answer, adv_s_idx, adv_e_idx = write_to_ans(p1, p2, batch, adv_answers)
        batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
        adv_loss += batch_loss.item()

        for bi, (start_target, end_target) in enumerate(zip(batch_start_target, batch_end_target)):
            start_output = adv_s_idx
            end_output = adv_e_idx
            targeted_success += int(compare(start_output, start_target.item(), end_output, end_target.item()))
            untargeted_success += int(
                compare_untargeted(start_output, start_target.item(), end_output, end_target.item()))

        for i in range(len(add_sents)):
            logger.info(("orig:", transform(add_sents[i])))
            try:
                logger.info(("adv:", transform(cw.o_best_sent[i])))
                adv_text.append({'adv_text': transform(cw.o_best_sent[i]),
                                 'qas_id': batch.id[i],
                                 'adv_predict': (orig_s_idx, orig_e_idx),
                                 'orig_predict': (adv_s_idx, adv_e_idx),
                                 'Orig answer:': orig_answer,
                                 'Adv answer:': adv_answer
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')
            except:
                adv_text.append({'adv_text': transform(add_sents[i]),
                                 'qas_id': batch.id[i],
                                 'adv_predict': (orig_s_idx, orig_e_idx),
                                 'orig_predict': (adv_s_idx, adv_e_idx),
                                 'Orig answer:': orig_answer,
                                 'Adv answer:': adv_answer
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')
                continue
        # for batch size = 1
        tot += 1
        logger.info(("orig predict", (orig_s_idx, orig_e_idx)))
        logger.info(("adv append predict", (adv_s_idx, adv_e_idx)))
        logger.info(("targeted successful rate:", targeted_success))
        logger.info(("untargetd successful rate:", untargeted_success))
        logger.info(("Orig answer:", orig_answer))
        logger.info(("Adv answer:", adv_answer))
        logger.info(("tot:", tot))

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data.copy_(backup_params.get(name))

    with open(options.prediction_file, 'w', encoding='utf-8') as f:
        print(json.dumps(answers), file=f)
    with open(options.prediction_file + '_adv.json', 'w', encoding='utf-8') as f:
        print(json.dumps(adv_answers), file=f)
    results = evaluate.main(options)
    logger.info(tot)
    logger.info(("adv loss, results['exact_match'], results['f1']", loss, results['exact_match'], results['f1']))
    return loss, results['exact_match'], results['f1']

# for random token
def add_random_tokens(origin_text, target_start, target_end, random_num=10):
    """
    inpurt:
        origin_text: list[int] list of idx
        random_num: number of insert tokens
    output:
        new_seq: list[int] list of idx after inserting idxes
        new_seq_len: int
        allow_idx: indexes in new_seq which are allowed to modify
    """
    insert_num = min(len(origin_text) // 6 + 1, random_num)
    range1 = list(range(target_start))
    range2 = list(range(target_end + 1, len(origin_text)))
    insert_idx = sorted(random.sample((range1 + range2), insert_num))
    insert_idx.append(len(origin_text))
    allow_idx = []
    new_seq = origin_text[:insert_idx[0]]
    new_target_start = target_start
    new_target_end = target_end
    for i in range(insert_num):
        allow_idx.append(len(new_seq))
        new_seq.append(data.WORD.vocab.stoi["the"])
        new_seq.extend(origin_text[insert_idx[i]:insert_idx[i + 1]])

        if insert_idx[i] <= target_start:
            new_target_start += 1
            new_target_end += 1
    new_seq_len = len(new_seq)
    return new_seq, new_seq_len, allow_idx, new_target_start, new_target_end

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def append_random_input(batch):
    """
    :param batch: batch data
    :param vocab: give ae vocab if have one
    :return:
    """
    append_input = torch.tensor(batch.c_word[0])
    assert append_input.size(0) == 1, "Only support batch size = 1"
    example_id = batch.id[0]
    target_start = batch.s_idx[0].item()
    target_end = batch.e_idx[0].item()
    original_text = to_list(append_input[0])
    print(transform(original_text)[target_start:target_end+1])
    new_seq, new_seq_len, allow_idx, target_start, target_end = \
        add_random_tokens(original_text, target_start, target_end)
    append_info = {'new_seq': [new_seq], 'new_seq_len': [new_seq_len], 'allow_idx': [allow_idx],
                   'target_start': [target_start], 'target_end': [target_end]}
    concat_input = torch.unsqueeze(torch.LongTensor(new_seq).cuda(), dim=0)
    batch_length = torch.tensor(batch.c_word[1])
    batch_length[0] = new_seq_len
    batch.c_word = (concat_input, batch_length)
    # for append to end
    append_info['qas_id'] = example_id
    append_info['tot_length'] = new_seq_len
    return append_info


def cw_random_word_attack():
    cw = CarliniL2_untargeted_qa(debug=args.debugging)
    criterion = nn.CrossEntropyLoss()
    loss = 0
    adv_loss = 0
    targeted_success = 0
    untargeted_success = 0
    adv_text = []
    answers = dict()
    adv_answers = dict()

    backup_params = EMA(0)
    for name, param in model.named_parameters():
        if param.requires_grad:
            backup_params.register(name, param.data)
            param.data.copy_(ema.get(name))
    tot = 0
    for batch in tqdm(iter(data.dev_iter), total=1000):
        p1, p2 = model(batch)
        orig_answer, orig_s_idx, orig_e_idx = write_to_ans(p1, p2, batch, answers)
        batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
        loss += batch_loss.item()

        append_info = append_random_input(batch)
        allow_idxs = append_info['allow_idx']
        batch_start_target = torch.LongTensor([0]).to(device)
        batch_end_target = torch.LongTensor([0]).to(device)

        input_embedding = model.word_emb(batch.c_word[0])
        cw_mask = np.zeros(input_embedding.shape).astype(np.float32)
        cw_mask = torch.from_numpy(cw_mask).float().to(device)

        for bi, allow_idx in enumerate(allow_idxs):
            cw_mask[bi, np.array(allow_idx)] = 1
        cw.wv = model.word_emb.weight
        cw.inputs = batch
        cw.mask = cw_mask
        cw.batch_info = append_info
        cw.num_classes = append_info['tot_length']
        # print(transform(to_list(batch.c_word[0][0])))
        cw.run(model, input_embedding, (batch_start_target, batch_end_target))

        # re-test
        for bi, allow_idx in enumerate(allow_idxs):
            if bi in cw.o_best_sent:
                for i, idx in enumerate(allow_idx):
                    batch.c_word[0].data[bi, idx] = cw.o_best_sent[bi][i]
        p1, p2 = model(batch)
        adv_answer, adv_s_idx, adv_e_idx = write_to_ans(p1, p2, batch, adv_answers)
        batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
        adv_loss += batch_loss.item()

        for bi, (start_target, end_target) in enumerate(zip(batch_start_target, batch_end_target)):
            start_output = adv_s_idx
            end_output = adv_e_idx
            targeted_success += int(compare(start_output, start_target.item(), end_output, end_target.item()))
            untargeted_success += int(
                compare_untargeted(start_output, start_target.item(), end_output, end_target.item()))
        for i in range(len(allow_idxs)):
            try:
                logger.info(("adv:", transform(cw.o_best_sent[i])))
                adv_text.append({'added_text': transform(cw.o_best_sent[i]),
                                 'adv_text': transform(to_list(batch.c_word[0][0])),
                                 'qas_id': batch.id[i],
                                 'adv_predict': (orig_s_idx, orig_e_idx),
                                 'orig_predict': (adv_s_idx, adv_e_idx),
                                 'Orig answer:': orig_answer,
                                 'Adv answer:': adv_answer
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')
            except:
                adv_text.append({
                                 'adv_text': transform(to_list(batch.c_word[0][0])),
                                 'qas_id': batch.id[i],
                                 'adv_predict': (orig_s_idx, orig_e_idx),
                                 'orig_predict': (adv_s_idx, adv_e_idx),
                                 'Orig answer:': orig_answer,
                                 'Adv answer:': adv_answer
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')
                continue
        # for batch size = 1
        tot += 1
        logger.info(("orig predict", (orig_s_idx, orig_e_idx)))
        logger.info(("adv append predict", (adv_s_idx, adv_e_idx)))
        logger.info(("targeted successful rate:", targeted_success))
        logger.info(("untargetd successful rate:", untargeted_success))
        logger.info(("Orig answer:", orig_answer))
        logger.info(("Adv answer:", adv_answer))
        logger.info(("tot:", tot))

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data.copy_(backup_params.get(name))

    with open(options.prediction_file, 'w', encoding='utf-8') as f:
        print(json.dumps(answers), file=f)
    with open(options.prediction_file + '_adv.json', 'w', encoding='utf-8') as f:
        print(json.dumps(adv_answers), file=f)
    results = evaluate.main(options)
    logger.info(tot)
    logger.info(("adv loss, results['exact_match'], results['f1']", loss, results['exact_match'], results['f1']))
    return loss, results['exact_match'], results['f1']


def cw_tree_attack():
    cw = CarliniL2_qa(debug=args.debugging)
    criterion = nn.CrossEntropyLoss()
    loss = 0
    tot = 0
    adv_loss = 0
    targeted_success = 0
    untargeted_success = 0
    adv_text = []
    answers = dict()
    adv_answers = dict()

    embed = torch.load(args.word_vector)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    vocab = Vocab(filename=args.dictionary, data=[PAD_WORD, UNK_WORD, EOS_WORD, SOS_WORD])
    generator = Generator(args.test_data, vocab=vocab, embed=embed)
    transfered_embedding = torch.load('bidaf_transfered_embedding.pth')
    transfer_emb = torch.nn.Embedding.from_pretrained(transfered_embedding).to(device)
    seqback = WrappedSeqback(embed, device, attack=True, seqback_model=generator.seqback_model, vocab=vocab,
                             transfer_emb=transfer_emb)
    treelstm = generator.tree_model
    generator.load_state_dict(torch.load(args.load_ae))

    backup_params = EMA(0)
    for name, param in model.named_parameters():
        if param.requires_grad:
            backup_params.register(name, param.data)
            param.data.copy_(ema.get(name))

    class TreeModel(nn.Module):
        def __init__(self):
            super(TreeModel, self).__init__()
            self.inputs = None

        def forward(self, hidden):
            self.embedding = seqback(hidden)
            return model(batch, perturbed=self.embedding)

        def set_temp(self, temp):
            seqback.temp = temp

        def get_embedding(self):
            return self.embedding

        def get_seqback(self):
            return seqback

    tree_model = TreeModel()
    for batch in tqdm(iter(data.dev_iter), total=1000):
        p1, p2 = model(batch)
        orig_answer, orig_s_idx, orig_e_idx = write_to_ans(p1, p2, batch, answers)
        batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
        loss += batch_loss.item()

        append_info = append_input(batch, vocab)
        batch_add_start = append_info['add_start']
        batch_add_end = append_info['add_end']
        batch_start_target = torch.LongTensor(append_info['target_start']).to(device)
        batch_end_target = torch.LongTensor(append_info['target_end']).to(device)
        add_sents = append_info['append_sent']

        input_embedding = model.word_emb(batch.c_word[0])
        append_info['tree'] = [generator.get_tree(append_info['tree'])]
        seqback.sentences = input_embedding.clone().detach()
        seqback.batch_trees = append_info['tree']
        seqback.batch_add_sent = append_info['ae_sent']
        seqback.start = append_info['add_start']
        seqback.end = append_info['add_end']
        seqback.adv_sent = []

        batch_tree_embedding = []
        for bi, append_sent in enumerate(append_info['ae_sent']):
            sentences = [torch.tensor(append_sent, dtype=torch.long, device=device)]
            trees = [append_info['tree'][bi]]
            tree_embedding = treelstm(sentences, trees)[0][0].detach()
            batch_tree_embedding.append(tree_embedding)
        hidden = torch.cat(batch_tree_embedding, dim=0)
        cw.batch_info = append_info
        cw.num_classes = append_info['tot_length']
        cw.run(tree_model, hidden, (batch_start_target, batch_end_target), input_token=input_embedding)
        seqback.adv_sent = []

        # re-test
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            if bi in cw.o_best_sent:
                ae_words = cw.o_best_sent[bi]
                bidaf_tokens = bidaf_convert_to_idx(ae_words)
                batch.c_word[0].data[bi, add_start:add_end] = torch.LongTensor(bidaf_tokens)
        p1, p2 = model(batch)
        adv_answer, adv_s_idx, adv_e_idx = write_to_ans(p1, p2, batch, adv_answers)
        batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
        adv_loss += batch_loss.item()

        for bi, (start_target, end_target) in enumerate(zip(batch_start_target, batch_end_target)):
            start_output = adv_s_idx
            end_output = adv_e_idx
            targeted_success += int(compare(start_output, start_target.item(), end_output, end_target.item()))
            untargeted_success += int(
                compare_untargeted(start_output, start_target.item(), end_output, end_target.item()))

        for i in range(len(add_sents)):
            logger.info(("orig:", transform(add_sents[i])))
            try:
                logger.info(("adv:", cw.o_best_sent[i]))
                adv_text.append({'adv_text': cw.o_best_sent[i],
                                 'qas_id': batch.id[i],
                                 'adv_predict': (orig_s_idx, orig_e_idx),
                                 'orig_predict': (adv_s_idx, adv_e_idx),
                                 'Orig answer:': orig_answer,
                                 'Adv answer:': adv_answer
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')
            except:
                adv_text.append({'adv_text': transform(add_sents[i]),
                                 'qas_id': batch.id[i],
                                 'adv_predict': (orig_s_idx, orig_e_idx),
                                 'orig_predict': (adv_s_idx, adv_e_idx),
                                 'Orig answer:': orig_answer,
                                 'Adv answer:': adv_answer
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')
                continue
        # for batch size = 1
        tot += 1
        logger.info(("orig predict", (orig_s_idx, orig_e_idx)))
        logger.info(("adv append predict", (adv_s_idx, adv_e_idx)))
        logger.info(("targeted successful rate:", targeted_success))
        logger.info(("untargetd successful rate:", untargeted_success))
        logger.info(("Orig answer:", orig_answer))
        logger.info(("Adv answer:", adv_answer))
        logger.info(("tot:", tot))

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data.copy_(backup_params.get(name))

    with open(options.prediction_file, 'w', encoding='utf-8') as f:
        print(json.dumps(answers), file=f)
    with open(options.prediction_file + '_adv.json', 'w', encoding='utf-8') as f:
        print(json.dumps(adv_answers), file=f)
    results = evaluate.main(options)
    logger.info(tot)
    logger.info(("adv loss, results['exact_match'], results['f1']", loss, results['exact_match'], results['f1']))
    return loss, results['exact_match'], results['f1']

def cw_tree_attack_targeted():
    cw = CarliniL2_qa(debug=args.debugging)
    criterion = nn.CrossEntropyLoss()
    loss = 0
    tot = 0
    adv_loss = 0
    targeted_success = 0
    untargeted_success = 0
    adv_text = []
    answers = dict()
    adv_answers = dict()
    # model.eval()

    embed = torch.load(args.word_vector)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    vocab = Vocab(filename=args.dictionary, data=[PAD_WORD, UNK_WORD, EOS_WORD, SOS_WORD])
    generator = Generator(args.test_data, vocab=vocab, embed=embed)
    transfered_embedding = torch.load('bidaf_transfered_embedding.pth')
    transfer_emb = torch.nn.Embedding.from_pretrained(transfered_embedding).to(device)
    seqback = WrappedSeqback(embed, device, attack=True, seqback_model=generator.seqback_model, vocab=vocab,
                             transfer_emb=transfer_emb)
    treelstm = generator.tree_model
    generator.load_state_dict(torch.load(args.load_ae))

    backup_params = EMA(0)
    for name, param in model.named_parameters():
        if param.requires_grad:
            backup_params.register(name, param.data)
            param.data.copy_(ema.get(name))

    class TreeModel(nn.Module):
        def __init__(self):
            super(TreeModel, self).__init__()
            self.inputs = None

        def forward(self, hidden):
            self.embedding = seqback(hidden)
            return model(batch, perturbed=self.embedding)

        def set_temp(self, temp):
            seqback.temp = temp

        def get_embedding(self):
            return self.embedding

        def get_seqback(self):
            return seqback

    tree_model = TreeModel()
    for batch in tqdm(iter(data.dev_iter), total=1000):
        p1, p2 = model(batch)
        orig_answer, orig_s_idx, orig_e_idx = write_to_ans(p1, p2, batch, answers)
        batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
        loss += batch_loss.item()

        append_info = append_input(batch, vocab)
        batch_add_start = append_info['add_start']
        batch_add_end = append_info['add_end']
        batch_start_target = torch.LongTensor(append_info['target_start']).to(device)
        batch_end_target = torch.LongTensor(append_info['target_end']).to(device)
        add_sents = append_info['append_sent']

        input_embedding = model.word_emb(batch.c_word[0])
        append_info['tree'] = [generator.get_tree(append_info['tree'])]
        seqback.sentences = input_embedding.clone().detach()
        seqback.batch_trees = append_info['tree']
        seqback.batch_add_sent = append_info['ae_sent']
        seqback.start = append_info['add_start']
        seqback.end = append_info['add_end']
        seqback.adv_sent = []

        batch_tree_embedding = []
        for bi, append_sent in enumerate(append_info['ae_sent']):
            seqback.target_start = append_info['target_start'][0] - append_info['add_start'][0]
            seqback.target_end = append_info['target_end'][0] - append_info['add_start'][0]
            sentences = [torch.tensor(append_sent, dtype=torch.long, device=device)]
            seqback.target = sentences[0][seqback.target_start:seqback.target_end+1]
            trees = [append_info['tree'][bi]]
            tree_embedding = treelstm(sentences, trees)[0][0].detach()
            batch_tree_embedding.append(tree_embedding)
        hidden = torch.cat(batch_tree_embedding, dim=0)
        cw.batch_info = append_info
        cw.num_classes = append_info['tot_length']

        adv_hidden = cw.run(tree_model, hidden, (batch_start_target, batch_end_target), input_token=input_embedding)
        seqback.adv_sent = []

        # re-test
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            if bi in cw.o_best_sent:
                ae_words = cw.o_best_sent[bi]
                bidaf_tokens = bidaf_convert_to_idx(ae_words)
                batch.c_word[0].data[bi, add_start:add_end] = torch.LongTensor(bidaf_tokens)
        p1, p2 = model(batch)
        adv_answer, adv_s_idx, adv_e_idx = write_to_ans(p1, p2, batch, adv_answers)
        batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
        adv_loss += batch_loss.item()

        for bi, (start_target, end_target) in enumerate(zip(batch_start_target, batch_end_target)):
            start_output = adv_s_idx
            end_output = adv_e_idx
            targeted_success += int(compare(start_output, start_target.item(), end_output, end_target.item()))
            untargeted_success += int(
                compare_untargeted(start_output, start_target.item(), end_output, end_target.item()))

        for i in range(len(add_sents)):
            logger.info(("orig:", transform(add_sents[i])))
            try:
                logger.info(("adv:", cw.o_best_sent[i]))
                adv_text.append({'adv_text': cw.o_best_sent[i],
                                 'qas_id': batch.id[i],
                                 'adv_predict': (orig_s_idx, orig_e_idx),
                                 'orig_predict': (adv_s_idx, adv_e_idx),
                                 'Orig answer:': orig_answer,
                                 'Adv answer:': adv_answer
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')
            except:
                adv_text.append({'adv_text': transform(add_sents[i]),
                                 'qas_id': batch.id[i],
                                 'adv_predict': (orig_s_idx, orig_e_idx),
                                 'orig_predict': (adv_s_idx, adv_e_idx),
                                 'Orig answer:': orig_answer,
                                 'Adv answer:': adv_answer
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')
                continue
        # for batch size = 1
        tot += 1
        logger.info(("orig predict", (orig_s_idx, orig_e_idx)))
        logger.info(("adv append predict", (adv_s_idx, adv_e_idx)))
        logger.info(("targeted successful rate:", targeted_success))
        logger.info(("untargetd successful rate:", untargeted_success))
        logger.info(("Orig answer:", orig_answer))
        logger.info(("Adv answer:", adv_answer))
        logger.info(("tot:", tot))

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data.copy_(backup_params.get(name))

    with open(options.prediction_file, 'w', encoding='utf-8') as f:
        print(json.dumps(answers), file=f)
    with open(options.prediction_file + '_adv.json', 'w', encoding='utf-8') as f:
        print(json.dumps(adv_answers), file=f)
    results = evaluate.main(options)
    logger.info(tot)
    logger.info(("adv loss, results['exact_match'], results['f1']", loss, results['exact_match'], results['f1']))
    return loss, results['exact_match'], results['f1']


def generate_adv_dataset(input_file, adv_dic):
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)
    output_data = {'data': [], 'version': input_data['version']}
    dataset = input_data["data"]
    for article in dataset:
        new_article = article.copy()
        new_article['paragraphs'] = []

        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                try:
                    new_context = paragraph['context'] + " " + adv_dic[qa['id']][-1]
                except:
                    new_context = paragraph['context']
                new_pqa = {'context': new_context, 'qas': [qa.copy()]}
                new_article['paragraphs'].append(new_pqa)
        output_data['data'].append(new_article)

    with open(root_dir + '/adv_text_{}'.format(input_file), "w") as f:
        json.dump(output_data, f)


def generate_adv_dataset_for_random_words(input_file, adv_dic):
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)
    output_data = {'data': [], 'version': input_data['version']}
    dataset = input_data["data"]
    for article in dataset:
        new_article = article.copy()
        new_article['paragraphs'] = []

        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                try:
                    if adv_dic[qa['id']][-1] is not None:
                        new_context = adv_dic[qa['id']][-1]
                except:
                    new_context = paragraph['context']
                new_pqa = {'context': new_context, 'qas': [qa.copy()]}
                new_article['paragraphs'].append(new_pqa)
        output_data['data'].append(new_article)

    with open(root_dir + '/adv_text_{}'.format(input_file), "w") as f:
        json.dump(output_data, f)


def get_adv_dic(adv_data):
    result = {}
    for data in adv_data:
        try:
            result[data['qas_id']] = [' '.join(data['adv_text'])]
        except:
            # result[data['qas_id']] = [' '.join(transform(data['adv_text']))]
            result[data['qas_id']] = None
    return result


if __name__ == '__main__':
    options = args
    device = torch.device("cuda:{}".format(options.gpu))
    best_model_file_name = "model.bin"
    best_ema = "ema.pth"

    # ===-----------------------------------------------------------------------===
    # Log some stuff about this run
    # ===-----------------------------------------------------------------------===
    logger.info(' '.join(sys.argv))
    logger.info('')
    logger.info(options)

    logger.info('loading SQuAD data...')
    data = SQuAD(options)
    setattr(options, 'char_vocab_size', len(data.CHAR.vocab))
    setattr(options, 'word_vocab_size', len(data.WORD.vocab))
    if options.test_file is not None:
        print("testing")
        setattr(options, 'dataset_file', '.data/squad/{}'.format(options.test_file))
    else:
        setattr(options, 'dataset_file', '.data/squad/{}'.format(options.dev_file))
    setattr(options, 'prediction_file', os.path.join(root_dir, 'prediction.json'))
    setattr(options, 'model_time', strftime('%H:%M:%S', gmtime()))
    logger.info('data loading complete!')

    options.old_model = best_model_file_name
    options.old_ema = best_ema


    answer_append_sentences = joblib.load('sampled_perturb_answer_sentences.pkl')
    question_append_sentences = joblib.load('sampled_perturb_question_sentences.pkl')

    model = BiDAF(options, data.WORD.vocab.vectors).to(device)
    if options.old_model is not None:
        model.load_state_dict(torch.load(options.old_model, map_location="cuda:{}".format(options.gpu)))
    if options.old_ema is not None:
        # ema = pickle.load(open(options.old_ema, "rb"))
        ema = torch.load(options.old_ema, map_location=device)
    else:
        ema = EMA(options.exp_decay_rate)
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    if args.model == 'word_attack':
        # dev_loss, dev_exact, dev_f1 = cw_word_attack()
        # dev_loss, dev_exact, dev_f1 = cw_word_attack_target()
        dev_loss, dev_exact, dev_f1 = cw_random_word_attack()
        logger.info('final: adv loss: {:.3f} dev EM: {:.3f} dev F1: {:.3f}'.format(dev_loss, dev_exact, dev_f1))
        adv_data = joblib.load(root_dir + '/adv_text.pkl')
        adv_dict = get_adv_dic(adv_data)
        generate_adv_dataset_for_random_words(args.test_file, adv_dic=adv_dict)
        # generate_adv_dataset(args.test_file, adv_dic=adv_dict)
        options.dataset_file = root_dir + '/adv_text_{}'.format(args.test_file)
        options.test_file = root_dir + '/adv_text_{}'.format(args.test_file)
        data = SQuAD(options, cache=False)
        dev_loss, dev_exact, dev_f1 = test(model, ema, options, data)
    elif args.model == 'tree_attack':
        if args.targeted:
            dev_loss, dev_exact, dev_f1 = cw_tree_attack_targeted()
        else:
            dev_loss, dev_exact, dev_f1 = cw_tree_attack()
        logger.info('final: adv loss: {:.3f} dev EM: {:.3f} dev F1: {:.3f}'.format(dev_loss, dev_exact, dev_f1))
        adv_data = joblib.load(root_dir + '/adv_text.pkl')
        adv_dict = get_adv_dic(adv_data)
        generate_adv_dataset(args.test_file, adv_dic=adv_dict)
        options.dataset_file = root_dir + '/adv_text_{}'.format(args.test_file)
        options.test_file = root_dir + '/adv_text_{}'.format(args.test_file)
        data = SQuAD(options, cache=False)
        dev_loss, dev_exact, dev_f1 = test(model, ema, options, data)
    else:
        dev_loss, dev_exact, dev_f1 = test(model, ema, options, data)
    logger.info('final: adv loss: {:.3f} dev EM: {:.3f} dev F1: {:.3f}'.format(dev_loss, dev_exact, dev_f1))

