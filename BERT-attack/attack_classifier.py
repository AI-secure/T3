import random
import nltk
import numpy as np
import torch
import joblib
from pytorch_transformers import BertTokenizer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from CW_attack import CarliniL2
from CW_attack_random import CarliniL2_random
from my_generator.model import Generator, WrappedSeqback
from util import args, logger, root_dir, PAD_WORD, UNK_WORD, EOS_WORD, SOS_WORD
import bertmodel
from vocab import Vocab


def prepare_bert_input(raw_text):
    seq = tokenizer.encode("[CLS] " + raw_text)
    return seq[:512]


def get_append_sent(raw_text):
    raw_sents = nltk.sent_tokenize(raw_text)
    raw_sents = list(filter(lambda x: len(x) > 3, [prepare_bert_input(sent) for sent in raw_sents]))
    return random.choice(raw_sents)


def get_batch(data_val):
    for bi, i in enumerate(tqdm(range(0, len(data_val), args.batch_size))):
        batch_data = data_val[i:min(i + args.batch_size, len(data_val))]
        raw_text = [prepare_bert_input(x['raw_text']) for x in batch_data]
        data = [torch.LongTensor(x) for x in raw_text]
        seq_len = [len(x) for x in data]
        label = [x['label'] for x in batch_data]
        batch = {'data': pad_sequence(data, batch_first=True).to(device)}
        batch['seq_len'] = torch.LongTensor(seq_len).to(device)
        batch['label'] = torch.LongTensor(label).to(device)

        data = batch['data']
        seq_len = batch['seq_len']
        label = batch['label']
        out = model(data, seq_len)['pred']
        prediction = torch.max(out, 1)[1]
        batch['orig_correct'] = torch.sum((prediction == label).float())

        append_sents = [get_append_sent(x['raw_text']) for x in batch_data]
        new_data = [x + y[1:] for x, y in zip(append_sents, raw_text)]
        new_data = [torch.LongTensor(x[:512]).to(device) for x in new_data]
        batch['seq_len'] = torch.LongTensor([len(x) for x in new_data]).to(device)
        batch['data'] = pad_sequence(new_data, batch_first=True).to(device)

        out = model(batch['data'], batch['seq_len'])['pred']
        prediction = torch.max(out, 1)[1]
        batch['orig_append_correct'] = torch.sum((prediction == label).float())

        batch['add_start'] = [1] * len(label)
        batch['add_end'] = [min(len(x), 512) for x in append_sents]
        batch['add_sents'] = append_sents
        batch['attack_targets'] = torch.LongTensor([4 if x < 2 else 0 for x in label]).to(device)
        yield batch

def get_tree_append_sent(split_text, tree_data, vocab):
    text = list(filter(lambda x: len(x[0]) > 3, zip(split_text, tree_data)))
    text, tree = random.choice(text)
    words = vocab.convertToLabels(text)
    words = ['[CLS]'] + words
    return text, tokenizer.convert_tokens_to_ids(words), tree

def get_tree_batch(data_val, tree_data, vocab):
    for bi, i in enumerate(tqdm(range(0, len(data_val), args.batch_size))):
        batch_data = data_val[i:min(i + args.batch_size, len(data_val))]
        raw_text = [prepare_bert_input(x['raw_text']) for x in batch_data]
        data = [torch.LongTensor(x) for x in raw_text]
        seq_len = [len(x) for x in data]
        label = [x['label'] for x in batch_data]
        batch = {'data': pad_sequence(data, batch_first=True).to(device)}
        batch['seq_len'] = torch.LongTensor(seq_len).to(device)
        batch['label'] = torch.LongTensor(label).to(device)

        data = batch['data']
        seq_len = batch['seq_len']
        label = batch['label']
        out = model(data, seq_len)['pred']
        prediction = torch.max(out, 1)[1]
        batch['orig_correct'] = torch.sum((prediction == label).float())

        batch_tree_data = tree_data[i:min(i + args.batch_size, len(data_val))]
        batch_split_text = [x['words'] for x in batch_tree_data]

        append_sents = [get_tree_append_sent(split_text, tree_data['tree'], vocab)
                        for split_text, tree_data in zip(batch_split_text, batch_tree_data)]
        ae_append_sents = [x[0] for x in append_sents]
        bert_append_sents = [x[1] for x in append_sents]
        batch_trees = [x[2] for x in append_sents]
        new_data = [x + y[1:] for x, y in zip(bert_append_sents, raw_text)]
        new_data = [torch.LongTensor(x[:512]).to(device) for x in new_data]
        batch['seq_len'] = torch.LongTensor([len(x) for x in new_data]).to(device)
        batch['data'] = pad_sequence(new_data, batch_first=True).to(device)

        out = model(batch['data'], batch['seq_len'])['pred']
        prediction = torch.max(out, 1)[1]
        batch['orig_append_correct'] = torch.sum((prediction == label).float())

        batch['add_start'] = [1] * len(label)
        batch['add_end'] = [min(len(x), 512) for x in bert_append_sents]
        batch['add_sents'] = bert_append_sents
        batch['attack_targets'] = torch.LongTensor([4 if x < 2 else 0 for x in label]).to(device)
        batch['tree'] = batch_trees
        batch['ae_add_sents'] = ae_append_sents
        yield batch


def transform(seq):
    if not isinstance(seq, list):
        seq = seq.squeeze().cpu().numpy().tolist()
    return tokenizer.convert_tokens_to_string([tokenizer._convert_id_to_token(x) for x in seq])


def untargeted_success_rate(predictions, labels):
    tot = 0
    for prediction, label in zip(predictions, labels):
        if prediction.item() < 2 <= label or label < 2 <= prediction.item():
            tot += 1
    return tot


def cw_word_attack(data_val):
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
        seq_len = batch['seq_len']
        label = batch['label']
        batch_add_start = batch['add_start']
        batch_add_end = batch['add_end']
        attack_targets = batch['attack_targets']
        add_sents = batch['add_sents']
        tot += len(label)

        input_embedding = model.bert.embeddings.word_embeddings(data)
        cw_mask = np.zeros(input_embedding.shape).astype(np.float32)
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            cw_mask[bi, add_start:add_end] = 1
        cw_mask = torch.from_numpy(cw_mask).float().to(device)
        cw.wv = model.bert.embeddings.word_embeddings.weight
        cw.mask = cw_mask
        cw.seq = data
        cw.batch_info = batch
        cw.seq_len = seq_len
        adv_data = cw.run(model, input_embedding, attack_targets)

        adv_seq = torch.tensor(batch['data']).to(device)
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            if bi in cw.o_best_sent:
                adv_seq.data[bi, add_start:add_end] = torch.LongTensor(cw.o_best_sent[bi])
        out = model(adv_seq, seq_len)['pred']
        prediction = torch.max(out, 1)[1]
        orig_correct += batch['orig_correct'].item()
        orig_append_correct += batch['orig_append_correct'].item()
        adv_correct += torch.sum((prediction == label).float()).item()
        targeted_success += torch.sum((prediction == attack_targets).float()).item()
        untargeted_success += untargeted_success_rate(prediction, label)

        for i in range(len(add_sents)):
            adv_pickle.append({
                'raw_text': transform(adv_seq[i]),
                'label': label[i].item()
            })
            try:
                logger.info(("orig:", transform(add_sents[i][1:])))
                logger.info(("adv:", transform(cw.o_best_sent[i])))
            except:
                continue

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
    insert_idx = sorted(random.sample(range(1, min(512, len(origin_text))), insert_num))
    insert_idx.append(len(origin_text))
    allow_idx = []
    new_seq = origin_text[:insert_idx[0]]
    for i in range(insert_num):
        if len(new_seq) >= 512:
            break
        allow_idx.append(len(new_seq))
        new_seq.append(tokenizer.vocab["the"])
        new_seq.extend(origin_text[insert_idx[i]:insert_idx[i + 1]])
    new_seq = new_seq[:512]
    new_seq_len = len(new_seq)
    return new_seq, new_seq_len, allow_idx


def get_random_word_batch(data_val):
    for bi, i in enumerate(tqdm(range(0, len(data_val), args.batch_size))):
        batch_data = data_val[i:min(i + args.batch_size, len(data_val))]
        raw_text = [prepare_bert_input(x['raw_text']) for x in batch_data]
        data = [torch.LongTensor(x) for x in raw_text]
        seq_len = [len(x) for x in data]
        label = [x['label'] for x in batch_data]
        batch = {'data': pad_sequence(data, batch_first=True).to(device)}
        batch['seq_len'] = torch.LongTensor(seq_len).to(device)
        batch['label'] = torch.LongTensor(label).to(device)

        data = batch['data']
        seq_len = batch['seq_len']
        label = batch['label']
        out = model(data, seq_len)['pred']
        prediction = torch.max(out, 1)[1]
        batch['orig_correct'] = torch.sum((prediction == label).float())

        new_text = []
        new_text_len = []
        allow_idx = []

        random_texts = [add_random_tokens(x) for x in raw_text]
        for random_text in random_texts:
            new_text.append(random_text[0])
            new_text_len.append(random_text[1])
            allow_idx.append(random_text[2])

        data = [torch.LongTensor(x) for x in new_text]
        batch['data'] = pad_sequence(data, batch_first=True).to(device)
        batch['seq_len'] = torch.LongTensor(new_text_len).to(device)

        out = model(batch['data'], batch['seq_len'])['pred']
        prediction = torch.max(out, 1)[1]
        batch['orig_append_correct'] = torch.sum((prediction == label).float())

        batch['add_start'] = []
        batch['add_end'] = []
        batch['add_sents'] = []
        batch['allow_idx'] = allow_idx
        batch['new_text'] = new_text
        batch['new_text_len'] = new_text_len
        batch['attack_targets'] = torch.LongTensor([4 if x < 2 else 0 for x in label]).to(device)
        yield batch


def cw_rand_words_attack(data_val):
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
        seq_len = batch['seq_len']
        label = batch['label']
        batch_add_start = batch['add_start']
        batch_add_end = batch['add_end']
        attack_targets = batch['attack_targets']
        add_sents = batch['add_sents']
        allow_idxs = batch['allow_idx']
        tot += len(label)

        input_embedding = model.bert.embeddings.word_embeddings(data)
        cw_mask = np.zeros(input_embedding.shape).astype(np.float32)
        for bi, allow_idx in enumerate(allow_idxs):
            cw_mask[bi, np.array(allow_idx)] = 1
        cw_mask = torch.from_numpy(cw_mask).float().to(device)
        cw.wv = model.bert.embeddings.word_embeddings.weight
        cw.mask = cw_mask
        cw.seq = data
        cw.batch_info = batch
        cw.seq_len = seq_len
        adv_data = cw.run(model, input_embedding, attack_targets)

        adv_seq = torch.tensor(batch['data']).to(device)
        for bi, allow_idx in enumerate(allow_idxs):
            if bi in cw.o_best_sent:
                for i, idx in enumerate(allow_idx):
                    adv_seq.data[bi, idx] = cw.o_best_sent[bi][i]
        out = model(adv_seq, seq_len)['pred']
        prediction = torch.max(out, 1)[1]
        orig_correct += batch['orig_correct'].item()
        orig_append_correct += batch['orig_append_correct'].item()
        adv_correct += torch.sum((prediction == label).float()).item()
        targeted_success += torch.sum((prediction == attack_targets).float()).item()
        untargeted_success += untargeted_success_rate(prediction, label)

        for i in range(len(adv_seq)):
            adv_pickle.append({
                'raw_text': transform(adv_seq[i]),
                'label': label[i].item()
            })
            try:
                # logger.info(("orig:", transform(add_sents[i][1:])))
                logger.info(("adv:", transform(cw.o_best_sent[i])))
            except:
                continue

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


def cw_tree_attack(data_val, tree_data):
    adv_correct = 0
    targeted_success = 0
    untargeted_success = 0
    orig_correct = 0
    tot = 0
    orig_append_correct = 0
    adv_pickle = []

    cw = CarliniL2(debug=args.debugging)
    embed = torch.load(args.word_vector)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    vocab = Vocab(filename=args.dictionary, data=[PAD_WORD, UNK_WORD, EOS_WORD, SOS_WORD])
    generator = Generator(args.test_data, vocab=vocab, embed=embed, data_set=data_val)
    bert_transfered_embedding = torch.load('bert_transfered_embedding.pth')
    transfer_emb = torch.nn.Embedding(bert_transfered_embedding.size(0), bert_transfered_embedding.size(1)).to(device)
    # transfer_emb = torch.nn.Embedding.from_pretrained(bert_transfered_embedding).to(device)
    transfer_emb.weight.data.copy_(bert_transfered_embedding)
    seqback = WrappedSeqback(embed, device, attack=True, seqback_model=generator.seqback_model, vocab=vocab,
                             transfer_emb=transfer_emb)
    treelstm = generator.tree_model
    generator.load_state_dict(torch.load(args.load_ae))

    class TreeModel(nn.Module):
        def __init__(self):
            super(TreeModel, self).__init__()

        def forward(self, hidden):
            self.embedding = seqback(hidden)
            return model(batch['data'], batch['seq_len'], perturbed=self.embedding)['pred']

        def set_temp(self, temp):
            seqback.temp = temp

        def get_embedding(self):
            return self.embedding

        def get_seqback(self):
            return seqback

    tree_model = TreeModel()
    for batch in get_tree_batch(data_val, tree_data, vocab):
        input_embedding = model.bert.embeddings.word_embeddings(batch['data'])
        batch['tree'] = [generator.get_tree(tree) for tree in batch['tree']]
        seqback.sentences = input_embedding.clone().detach()
        seqback.batch_trees = batch['tree']
        seqback.batch_add_sent = batch['ae_add_sents']
        seqback.start = batch['add_start']
        seqback.end = batch['add_end']
        seqback.adv_sent = []

        batch_tree_embedding = []
        for bi, append_sent in enumerate(batch['ae_add_sents']):
            sentences = [torch.tensor(append_sent, dtype=torch.long, device=device)]
            trees = [batch['tree'][bi]]
            tree_embedding = treelstm(sentences, trees)[0][0].detach()
            batch_tree_embedding.append(tree_embedding)

        hidden = torch.cat(batch_tree_embedding, dim=0)
        cw.batch_info = batch

        adv_hidden = cw.run(tree_model, hidden, batch['attack_targets'], batch_size=hidden.shape[0],
                            input_token=input_embedding)
        seqback.adv_sent = []

        adv_seq = torch.tensor(batch['data']).to(device)
        for bi, (add_start, add_end) in enumerate(zip(batch['add_start'], batch['add_end'])):
            if bi in cw.o_best_sent:
                ae_words = cw.o_best_sent[bi]
                bert_tokens = tokenizer.convert_tokens_to_ids(ae_words)
                adv_seq[bi, add_start:add_end] = torch.LongTensor(bert_tokens)

        out = model(adv_seq, batch['seq_len'])['pred']
        prediction = torch.max(out, 1)[1]
        orig_correct += batch['orig_correct'].item()
        orig_append_correct += batch['orig_append_correct'].item()
        adv_correct += torch.sum((prediction == batch['label']).float()).item()
        targeted_success += torch.sum((prediction == batch['attack_targets']).float()).item()
        untargeted_success += untargeted_success_rate(prediction, batch['label'])
        tot += len(batch['label'])

        for i in range(len(batch['label'])):
            adv_pickle.append({
                'raw_text': transform(adv_seq[i]),
                'label': batch['label'][i].item()
            })
            try:
                logger.info(("orig:", transform(batch['add_sents'][i])))
                logger.info(("adv:", cw.o_best_sent[i]))
            except:
                continue

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



if __name__ == '__main__':
    model = bertmodel.BertC(dropout=0.1, num_class=5)
    model.load_state_dict(torch.load(args.load, map_location="cuda:{}".format(args.device)))
    device = torch.device("cuda:{}".format(args.device))
    model = model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    dataset = joblib.load(args.test_data)
    tree_data = joblib.load(args.test_tree)
    if args.model == 'tree_attack':
        cw_tree_attack(dataset, tree_data)
    elif args.model == 'random_attack':
        cw_rand_words_attack(dataset)
    else:
        cw_word_attack(dataset)
