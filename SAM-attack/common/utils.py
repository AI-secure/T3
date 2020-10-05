from __future__ import division
from __future__ import print_function

import os
import re
import html
import nltk
import math
import json
import torch
import codecs
import bleach
import logging

from tqdm import tqdm

from common.vocab import Vocab
from nltk.parse.corenlp import CoreNLPDependencyParser

from random import getrandbits
import tensorflow as tf
import torch
from torch.autograd import Variable

from util import get_args

logger = logging.getLogger(__name__)
info = logger.info


# https://gist.github.com/kingspp/3ec7d9958c13b94310c1a365759aa3f4
# Pyfunc Gradient Function
def _py_func_with_gradient(func, inp, Tout, stateful=True, name=None,
                           grad_func=None):
    """
    PyFunc defined as given by Tensorflow
    :param func: Custom Function
    :param inp: Function Inputs
    :param Tout: Ouput Type of out Custom Function
    :param stateful: Calculate Gradients when stateful is True
    :param name: Name of the PyFunction
    :param grad: Custom Gradient Function
    :return:
    """
    # Generate random name in order to avoid conflicts with inbuilt names
    rnd_name = 'PyFuncGrad-' + '%0x' % getrandbits(30 * 4)

    # Register Tensorflow Gradient
    tf.RegisterGradient(rnd_name)(grad_func)

    # Get current graph
    g = tf.get_default_graph()

    # Add gradient override map
    with g.gradient_override_map({"PyFunc": rnd_name,
                                  "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def convert_pytorch_model_to_tf(model, out_dims=None):
    """
    Convert a pytorch model into a tensorflow op that allows backprop
    :param model: A pytorch nn.Model object
    :param out_dims: The number of output dimensions (classes) for the model
    :return: A model function that maps an input (tf.Tensor) to the
    output of the model (tf.Tensor)
    """
    torch_state = {
        'logits': None,
        'x': None,
    }
    if not out_dims:
        out_dims = list(model.modules())[-1].out_features

    def _fprop_fn(x_np):
        """TODO: write this"""
        x_tensor = torch.Tensor(x_np)
        if torch.cuda.is_available():
            x_tensor = x_tensor.cuda()
        torch_state['x'] = Variable(x_tensor, requires_grad=True)
        torch_state['logits'] = model(torch_state['x'])
        return torch_state['logits'].data.cpu().numpy()

    def _bprop_fn(x_np, grads_in_np):
        """TODO: write this"""
        _fprop_fn(x_np)

        grads_in_tensor = torch.Tensor(grads_in_np)
        if torch.cuda.is_available():
            grads_in_tensor = grads_in_tensor.cuda()

        # Run our backprop through our logits to our xs
        loss = torch.sum(torch_state['logits'] * grads_in_tensor)
        loss.backward()
        return torch_state['x'].grad.cpu().data.numpy()

    def _tf_gradient_fn(op, grads_in):
        """TODO: write this"""
        return tf.py_func(_bprop_fn, [op.inputs[0], grads_in],
                          Tout=[tf.float32])

    def tf_model_fn(x_op):
        """TODO: write this"""
        out = _py_func_with_gradient(_fprop_fn, [x_op], Tout=[tf.float32],
                                     stateful=True,
                                     grad_func=_tf_gradient_fn)[0]
        out.set_shape(out_dims)
        return out

    return tf_model_fn


# loading GLOVE word vectors
# if .pth file is found, will load that
# else will load from .txt file & save
def load_word_vectors(path):
    if os.path.isfile(path + '.pth') and os.path.isfile(path + '.vocab'):
        print('==> File found, loading to memory')
        vectors = torch.load(path + '.pth')
        vocab = Vocab(filename=path + '.vocab')
        return vocab, vectors
    # saved file not found, read from txt file
    # and create tensors for word vectors
    print('==> File not found, preparing, be patient')
    count = sum(1 for line in open(path + '.txt', 'r', encoding='utf8', errors='ignore'))
    with open(path + '.txt', 'r') as f:
        contents = f.readline().rstrip('\n').split(' ')
        dim = len(contents[1:])
    words = [None] * (count)
    vectors = torch.zeros(count, dim, dtype=torch.float, device='cpu')
    with open(path + '.txt', 'r', encoding='utf8', errors='ignore') as f:
        idx = 0
        for line in f:
            contents = line.rstrip('\n').split(' ')
            words[idx] = contents[0]
            values = list(map(float, contents[1:]))
            vectors[idx] = torch.tensor(values, dtype=torch.float, device='cpu')
            idx += 1
    with open(path + '.vocab', 'w', encoding='utf8', errors='ignore') as f:
        for word in words:
            f.write(word + '\n')
    vocab = Vocab(filename=path + '.vocab')
    torch.save(vectors, path + '.pth')
    return vocab, vectors


# mapping from scalar to vector
def map_label_to_target(label, num_classes):
    target = torch.zeros(1, num_classes, dtype=torch.float, device='cpu')
    ceil = int(math.ceil(label))
    floor = int(math.floor(label))
    if ceil == floor:
        target[0, floor - 1] = 1
    else:
        target[0, floor - 1] = ceil - label
        target[0, ceil - 1] = label - floor
    return target


def parse_sentence(parse_values, splitted_sentence):
    """
    The function compare the parse tree with splitted token.
    1. Process blank situation
    2. Compare sentence length
    3. if every token equal, success
        else fail
    :param parse_values: parsed trees generaeted by stanford core parser
    :param splitted_sentence: splitted senetence token
    :return: triple, token, mask
    """

    def postpone(parse_graph, start, offset=1):
        for i in range(len(parse_graph)):
            if parse_graph[i]["head"] >= start:
                parse_graph[i]["head"] += offset
        for i in range(start, len(parse_graph)):
            parse_graph[i]["address"] = parse_graph[i]["address"] + offset

    words = [x["word"] for x in parse_values]
    for i, word in enumerate(splitted_sentence):
        if splitted_sentence[i] == '':
            postpone(parse_values, start=i)
            parse_values.insert(i, {"address": i + 1, "head": 0, "rel": "BLANK"})
            words.insert(i, '')
        elif i < len(words) and words[i] == splitted_sentence[i]:
            pass
        else:
            return None, splitted_sentence, False

    if len(splitted_sentence) != len(parse_values):
        return None, splitted_sentence, False
    else:
        triple = []
        for k in parse_values:
            if k["head"] is None:
                continue
            elif k["head"] == 0:
                triple.append((("ROOT", k["head"] - 1), (words[k["address"] - 1], k["address"] - 1), k["rel"]))
            else:
                triple.append(
                    ((words[k["head"] - 1], k["head"] - 1), (words[k["address"] - 1], k["address"] - 1),
                     k["rel"]))
        return triple, splitted_sentence, True


def parse_sentence_without_compare(sentence):
    """
    :param sentence: str
    :return: triples, words
    """
    parser = CoreNLPDependencyParser(url='http://localhost:9000')

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


def parse_by_token(tokens, splited_tokens):
    """
    :param tokens: list(list(str)) : a paragraph of sentences
    :return: triples, words, masks, failure
    """
    parser = CoreNLPDependencyParser(url='http://localhost:9000')
    parses = []

    for sent in splited_tokens:
        parses.append(parser.parse_sents([sent], properties={
            'tokenize.options': 'ptb3Escaping=false, normalizeFractions=false'}))

    parse_graphs = []
    try:
        for parse in parses:
            parse_graph = []
            parse_sents = list(parse)
            for i in parse_sents:
                for j in i:
                    if j is not None:
                        parse_graph.append(list(j.nodes.values()))
            if len(parse_graph) > 1:
                parse_graph = [None]
            parse_graphs += parse_graph
        if len(parse_graphs) != len(splited_tokens):
            raise Exception(
                "parsed {} senteces more than original {} sentences".format(len(parse_graphs), len(splited_tokens)))
    except Exception as e:
        print(e)
        return [None], tokens, [False], 1

    triples = []
    tokenized_p = []
    masks = []
    failure = 0

    for i, parse_graph in enumerate(parse_graphs):
        if parse_graph is not None:
            parse_values = []
            for k in parse_graph:
                if k is not None:
                    parse_values.append(k)
                else:
                    print("NONE happened", tokens)
            parse_values.sort(key=lambda x: x["address"])
            parse_values = parse_values[1:]
            triple, tokens, mask = parse_sentence(parse_values, splited_tokens[i])
            if triple is None:
                failure += 1
            triples.append(triple)
            tokenized_p.append(tokens)
            masks.append(mask)
        else:
            triples.append(None)
            tokenized_p.append(splited_tokens[i])
            masks.append(False)
            failure += 1

    return triples, tokenized_p, masks, failure


def parse(sentence):
    parser = CoreNLPDependencyParser(url='http://localhost:9000')
    parse = parser.raw_parse(sentence)
    parse_tree = list(parse)[0]

    triple = []
    parse_values = []
    for k in parse_tree.nodes.values():
        if k is not None:
            parse_values.append(k)
        else:
            print("NONE happened", sentence)
    parse_values.sort(key=lambda x: x["address"])
    parse_values = parse_values[1:]
    words = [x["word"] for x in parse_values]

    for k in parse_tree.nodes.values():
        try:
            if k["address"] == 0:
                continue
            elif k["head"] == 0:
                triple.append((("ROOT", k["head"] - 1), (words[k["address"] - 1], k["address"] - 1), k["rel"]))
            else:
                triple.append(
                    ((words[k["head"] - 1], k["head"] - 1), (words[k["address"] - 1], k["address"] - 1), k["rel"]))
        except IndexError:
            print(words)
    return triple, words


def build_vocab(filepaths, dst_path):
    vocab = set()
    for filepath in filepaths:
        with codecs.open(filepath, "r", 'utf-8') as f:
            sen_tris = json.load(f, encoding='utf-8')
            for sen_tri in sen_tris:
                for word in sen_tri['words']:
                    vocab |= set(word)
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')


def build_rel_vocab(filepaths, dst_path):
    vocab = set()
    for filepath in filepaths:
        with codecs.open(filepath, "r", 'utf-8') as f:
            sen_tris = json.load(f, encoding='utf-8')
            for article in sen_tris:
                for paragraph in article:
                    if paragraph['triples'] is not None:
                        for sentence in paragraph['triples']:
                            if sentence is not None:
                                for rel in sentence:
                                    vocab.add(rel[2])

    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')


def build_path(triple, words):
    triple_dict = {}
    parents = []
    for tri in triple:
        triple_dict[tri[1][1]] = tri[0][1]
        parents.append(tri[0][1])
    parents = set(parents)
    paths = []
    for i, word in enumerate(words):
        if word not in [',', '.']:
            path = []
            now_node = i
            while now_node != -1:
                path.insert(0, words[now_node])
                # path.append(words[now_node])
                if now_node in triple_dict.keys():
                    now_node = triple_dict[now_node]
                else:
                    print('now_node', now_node)
                    print('Wrong triple_dict', triple_dict)
                    break
            # print('PATH', path)
            paths.append(path)
    return paths


def tokenize(paragraph):
    # cleaning HTMLs
    paragraph = html.unescape(paragraph)
    paragraph = bleach.clean(paragraph, tags=[], strip=True).strip()
    paragraph = html.unescape(paragraph)
    # cutting sentences
    sentences = nltk.sent_tokenize(paragraph)
    # print('Sentence Length', len(sentences))

    sentence_words = []
    sentence_triples = []
    sentence_paths = []

    for sentence in sentences:
        sentence = re.sub('[^.,a-zA-Z0-9 \n\.]', '', sentence)
        sentence = sentence.lower()
        # Build Dependency Tree
        try:
            tri, words = parse(sentence)
            if not words:
                continue
            path = build_path(tri, words)
            # print('Path', path)
            if not path:
                continue

            sentence_words.append(words)
            sentence_triples.append(tri)
            sentence_paths.append(path)
        except Exception as e:
            print(e)
            print(sentence)
    return sentence_words, sentence_triples, sentence_paths


def processe_raw_data(data_file):
    for j, label in enumerate(['pos', 'neg']):
        curdir = os.path.join(data_file, label)
        # outfile = os.path.join(datadir, '{0}-{1}.json'.format(i, j))

        info('reading {}'.format(curdir))
        sen_tris = []
        for k, elm in enumerate(os.listdir(curdir)):
            print('K and ELM:', k, elm)
            # if k > 0:
            #     break
            with open(os.path.join(curdir, elm), 'r') as r:
                sentence = re.sub('[ \t\n]+', ' ', r.read().strip())
                (sentence_words, sentence_triples, sentence_paths) = tokenize(sentence)
                sen_tri = {}
                sen_tri['id'] = k
                sen_tri['words'] = sentence_words
                sen_tri['triples'] = sentence_triples
                sen_tri['paths'] = sentence_paths
                sen_tris.append(sen_tri)
        with codecs.open(os.path.join(data_file, label + ".json"), "wb", 'utf-8') as f:
            json.dump(sen_tris, f, ensure_ascii=False, separators=(',', ':'))
