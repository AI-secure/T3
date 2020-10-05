import json
import os
import random
import nltk
import numpy as np
import torch
import joblib
from pytorch_transformers import BertTokenizer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from CW_QA_attack import CarliniL2_qa
from CW_QA_attack_ensemble import CarliniL2QAEnsemble
from CW_QA_attack_random import CarliniL2_qa_random
from CW_attack_random import CarliniL2_random
from my_generator.model import Generator, WrappedSeqback
from util import args, logger, root_dir, PAD_WORD, UNK_WORD, EOS_WORD, SOS_WORD
import bertmodel
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from utils_squad import read_squad_examples, convert_examples_to_features, RawResult, write_predictions
from utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad
from vocab import Vocab


def load_and_cache_examples(input_file, tokenizer, load_cache=True):
    examples = read_squad_examples(input_file=input_file,
                                   is_training=False,
                                   version_2_with_negative=False)
    print("Creating features from dataset file at %s", input_file)
    cached_features_file = '{}_{}_cached'.format(dev_file, str(max_seq_length))
    if os.path.exists(cached_features_file) and load_cache:
        print("Loading features from cached file {}".format(cached_features_file))
        features = torch.load(cached_features_file)
    else:
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=max_seq_length,
                                                doc_stride=doc_stride,
                                                max_query_length=max_query_length,
                                                is_training=False)
        print("Saving features into cached file {}".format(cached_features_file))
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_example_index, all_cls_index, all_p_mask)
    return dataset, examples, features


def to_list(tensor):
    return tensor.detach().cpu().tolist()


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


def append_input(inputs, example_ids, vocab=None, heuristic=0):
    """
    :param heuristic: ablation study control
    :param inputs: batch of data
    :param example_ids: batch of exmaples
    :param vocab: give ae vocab if have one
    :return:
    """
    append_input = torch.tensor(inputs['input_ids'])
    append_mask = torch.tensor(inputs['attention_mask'])
    append_segment = torch.tensor(inputs['token_type_ids'])
    length = torch.sum(append_mask).item()
    assert len(example_ids) == 1, "Only support batch size = 1"
    example_id = example_ids[0]
    if heuristic == 1:  # use perturbed answer sentences as initial seed
        ans_append = answer_append_sentences[example_id]
        append_info = ans_append.copy()
    elif heuristic == 2:  # use random sentences as initial seed
        ans_append = answer_append_sentences[example_id]
        append_info = ans_append.copy()
    else:  # use question words as initial seed
        ans_append = answer_append_sentences[example_id]
        qas_append = question_append_sentences[example_id]
        # if fail, use perturbed answer sentences as initial seed
        append_info = qas_append.copy() if qas_append['append_sent'] is not None else ans_append.copy()
    if vocab is not None:
        append_info = generate_tree_info(append_info)
        append_info['ae_sent'] = vocab.convertToIdx(append_info['append_sent'], UNK_WORD)
        append_info['ae_sent'] = [append_info['ae_sent']]
    try:
        append_info['append_sent'] = tokenizer.convert_tokens_to_ids(append_info['append_sent'])
    except Exception as e:
        print(e)
        print(append_info['append_sent'])
        print(example_id)
        print(ans_append)
        print(qas_append)
    x = append_info['append_sent']
    append_info['add_start'] = length
    append_info['add_end'] = length + len(x)
    append_info['target_start'] += length
    append_info['target_end'] += length

    if length + len(x) < max_seq_length:
        for i in range(length, length + len(x)):
            append_input[0, i] = x[i - length]
            append_mask[0, i] = 1
            append_segment[0, i] = 1
    elif length < max_seq_length:
        for i in range(length, max_seq_length):
            append_input[0, i] = x[i - length]
            append_mask[0, i] = 1
            append_segment[0, i] = 1
        concat_input = torch.unsqueeze(torch.LongTensor(x[max_seq_length - length:]).cuda(), dim=0)
        concat_mask = torch.ones_like(concat_input)
        concat_segment = torch.ones_like(concat_input)
        append_input = torch.cat((append_input, concat_input), dim=1)
        append_mask = torch.cat((append_mask, concat_mask), dim=1)
        append_segment = torch.cat((append_segment, concat_segment), dim=1)
    else:
        concat_input = torch.unsqueeze(torch.LongTensor(x).cuda(), dim=0)
        concat_mask = torch.ones_like(concat_input)
        concat_segment = torch.ones_like(concat_input)
        append_input = torch.cat((append_input, concat_input), dim=1)
        append_mask = torch.cat((append_mask, concat_mask), dim=1)
        append_segment = torch.cat((append_segment, concat_segment), dim=1)
    append_info['append_inputs'] = {
        'input_ids': append_input,
        'attention_mask': append_mask,
        'token_type_ids': append_segment
    }
    append_info['tot_length'] = max(max_seq_length, append_info['add_end'])
    append_info['add_start'] = [append_info['add_start']]
    append_info['add_end'] = [append_info['add_end']]
    append_info['target_start'] = [append_info['target_start']]
    append_info['target_end'] = [append_info['target_end']]
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


def transform(seq):
    if not isinstance(seq, list):
        seq = seq.squeeze().cpu().numpy().tolist()
    return tokenizer.convert_tokens_to_string([tokenizer._convert_id_to_token(x) for x in seq])


def eval(load_cache=True):
    dataset, examples, features = load_and_cache_examples(dev_file, tokenizer, load_cache=load_cache)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size)

    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]
                      }
            example_indices = batch[3]
            outputs = model(**inputs)
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = RawResult(unique_id=unique_id,
                               start_logits=to_list(outputs[0][i]),
                               end_logits=to_list(outputs[1][i]))
            all_results.append(result)

    # Compute predictions
    output_prediction_file = root_dir + "/predictions_.json"
    output_nbest_file = root_dir + "/nbest_predictions_.json"

    write_predictions(examples, features, all_results, 20,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, False,
                      False, null_score_diff_threshold)

    # Evaluate with the official SQuAD script
    evaluate_options = EVAL_OPTS(data_file=dev_file,
                                 pred_file=output_prediction_file,
                                 na_prob_file=output_null_log_odds_file)
    results = evaluate_on_squad(evaluate_options)
    print(results)
    logger.info(str(results))


def cw_word_attack():
    dataset, examples, features = load_and_cache_examples(dev_file, tokenizer)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size)

    all_results = []
    cw = CarliniL2_qa(debug=args.debugging)

    targeted_success = 0
    untargeted_success = 0
    tot = 0
    adv_text = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]
                      }
            example_indices = batch[3]
            outputs = model(**inputs)
        # append origin
        with torch.no_grad():
            example_id = example_indices.item()
            example_id = features[example_id].example_index
            append_info = append_input(inputs, [example_id], heuristic=2)
            append_inputs = append_info['append_inputs']
            batch_add_start = append_info['add_start']
            batch_add_end = append_info['add_end']
            batch_start_target = torch.LongTensor(append_info['target_start']).to(device)
            batch_end_target = torch.LongTensor(append_info['target_end']).to(device)
            add_sents = append_info['append_sent']

            append_outputs = model(**append_inputs)

        # get adv sent
        input_embedding = model.bert.embeddings.word_embeddings(append_inputs['input_ids'])
        cw_mask = np.zeros(input_embedding.shape).astype(np.float32)
        cw_mask = torch.from_numpy(cw_mask).float().to(device)
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            cw_mask[bi, add_start:add_end] = 1
        cw.wv = model.bert.embeddings.word_embeddings.weight
        cw.mask = cw_mask
        cw.batch_info = append_info
        append_inputs['perturbed'] = input_embedding
        cw.inputs = append_inputs
        cw.num_classes = append_info['tot_length']
        cw.run(model, input_embedding, (batch_start_target, batch_end_target))

        # re-test
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            if bi in cw.o_best_sent:
                append_inputs['input_ids'].data[bi, add_start:add_end] = torch.LongTensor(cw.o_best_sent[bi])
        append_inputs['perturbed'] = None
        adv_outputs = model(**append_inputs)
        adv_start_predict = to_list(torch.max(adv_outputs[0], 1)[1])
        adv_end_predict = to_list(torch.max(adv_outputs[1], 1)[1])

        orig_start_predict = to_list(torch.max(outputs[0], 1)[1])
        orig_end_predict = to_list(torch.max(outputs[1], 1)[1])
        append_start_predict = to_list(torch.max(append_outputs[0], 1)[1])
        append_end_predict = to_list(torch.max(append_outputs[1], 1)[1])

        for bi, (start_target, end_target) in enumerate(zip(batch_start_target, batch_end_target)):
            start_output = adv_start_predict[bi]
            end_output = adv_end_predict[bi]
            targeted_success += int(compare(start_output, start_target.item(), end_output, end_target.item()))
            untargeted_success += int(
                compare_untargeted(start_output, start_target.item(), end_output, end_target.item()))

        for i in range(len(cw.o_best_sent)):
            try:
                logger.info(("orig:", transform(add_sents[i])))
                logger.info(("adv:", transform(cw.o_best_sent[i])))
            except:
                continue
        tot += 1
        logger.info(("orig start predict", orig_start_predict[0]))
        logger.info(("orig end predict", orig_end_predict[0]))
        logger.info(("orig append start predict", append_start_predict[0]))
        logger.info(("orig append end predict", append_end_predict[0]))
        logger.info(("adv append start predict", adv_start_predict[0]))
        logger.info(("adv append end predict", adv_end_predict[0]))
        logger.info(("targeted successful rate:", targeted_success))
        logger.info(("untargetd successful rate:", untargeted_success))
        logger.info(("tot:", tot))

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = RawResult(unique_id=unique_id,
                               start_logits=to_list(outputs[0][i]),
                               end_logits=to_list(outputs[1][i]))
            all_results.append(result)
            try:
                adv_text.append({'adv_text': cw.o_best_sent[i],
                                 'qas_id': examples[eval_feature.example_index].qas_id,
                                 'adv_predict': (adv_start_predict[0], adv_end_predict[0]),
                                 'orig_predict': (orig_start_predict[0], orig_end_predict[0]),
                                 'orig_append_predict': (append_start_predict[0], append_end_predict[0])
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')
            except:
                adv_text.append({'adv_text': add_sents[i],
                                 'qas_id': examples[eval_feature.example_index].qas_id,
                                 'adv_predict': (adv_start_predict[0], adv_end_predict[0]),
                                 'orig_predict': (orig_start_predict[0], orig_end_predict[0]),
                                 'orig_append_predict': (append_start_predict[0], append_end_predict[0])
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')

    # Compute predictions
    output_prediction_file = "predictions_.json"
    output_nbest_file = "nbest_predictions_.json"

    write_predictions(examples, features, all_results, 20,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, False,
                      False, null_score_diff_threshold)

    # Evaluate with the official SQuAD script
    evaluate_options = EVAL_OPTS(data_file=dev_file,
                                 pred_file=output_prediction_file,
                                 na_prob_file=output_null_log_odds_file)
    results = evaluate_on_squad(evaluate_options)
    print(results)


def cw_word_attack_ensemble():
    dataset, examples, features = load_and_cache_examples(dev_file, tokenizer)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size)
    models = [model, model1, model2]
    all_results = []
    cw = CarliniL2QAEnsemble(debug=args.debugging)

    targeted_success = 0
    untargeted_success = 0
    tot = 0
    adv_text = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]
                      }
            example_indices = batch[3]
            outputs = model(**inputs)
        # append origin
        with torch.no_grad():
            example_id = example_indices.item()
            example_id = features[example_id].example_index
            append_info = append_input(inputs, [example_id])
            append_inputs = append_info['append_inputs']
            batch_add_start = append_info['add_start']
            batch_add_end = append_info['add_end']
            batch_start_target = torch.LongTensor(append_info['target_start']).to(device)
            batch_end_target = torch.LongTensor(append_info['target_end']).to(device)
            add_sents = append_info['append_sent']

            append_outputs = model(**append_inputs)

        # get adv sent
        input_embedding = model.bert.embeddings.word_embeddings(append_inputs['input_ids'])
        cw_mask = np.zeros(input_embedding.shape).astype(np.float32)
        cw_mask = torch.from_numpy(cw_mask).float().to(device)
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            cw_mask[bi, add_start:add_end] = 1
        cw.wv = model.bert.embeddings.word_embeddings.weight
        cw.mask = cw_mask
        cw.batch_info = append_info
        append_inputs['perturbed'] = input_embedding
        cw.inputs = append_inputs
        cw.num_classes = append_info['tot_length']
        cw.run(models, input_embedding, (batch_start_target, batch_end_target))

        # re-test
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            if bi in cw.o_best_sent:
                append_inputs['input_ids'].data[bi, add_start:add_end] = torch.LongTensor(cw.o_best_sent[bi])
        append_inputs['perturbed'] = None
        adv_outputs = model(**append_inputs)
        adv_start_predict = to_list(torch.max(adv_outputs[0], 1)[1])
        adv_end_predict = to_list(torch.max(adv_outputs[1], 1)[1])

        orig_start_predict = to_list(torch.max(outputs[0], 1)[1])
        orig_end_predict = to_list(torch.max(outputs[1], 1)[1])
        append_start_predict = to_list(torch.max(append_outputs[0], 1)[1])
        append_end_predict = to_list(torch.max(append_outputs[1], 1)[1])

        for bi, (start_target, end_target) in enumerate(zip(batch_start_target, batch_end_target)):
            start_output = adv_start_predict[bi]
            end_output = adv_end_predict[bi]
            targeted_success += int(compare(start_output, start_target.item(), end_output, end_target.item()))
            untargeted_success += int(
                compare_untargeted(start_output, start_target.item(), end_output, end_target.item()))

        for i in range(len(cw.o_best_sent)):
            try:
                logger.info(("orig:", transform(add_sents[i])))
                logger.info(("adv:", transform(cw.o_best_sent[i])))
            except:
                continue
        tot += 1
        logger.info(("orig start predict", orig_start_predict[0]))
        logger.info(("orig end predict", orig_end_predict[0]))
        logger.info(("orig append start predict", append_start_predict[0]))
        logger.info(("orig append end predict", append_end_predict[0]))
        logger.info(("adv append start predict", adv_start_predict[0]))
        logger.info(("adv append end predict", adv_end_predict[0]))
        logger.info(("targeted successful rate:", targeted_success))
        logger.info(("untargetd successful rate:", untargeted_success))
        logger.info(("tot:", tot))

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = RawResult(unique_id=unique_id,
                               start_logits=to_list(outputs[0][i]),
                               end_logits=to_list(outputs[1][i]))
            all_results.append(result)
            try:
                adv_text.append({'adv_text': cw.o_best_sent[i],
                                 'qas_id': examples[eval_feature.example_index].qas_id,
                                 'adv_predict': (adv_start_predict[0], adv_end_predict[0]),
                                 'orig_predict': (orig_start_predict[0], orig_end_predict[0]),
                                 'orig_append_predict': (append_start_predict[0], append_end_predict[0])
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')
            except:
                adv_text.append({'adv_text': add_sents[i],
                                 'qas_id': examples[eval_feature.example_index].qas_id,
                                 'adv_predict': (adv_start_predict[0], adv_end_predict[0]),
                                 'orig_predict': (orig_start_predict[0], orig_end_predict[0]),
                                 'orig_append_predict': (append_start_predict[0], append_end_predict[0])
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')

    # Compute predictions
    output_prediction_file = "predictions_.json"
    output_nbest_file = "nbest_predictions_.json"

    write_predictions(examples, features, all_results, 20,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, False,
                      False, null_score_diff_threshold)

    # Evaluate with the official SQuAD script
    evaluate_options = EVAL_OPTS(data_file=dev_file,
                                 pred_file=output_prediction_file,
                                 na_prob_file=output_null_log_odds_file)
    results = evaluate_on_squad(evaluate_options)
    print(results)


def cw_word_attack_target():
    dataset, examples, features = load_and_cache_examples(dev_file, tokenizer)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size)

    all_results = []
    cw = CarliniL2_qa(debug=args.debugging)

    targeted_success = 0
    untargeted_success = 0
    tot = 0
    adv_text = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]
                      }
            example_indices = batch[3]
            outputs = model(**inputs)
        # append origin
        with torch.no_grad():
            example_id = example_indices.item()
            example_id = features[example_id].example_index
            append_info = append_input(inputs, [example_id])
            append_inputs = append_info['append_inputs']
            batch_add_start = append_info['add_start']
            batch_add_end = append_info['add_end']
            batch_start_target = torch.LongTensor(append_info['target_start']).to(device)
            batch_end_target = torch.LongTensor(append_info['target_end']).to(device)
            add_sents = append_info['append_sent']

            append_outputs = model(**append_inputs)

        # get adv sent
        input_embedding = model.bert.embeddings.word_embeddings(append_inputs['input_ids'])
        cw_mask = np.zeros(input_embedding.shape).astype(np.float32)
        cw_mask = torch.from_numpy(cw_mask).float().to(device)
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            cw_mask[bi, add_start:add_end] = 1
        for bi, (add_start, add_end) in enumerate(zip(append_info['target_start'], append_info['target_end'])):
            # unmask the target
            cw_mask[bi, add_start:add_end + 1] = 0
        cw.wv = model.bert.embeddings.word_embeddings.weight
        cw.mask = cw_mask
        cw.batch_info = append_info
        append_inputs['perturbed'] = input_embedding
        cw.inputs = append_inputs
        cw.num_classes = append_info['tot_length']
        cw.run(model, input_embedding, (batch_start_target, batch_end_target))

        # re-test
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            if bi in cw.o_best_sent:
                append_inputs['input_ids'].data[bi, add_start:add_end] = torch.LongTensor(cw.o_best_sent[bi])
        append_inputs['perturbed'] = None
        adv_outputs = model(**append_inputs)
        adv_start_predict = to_list(torch.max(adv_outputs[0], 1)[1])
        adv_end_predict = to_list(torch.max(adv_outputs[1], 1)[1])

        orig_start_predict = to_list(torch.max(outputs[0], 1)[1])
        orig_end_predict = to_list(torch.max(outputs[1], 1)[1])
        append_start_predict = to_list(torch.max(append_outputs[0], 1)[1])
        append_end_predict = to_list(torch.max(append_outputs[1], 1)[1])

        for bi, (start_target, end_target) in enumerate(zip(batch_start_target, batch_end_target)):
            start_output = adv_start_predict[bi]
            end_output = adv_end_predict[bi]
            targeted_success += int(compare(start_output, start_target.item(), end_output, end_target.item()))
            untargeted_success += int(
                compare_untargeted(start_output, start_target.item(), end_output, end_target.item()))

        for i in range(len(cw.o_best_sent)):
            try:
                logger.info(("orig:", transform(add_sents[i])))
                logger.info(("adv:", transform(cw.o_best_sent[i])))
            except:
                continue
        tot += 1
        logger.info(("orig start predict", orig_start_predict[0]))
        logger.info(("orig end predict", orig_end_predict[0]))
        logger.info(("orig append start predict", append_start_predict[0]))
        logger.info(("orig append end predict", append_end_predict[0]))
        logger.info(("adv append start predict", adv_start_predict[0]))
        logger.info(("adv append end predict", adv_end_predict[0]))
        logger.info(("targeted successful rate:", targeted_success))
        logger.info(("untargetd successful rate:", untargeted_success))
        logger.info(("tot:", tot))

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = RawResult(unique_id=unique_id,
                               start_logits=to_list(outputs[0][i]),
                               end_logits=to_list(outputs[1][i]))
            all_results.append(result)
            try:
                adv_text.append({'adv_text': cw.o_best_sent[i],
                                 'qas_id': examples[eval_feature.example_index].qas_id,
                                 'adv_predict': (adv_start_predict[0], adv_end_predict[0]),
                                 'orig_predict': (orig_start_predict[0], orig_end_predict[0]),
                                 'orig_append_predict': (append_start_predict[0], append_end_predict[0])
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')
            except:
                adv_text.append({'adv_text': add_sents[i],
                                 'qas_id': examples[eval_feature.example_index].qas_id,
                                 'adv_predict': (adv_start_predict[0], adv_end_predict[0]),
                                 'orig_predict': (orig_start_predict[0], orig_end_predict[0]),
                                 'orig_append_predict': (append_start_predict[0], append_end_predict[0])
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')

    # Compute predictions
    output_prediction_file = "predictions_.json"
    output_nbest_file = "nbest_predictions_.json"

    write_predictions(examples, features, all_results, 20,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, False,
                      False, null_score_diff_threshold)

    # Evaluate with the official SQuAD script
    evaluate_options = EVAL_OPTS(data_file=dev_file,
                                 pred_file=output_prediction_file,
                                 na_prob_file=output_null_log_odds_file)
    results = evaluate_on_squad(evaluate_options)
    print(results)


# for random token
def add_random_tokens(origin_text, mask, segment, target_start, target_end, random_num=args.scatter_number, targeted=False):
    """
    inpurt:
        origin_text: list[int] list of idx
        random_num: number of insert tokens
    output:
        new_seq: list[int] list of idx after inserting idxes
        new_seq_len: int
        allow_idx: indexes in new_seq which are allowed to modify
    """
    insert_num = min(len(origin_text) // 3 + 1, random_num)
    paragraph_idx = np.nonzero(segment)[0]
    # print(transform(origin_text[paragraph_idx[0]:paragraph_idx[-1]]))
    range1 = list(range(paragraph_idx[0], target_start + 1))
    range2 = list(range(target_end + 1, paragraph_idx[-1]))
    insert_num = min(insert_num, len((range1 + range2)))
    insert_idx = sorted(random.sample((range1 + range2), insert_num))
    insert_idx.append(len(origin_text))
    allow_idx = []
    new_seq = origin_text[:insert_idx[0]]
    new_mask = mask[:insert_idx[0]]
    new_segment = segment[:insert_idx[0]]
    if targeted:  # predefined fake answers
        target_start = paragraph_idx[0]
        target_end = paragraph_idx[0] + 2
    new_target_start = target_start
    new_target_end = target_end
    for i in range(insert_num):
        allow_idx.append(len(new_seq))
        new_seq.append(tokenizer.vocab["the"])
        new_mask.append(1)
        new_segment.append(1)
        new_seq.extend(origin_text[insert_idx[i]:insert_idx[i + 1]])
        new_mask.extend(mask[insert_idx[i]:insert_idx[i + 1]])
        new_segment.extend(segment[insert_idx[i]:insert_idx[i + 1]])

        if insert_idx[i] <= target_start:
            new_target_start += 1
            new_target_end += 1
    new_seq_len = len(new_seq)
    append_info = {
        'new_seq': [new_seq],
        'new_mask': [new_mask],
        'new_segment': [new_segment],
        'new_seq_len': new_seq_len,
        'allow_idx': [allow_idx],
        'target_start': [new_target_start],
        'target_end': [new_target_end]
    }
    return append_info


def append_random_input(inputs, feature_ids, targeted=False):
    """
    :param targeted: untargeted/targeted attack
    :param inputs: batch of data
    :param feature_ids: batch of features
    :return:
    """
    append_input = to_list(inputs['input_ids'])[0]
    append_mask = to_list(inputs['attention_mask'])[0]
    append_segment = to_list(inputs['token_type_ids'])[0]
    assert len(feature_ids) == 1, "Only support batch size = 1"
    example_id = feature_ids[0]

    target_start = features_with_answers[example_id].start_position
    target_end = features_with_answers[example_id].end_position
    original_text = append_input

    append_info = add_random_tokens(original_text, append_mask, append_segment, target_start, target_end,
                                    targeted=targeted)
    logger.info(("orig ans:", transform(append_input[target_start:target_end + 1])))
    inputs = torch.LongTensor(append_info['new_seq']).cuda()
    mask = torch.LongTensor(append_info['new_mask']).cuda()
    segment = torch.LongTensor(append_info['new_segment']).cuda()
    append_info['append_inputs'] = {
        'input_ids': inputs,
        'attention_mask': mask,
        'token_type_ids': segment
    }
    append_info['tot_length'] = inputs.size(1)
    return append_info


def cw_random_word_attack_untargeted():
    dataset, examples, features = load_and_cache_examples(dev_file, tokenizer)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size)

    all_results = []
    cw = CarliniL2_qa_random(debug=args.debugging, targeted=False)

    targeted_success = 0
    untargeted_success = 0
    tot = 0
    adv_text = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]
                      }
            example_indices = batch[3]
            outputs = model(**inputs)
        # append origin
        with torch.no_grad():
            feature_id = example_indices.item()
            append_info = append_random_input(inputs, [feature_id])
            append_inputs = append_info['append_inputs']
            batch_start_target = torch.LongTensor(append_info['target_start']).to(device)
            batch_end_target = torch.LongTensor(append_info['target_end']).to(device)

            append_outputs = model(**append_inputs)

        # get adv sent
        allow_idxs = append_info['allow_idx']
        input_embedding = model.bert.embeddings.word_embeddings(append_inputs['input_ids'])
        cw_mask = np.zeros(input_embedding.shape).astype(np.float32)
        cw_mask = torch.from_numpy(cw_mask).float().to(device)
        for bi, allow_idx in enumerate(allow_idxs):
            cw_mask[bi, np.array(allow_idx)] = 1
        cw.wv = model.bert.embeddings.word_embeddings.weight
        cw.mask = cw_mask
        cw.batch_info = append_info
        append_inputs['perturbed'] = input_embedding
        cw.inputs = append_inputs
        cw.num_classes = append_info['tot_length']
        cw.run(model, input_embedding, (batch_start_target, batch_end_target))

        # re-test
        for bi, allow_idx in enumerate(allow_idxs):
            if bi in cw.o_best_sent:
                for i, idx in enumerate(allow_idx):
                    append_inputs['input_ids'].data[bi, idx] = cw.o_best_sent[bi][i]
        append_inputs['perturbed'] = None
        adv_outputs = model(**append_inputs)
        adv_start_predict = to_list(torch.max(adv_outputs[0], 1)[1])
        adv_end_predict = to_list(torch.max(adv_outputs[1], 1)[1])

        orig_start_predict = to_list(torch.max(outputs[0], 1)[1])
        orig_end_predict = to_list(torch.max(outputs[1], 1)[1])
        append_start_predict = to_list(torch.max(append_outputs[0], 1)[1])
        append_end_predict = to_list(torch.max(append_outputs[1], 1)[1])

        for bi, (start_target, end_target) in enumerate(zip(batch_start_target, batch_end_target)):
            start_output = adv_start_predict[bi]
            end_output = adv_end_predict[bi]
            logger.info(("adv ans:", transform(to_list(append_inputs['input_ids'][0][start_target:end_target + 1]))))
            targeted_success += int(
                compare(start_output, start_target.item(), end_output, end_target.item(), targeted=False))
            untargeted_success += int(
                compare_untargeted(start_output, start_target.item(), end_output, end_target.item(), targeted=False))

        content = to_list(append_inputs['input_ids'])[0]
        segment = to_list(append_inputs['token_type_ids'])[0]
        paragraph_idx = np.nonzero(segment)
        content = np.array(content)[paragraph_idx].tolist()

        for i in range(len(cw.o_best_sent)):
            try:
                logger.info(("adv:", transform(cw.o_best_sent[i])))
            except:
                continue
        tot += 1
        logger.info(("orig start predict", orig_start_predict[0]))
        logger.info(("orig end predict", orig_end_predict[0]))
        logger.info(("orig append start predict", append_start_predict[0]))
        logger.info(("orig append end predict", append_end_predict[0]))
        logger.info(("adv append start predict", adv_start_predict[0]))
        logger.info(("adv append end predict", adv_end_predict[0]))
        logger.info(("targeted successful rate:", targeted_success))
        logger.info(("untargetd successful rate:", untargeted_success))
        logger.info(("tot:", tot))

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = RawResult(unique_id=unique_id,
                               start_logits=to_list(outputs[0][i]),
                               end_logits=to_list(outputs[1][i]))
            all_results.append(result)
            try:
                adv_text.append({'added_text': cw.o_best_sent[i],
                                 'adv_text': content,
                                 'qas_id': examples[eval_feature.example_index].qas_id,
                                 'adv_predict': (adv_start_predict[0], adv_end_predict[0]),
                                 'orig_predict': (orig_start_predict[0], orig_end_predict[0]),
                                 'orig_append_predict': (append_start_predict[0], append_end_predict[0])
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')
            except:
                adv_text.append({
                    'adv_text': content,
                    'qas_id': examples[eval_feature.example_index].qas_id,
                    'adv_predict': (adv_start_predict[0], adv_end_predict[0]),
                    'orig_predict': (orig_start_predict[0], orig_end_predict[0]),
                    'orig_append_predict': (append_start_predict[0], append_end_predict[0])
                })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')

    # Compute predictions
    output_prediction_file = "predictions_.json"
    output_nbest_file = "nbest_predictions_.json"

    write_predictions(examples, features, all_results, 20,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, False,
                      False, null_score_diff_threshold)

    # Evaluate with the official SQuAD script
    evaluate_options = EVAL_OPTS(data_file=dev_file,
                                 pred_file=output_prediction_file,
                                 na_prob_file=output_null_log_odds_file)
    results = evaluate_on_squad(evaluate_options)
    print(results)


def cw_random_word_attack_position_targeted():
    dataset, examples, features = load_and_cache_examples(dev_file, tokenizer)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size)

    all_results = []
    cw = CarliniL2_qa_random(debug=args.debugging, targeted=True)

    targeted_success = 0
    untargeted_success = 0
    tot = 0
    adv_text = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]
                      }
            example_indices = batch[3]
            outputs = model(**inputs)
        # append origin
        with torch.no_grad():
            feature_id = example_indices.item()
            append_info = append_random_input(inputs, [feature_id], targeted=True)
            append_inputs = append_info['append_inputs']
            batch_start_target = torch.LongTensor(append_info['target_start']).to(device)
            batch_end_target = torch.LongTensor(append_info['target_end']).to(device)

            append_outputs = model(**append_inputs)

        # get adv sent
        allow_idxs = append_info['allow_idx']
        input_embedding = model.bert.embeddings.word_embeddings(append_inputs['input_ids'])
        cw_mask = np.zeros(input_embedding.shape).astype(np.float32)
        cw_mask = torch.from_numpy(cw_mask).float().to(device)
        for bi, allow_idx in enumerate(allow_idxs):
            cw_mask[bi, np.array(allow_idx)] = 1
        cw.wv = model.bert.embeddings.word_embeddings.weight
        cw.mask = cw_mask
        cw.batch_info = append_info
        append_inputs['perturbed'] = input_embedding
        cw.inputs = append_inputs
        cw.num_classes = append_info['tot_length']
        cw.run(model, input_embedding, (batch_start_target, batch_end_target))

        # re-test
        for bi, allow_idx in enumerate(allow_idxs):
            if bi in cw.o_best_sent:
                for i, idx in enumerate(allow_idx):
                    append_inputs['input_ids'].data[bi, idx] = cw.o_best_sent[bi][i]
        append_inputs['perturbed'] = None
        adv_outputs = model(**append_inputs)
        adv_start_predict = to_list(torch.max(adv_outputs[0], 1)[1])
        adv_end_predict = to_list(torch.max(adv_outputs[1], 1)[1])

        orig_start_predict = to_list(torch.max(outputs[0], 1)[1])
        orig_end_predict = to_list(torch.max(outputs[1], 1)[1])
        append_start_predict = to_list(torch.max(append_outputs[0], 1)[1])
        append_end_predict = to_list(torch.max(append_outputs[1], 1)[1])

        for bi, (start_target, end_target) in enumerate(zip(batch_start_target, batch_end_target)):
            start_output = adv_start_predict[bi]
            end_output = adv_end_predict[bi]
            logger.info(("adv ans:", transform(to_list(append_inputs['input_ids'][0][start_target:end_target + 1]))))
            targeted_success += int(
                compare(start_output, start_target.item(), end_output, end_target.item(), targeted=True))
            untargeted_success += int(
                compare_untargeted(start_output, start_target.item(), end_output, end_target.item(), targeted=True))

        content = to_list(append_inputs['input_ids'])[0]
        segment = to_list(append_inputs['token_type_ids'])[0]
        paragraph_idx = np.nonzero(segment)
        content = np.array(content)[paragraph_idx].tolist()

        for i in range(len(cw.o_best_sent)):
            try:
                logger.info(("adv:", transform(cw.o_best_sent[i])))
            except:
                continue
        tot += 1
        logger.info(("orig start predict", orig_start_predict[0]))
        logger.info(("orig end predict", orig_end_predict[0]))
        logger.info(("orig append start predict", append_start_predict[0]))
        logger.info(("orig append end predict", append_end_predict[0]))
        logger.info(("adv append start predict", adv_start_predict[0]))
        logger.info(("adv append end predict", adv_end_predict[0]))
        logger.info(("targeted successful rate:", targeted_success))
        logger.info(("untargetd successful rate:", untargeted_success))
        logger.info(("tot:", tot))

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = RawResult(unique_id=unique_id,
                               start_logits=to_list(outputs[0][i]),
                               end_logits=to_list(outputs[1][i]))
            all_results.append(result)
            try:
                adv_text.append({'added_text': cw.o_best_sent[i],
                                 'adv_text': content,
                                 'qas_id': examples[eval_feature.example_index].qas_id,
                                 'adv_predict': (adv_start_predict[0], adv_end_predict[0]),
                                 'orig_predict': (orig_start_predict[0], orig_end_predict[0]),
                                 'orig_append_predict': (append_start_predict[0], append_end_predict[0])
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')
            except:
                adv_text.append({
                    'adv_text': content,
                    'qas_id': examples[eval_feature.example_index].qas_id,
                    'adv_predict': (adv_start_predict[0], adv_end_predict[0]),
                    'orig_predict': (orig_start_predict[0], orig_end_predict[0]),
                    'orig_append_predict': (append_start_predict[0], append_end_predict[0])
                })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')

    # Compute predictions
    # joblib.dump(adv_text, root_dir + '/adv_text.pkl')
    output_prediction_file = "predictions_.json"
    output_nbest_file = "nbest_predictions_.json"

    write_predictions(examples, features, all_results, 20,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, False,
                      False, null_score_diff_threshold)

    # Evaluate with the official SQuAD script
    evaluate_options = EVAL_OPTS(data_file=dev_file,
                                 pred_file=output_prediction_file,
                                 na_prob_file=output_null_log_odds_file)
    results = evaluate_on_squad(evaluate_options)
    print(results)


def cw_tree_attack():
    dataset, examples, features = load_and_cache_examples(dev_file, tokenizer)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size)

    embed = torch.load(args.word_vector)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    vocab = Vocab(filename=args.dictionary, data=[PAD_WORD, UNK_WORD, EOS_WORD, SOS_WORD])
    generator = Generator(args.test_data, vocab=vocab, embed=embed)
    bert_transfered_embedding = torch.load('cased_bert_transfered_embedding.pth')
    transfer_emb = torch.nn.Embedding(bert_transfered_embedding.size(0), bert_transfered_embedding.size(1)).to(device)
    # transfer_emb = torch.nn.Embedding.from_pretrained(bert_transfered_embedding).to(device)
    transfer_emb.weight.data.copy_(bert_transfered_embedding)
    seqback = WrappedSeqback(embed, device, attack=True, seqback_model=generator.seqback_model, vocab=vocab,
                             transfer_emb=transfer_emb)
    treelstm = generator.tree_model
    generator.load_state_dict(torch.load(args.load_ae))

    all_results = []
    cw = CarliniL2_qa(debug=args.debugging)
    targeted_success = 0
    untargeted_success = 0
    tot = 0
    adv_text = []

    class TreeModel(nn.Module):
        def __init__(self):
            super(TreeModel, self).__init__()
            self.inputs = None

        def forward(self, hidden):
            self.embedding = seqback(hidden)
            self.inputs['perturbed'] = self.embedding
            return model(**self.inputs)

        def set_temp(self, temp):
            seqback.temp = temp

        def get_embedding(self):
            return self.embedding

        def get_seqback(self):
            return seqback

    tree_model = TreeModel()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]
                      }
            example_indices = batch[3]
            outputs = model(**inputs)

        # append origin
        with torch.no_grad():
            example_id = example_indices.item()
            example_id = features[example_id].example_index
            append_info = append_input(inputs, [example_id], vocab)
            append_inputs = append_info['append_inputs']
            batch_add_start = append_info['add_start']
            batch_add_end = append_info['add_end']
            batch_start_target = torch.LongTensor(append_info['target_start']).to(device)
            batch_end_target = torch.LongTensor(append_info['target_end']).to(device)
            add_sents = append_info['append_sent']

            append_outputs = model(**append_inputs)

        # get adv sent
        input_embedding = model.bert.embeddings.word_embeddings(append_inputs['input_ids'])
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
        append_inputs['perturbed'] = input_embedding
        tree_model.inputs = append_inputs
        cw.num_classes = append_info['tot_length']
        cw.run(tree_model, hidden, (batch_start_target, batch_end_target), input_token=input_embedding)
        seqback.adv_sent = []
        # re-test
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            if bi in cw.o_best_sent:
                for si, word in enumerate(cw.o_best_sent[bi]):
                    if word in tokenizer.vocab:
                        pass
                    elif word.capitalize() in tokenizer.vocab:
                        cw.o_best_sent[bi][si] = word.capitalize()
                ae_words = cw.o_best_sent[bi]
                cw.o_best_sent[bi] = tokenizer.convert_tokens_to_ids(ae_words)
                append_inputs['input_ids'].data[bi, add_start:add_end] = torch.LongTensor(cw.o_best_sent[bi])
        append_inputs['perturbed'] = None
        adv_outputs = model(**append_inputs)
        adv_start_predict = to_list(torch.max(adv_outputs[0], 1)[1])
        adv_end_predict = to_list(torch.max(adv_outputs[1], 1)[1])

        orig_start_predict = to_list(torch.max(outputs[0], 1)[1])
        orig_end_predict = to_list(torch.max(outputs[1], 1)[1])
        append_start_predict = to_list(torch.max(append_outputs[0], 1)[1])
        append_end_predict = to_list(torch.max(append_outputs[1], 1)[1])

        for bi, (start_target, end_target) in enumerate(zip(batch_start_target, batch_end_target)):
            start_output = adv_start_predict[bi]
            end_output = adv_end_predict[bi]
            targeted_success += int(compare(start_output, start_target.item(), end_output, end_target.item()))
            untargeted_success += int(
                compare_untargeted(start_output, start_target.item(), end_output, end_target.item()))

        for i in range(len(cw.o_best_sent)):
            try:
                logger.info(("orig:", transform(add_sents[i])))
                logger.info(("adv:", transform(cw.o_best_sent[i])))
            except:
                continue
        tot += 1
        logger.info(("orig start predict", orig_start_predict[0]))
        logger.info(("orig end predict", orig_end_predict[0]))
        logger.info(("orig append start predict", append_start_predict[0]))
        logger.info(("orig append end predict", append_end_predict[0]))
        logger.info(("adv append start predict", adv_start_predict[0]))
        logger.info(("adv append end predict", adv_end_predict[0]))
        logger.info(("targeted successful rate:", targeted_success))
        logger.info(("untargetd successful rate:", untargeted_success))
        logger.info(("tot:", tot))

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = RawResult(unique_id=unique_id,
                               start_logits=to_list(outputs[0][i]),
                               end_logits=to_list(outputs[1][i]))
            all_results.append(result)
            try:
                adv_text.append({'adv_text': cw.o_best_sent[i],
                                 'qas_id': examples[eval_feature.example_index].qas_id,
                                 'adv_predict': (adv_start_predict[0], adv_end_predict[0]),
                                 'orig_predict': (orig_start_predict[0], orig_end_predict[0]),
                                 'orig_append_predict': (append_start_predict[0], append_end_predict[0])
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')
            except:
                adv_text.append({'adv_text': add_sents[i],
                                 'qas_id': examples[eval_feature.example_index].qas_id,
                                 'adv_predict': (adv_start_predict[0], adv_end_predict[0]),
                                 'orig_predict': (orig_start_predict[0], orig_end_predict[0]),
                                 'orig_append_predict': (append_start_predict[0], append_end_predict[0])
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')

    # Compute predictions
    output_prediction_file = "predictions_.json"
    output_nbest_file = "nbest_predictions_.json"

    write_predictions(examples, features, all_results, 20,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, False,
                      False, null_score_diff_threshold)

    # Evaluate with the official SQuAD script
    evaluate_options = EVAL_OPTS(data_file=dev_file,
                                 pred_file=output_prediction_file,
                                 na_prob_file=output_null_log_odds_file)
    results = evaluate_on_squad(evaluate_options)
    print(results)


def cw_tree_attack_target():
    dataset, examples, features = load_and_cache_examples(dev_file, tokenizer)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size)

    embed = torch.load(args.word_vector)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    vocab = Vocab(filename=args.dictionary, data=[PAD_WORD, UNK_WORD, EOS_WORD, SOS_WORD])
    generator = Generator(args.test_data, vocab=vocab, embed=embed)
    bert_transfered_embedding = torch.load('cased_bert_transfered_embedding.pth')
    transfer_emb = torch.nn.Embedding(bert_transfered_embedding.size(0), bert_transfered_embedding.size(1)).to(device)
    # transfer_emb = torch.nn.Embedding.from_pretrained(bert_transfered_embedding).to(device)
    transfer_emb.weight.data.copy_(bert_transfered_embedding)
    seqback = WrappedSeqback(embed, device, attack=True, seqback_model=generator.seqback_model, vocab=vocab,
                             transfer_emb=transfer_emb)
    treelstm = generator.tree_model
    generator.load_state_dict(torch.load(args.load_ae))

    all_results = []
    cw = CarliniL2_qa(debug=args.debugging)
    targeted_success = 0
    untargeted_success = 0
    tot = 0
    adv_text = []

    class TreeModel(nn.Module):
        def __init__(self):
            super(TreeModel, self).__init__()
            self.inputs = None

        def forward(self, hidden):
            self.embedding = seqback(hidden)
            self.inputs['perturbed'] = self.embedding
            return model(**self.inputs)

        def set_temp(self, temp):
            seqback.temp = temp

        def get_embedding(self):
            return self.embedding

        def get_seqback(self):
            return seqback

    tree_model = TreeModel()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]
                      }
            example_indices = batch[3]
            outputs = model(**inputs)

        # append origin
        with torch.no_grad():
            example_id = example_indices.item()
            example_id = features[example_id].example_index
            append_info = append_input(inputs, [example_id], vocab)
            append_inputs = append_info['append_inputs']
            batch_add_start = append_info['add_start']
            batch_add_end = append_info['add_end']
            batch_start_target = torch.LongTensor(append_info['target_start']).to(device)
            batch_end_target = torch.LongTensor(append_info['target_end']).to(device)
            add_sents = append_info['append_sent']

            append_outputs = model(**append_inputs)

        # get adv sent
        input_embedding = model.bert.embeddings.word_embeddings(append_inputs['input_ids'])
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
            seqback.target = sentences[0][seqback.target_start:seqback.target_end + 1]
            trees = [append_info['tree'][bi]]
            tree_embedding = treelstm(sentences, trees)[0][0].detach()
            batch_tree_embedding.append(tree_embedding)
        hidden = torch.cat(batch_tree_embedding, dim=0)
        cw.batch_info = append_info
        append_inputs['perturbed'] = input_embedding
        tree_model.inputs = append_inputs
        cw.num_classes = append_info['tot_length']
        cw.run(tree_model, hidden, (batch_start_target, batch_end_target), input_token=input_embedding)
        seqback.adv_sent = []
        # re-test
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            if bi in cw.o_best_sent:
                for si, word in enumerate(cw.o_best_sent[bi]):
                    if word in tokenizer.vocab:
                        pass
                    elif word.capitalize() in tokenizer.vocab:
                        cw.o_best_sent[bi][si] = word.capitalize()
                ae_words = cw.o_best_sent[bi]
                cw.o_best_sent[bi] = tokenizer.convert_tokens_to_ids(ae_words)
                append_inputs['input_ids'].data[bi, add_start:add_end] = torch.LongTensor(cw.o_best_sent[bi])
        append_inputs['perturbed'] = None
        adv_outputs = model(**append_inputs)
        adv_start_predict = to_list(torch.max(adv_outputs[0], 1)[1])
        adv_end_predict = to_list(torch.max(adv_outputs[1], 1)[1])

        orig_start_predict = to_list(torch.max(outputs[0], 1)[1])
        orig_end_predict = to_list(torch.max(outputs[1], 1)[1])
        append_start_predict = to_list(torch.max(append_outputs[0], 1)[1])
        append_end_predict = to_list(torch.max(append_outputs[1], 1)[1])

        for bi, (start_target, end_target) in enumerate(zip(batch_start_target, batch_end_target)):
            start_output = adv_start_predict[bi]
            end_output = adv_end_predict[bi]
            targeted_success += int(compare(start_output, start_target.item(), end_output, end_target.item()))
            untargeted_success += int(
                compare_untargeted(start_output, start_target.item(), end_output, end_target.item()))

        for i in range(len(cw.o_best_sent)):
            try:
                logger.info(("orig:", transform(add_sents[i])))
                logger.info(("adv:", transform(cw.o_best_sent[i])))
            except:
                continue
        tot += 1
        logger.info(("orig start predict", orig_start_predict[0]))
        logger.info(("orig end predict", orig_end_predict[0]))
        logger.info(("orig append start predict", append_start_predict[0]))
        logger.info(("orig append end predict", append_end_predict[0]))
        logger.info(("adv append start predict", adv_start_predict[0]))
        logger.info(("adv append end predict", adv_end_predict[0]))
        logger.info(("targeted successful rate:", targeted_success))
        logger.info(("untargetd successful rate:", untargeted_success))
        logger.info(("tot:", tot))

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = RawResult(unique_id=unique_id,
                               start_logits=to_list(outputs[0][i]),
                               end_logits=to_list(outputs[1][i]))
            all_results.append(result)
            try:
                adv_text.append({'adv_text': cw.o_best_sent[i],
                                 'qas_id': examples[eval_feature.example_index].qas_id,
                                 'adv_predict': (adv_start_predict[0], adv_end_predict[0]),
                                 'orig_predict': (orig_start_predict[0], orig_end_predict[0]),
                                 'orig_append_predict': (append_start_predict[0], append_end_predict[0])
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')
            except:
                adv_text.append({'adv_text': add_sents[i],
                                 'qas_id': examples[eval_feature.example_index].qas_id,
                                 'adv_predict': (adv_start_predict[0], adv_end_predict[0]),
                                 'orig_predict': (orig_start_predict[0], orig_end_predict[0]),
                                 'orig_append_predict': (append_start_predict[0], append_end_predict[0])
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')

    # Compute predictions
    output_prediction_file = "predictions_.json"
    output_nbest_file = "nbest_predictions_.json"

    write_predictions(examples, features, all_results, 20,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, False,
                      False, null_score_diff_threshold)

    # Evaluate with the official SQuAD script
    evaluate_options = EVAL_OPTS(data_file=dev_file,
                                 pred_file=output_prediction_file,
                                 na_prob_file=output_null_log_odds_file)
    results = evaluate_on_squad(evaluate_options)
    print(results)


def cw_tree_attack_ensemble():
    dataset, examples, features = load_and_cache_examples(dev_file, tokenizer)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size)

    embed = torch.load(args.word_vector)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    vocab = Vocab(filename=args.dictionary, data=[PAD_WORD, UNK_WORD, EOS_WORD, SOS_WORD])
    generator = Generator(args.test_data, vocab=vocab, embed=embed)
    bert_transfered_embedding = torch.load('cased_bert_transfered_embedding.pth')
    transfer_emb = torch.nn.Embedding(bert_transfered_embedding.size(0), bert_transfered_embedding.size(1)).to(device)
    # transfer_emb = torch.nn.Embedding.from_pretrained(bert_transfered_embedding).to(device)
    transfer_emb.weight.data.copy_(bert_transfered_embedding)
    seqback = WrappedSeqback(embed, device, attack=True, seqback_model=generator.seqback_model, vocab=vocab,
                             transfer_emb=transfer_emb)
    treelstm = generator.tree_model
    generator.load_state_dict(torch.load(args.load_ae))

    all_results = []
    cw = CarliniL2_qa(debug=args.debugging)
    targeted_success = 0
    untargeted_success = 0
    tot = 0
    adv_text = []

    class TreeModel(nn.Module):
        def __init__(self, bert_model):
            super(TreeModel, self).__init__()
            self.model = bert_model
            self.inputs = None

        def forward(self, hidden):
            self.embedding = seqback(hidden)
            self.inputs['perturbed'] = self.embedding
            return self.model(**self.inputs)

        def set_temp(self, temp):
            seqback.temp = temp

        def get_embedding(self):
            return self.embedding

        def get_seqback(self):
            return seqback

    tree_model = TreeModel(model)
    tree_model2 = TreeModel(model1)
    tree_model3 = TreeModel(model2)

    models = [tree_model, tree_model2, tree_model3]
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]
                      }
            example_indices = batch[3]
            outputs = model(**inputs)

        # append origin
        with torch.no_grad():
            example_id = example_indices.item()
            example_id = features[example_id].example_index
            append_info = append_input(inputs, [example_id], vocab)
            append_inputs = append_info['append_inputs']
            batch_add_start = append_info['add_start']
            batch_add_end = append_info['add_end']
            batch_start_target = torch.LongTensor(append_info['target_start']).to(device)
            batch_end_target = torch.LongTensor(append_info['target_end']).to(device)
            add_sents = append_info['append_sent']

            append_outputs = model(**append_inputs)

        # get adv sent
        input_embedding = model.bert.embeddings.word_embeddings(append_inputs['input_ids'])
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
        append_inputs['perturbed'] = input_embedding
        tree_model.inputs = append_inputs
        cw.num_classes = append_info['tot_length']
        cw.run(models, hidden, (batch_start_target, batch_end_target), input_token=input_embedding)
        seqback.adv_sent = []
        # re-test
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            if bi in cw.o_best_sent:
                ae_words = cw.o_best_sent[bi]
                cw.o_best_sent[bi] = tokenizer.convert_tokens_to_ids(ae_words)
                append_inputs['input_ids'].data[bi, add_start:add_end] = torch.LongTensor(cw.o_best_sent[bi])
        append_inputs['perturbed'] = None
        adv_outputs = model(**append_inputs)
        adv_start_predict = to_list(torch.max(adv_outputs[0], 1)[1])
        adv_end_predict = to_list(torch.max(adv_outputs[1], 1)[1])

        orig_start_predict = to_list(torch.max(outputs[0], 1)[1])
        orig_end_predict = to_list(torch.max(outputs[1], 1)[1])
        append_start_predict = to_list(torch.max(append_outputs[0], 1)[1])
        append_end_predict = to_list(torch.max(append_outputs[1], 1)[1])

        for bi, (start_target, end_target) in enumerate(zip(batch_start_target, batch_end_target)):
            start_output = adv_start_predict[bi]
            end_output = adv_end_predict[bi]
            targeted_success += int(compare(start_output, start_target.item(), end_output, end_target.item()))
            untargeted_success += int(
                compare_untargeted(start_output, start_target.item(), end_output, end_target.item()))

        for i in range(len(cw.o_best_sent)):
            try:
                logger.info(("orig:", transform(add_sents[i])))
                logger.info(("adv:", transform(cw.o_best_sent[i])))
            except:
                continue
        tot += 1
        logger.info(("orig start predict", orig_start_predict[0]))
        logger.info(("orig end predict", orig_end_predict[0]))
        logger.info(("orig append start predict", append_start_predict[0]))
        logger.info(("orig append end predict", append_end_predict[0]))
        logger.info(("adv append start predict", adv_start_predict[0]))
        logger.info(("adv append end predict", adv_end_predict[0]))
        logger.info(("targeted successful rate:", targeted_success))
        logger.info(("untargetd successful rate:", untargeted_success))
        logger.info(("tot:", tot))

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = RawResult(unique_id=unique_id,
                               start_logits=to_list(outputs[0][i]),
                               end_logits=to_list(outputs[1][i]))
            all_results.append(result)
            try:
                adv_text.append({'adv_text': cw.o_best_sent[i],
                                 'qas_id': examples[eval_feature.example_index].qas_id,
                                 'adv_predict': (adv_start_predict[0], adv_end_predict[0]),
                                 'orig_predict': (orig_start_predict[0], orig_end_predict[0]),
                                 'orig_append_predict': (append_start_predict[0], append_end_predict[0])
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')
            except:
                adv_text.append({'adv_text': add_sents[i],
                                 'qas_id': examples[eval_feature.example_index].qas_id,
                                 'adv_predict': (adv_start_predict[0], adv_end_predict[0]),
                                 'orig_predict': (orig_start_predict[0], orig_end_predict[0]),
                                 'orig_append_predict': (append_start_predict[0], append_end_predict[0])
                                 })
                joblib.dump(adv_text, root_dir + '/adv_text.pkl')

    # Compute predictions
    output_prediction_file = "predictions_.json"
    output_nbest_file = "nbest_predictions_.json"

    write_predictions(examples, features, all_results, 20,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, False,
                      False, null_score_diff_threshold)

    # Evaluate with the official SQuAD script
    evaluate_options = EVAL_OPTS(data_file=dev_file,
                                 pred_file=output_prediction_file,
                                 na_prob_file=output_null_log_odds_file)
    results = evaluate_on_squad(evaluate_options)
    print(results)


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
                new_context = paragraph['context'] + " " + adv_dic[qa['id']][-1]
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
                new_context = adv_dic[qa['id']][-1]
                new_pqa = {'context': new_context, 'qas': [qa.copy()]}
                new_article['paragraphs'].append(new_pqa)
        output_data['data'].append(new_article)

    with open(root_dir + '/adv_text_{}'.format(input_file), "w") as f:
        json.dump(output_data, f)


def get_adv_dic(adv_data):
    result = {}
    for data in adv_data:
        if data['qas_id'] in result:
            result[data['qas_id']].append(transform(data['adv_text']))
        else:
            result[data['qas_id']] = [transform(data['adv_text'])]
    return result


if __name__ == '__main__':
    max_seq_length = 384
    model_name = "bert-base-cased"
    do_lower_case = False
    dev_file = args.dev_file
    doc_stride = 128
    max_query_length = 64
    device = torch.device("cuda:{}".format(args.device))
    checkpoint = "../bertQA/bertQA/result/official"
    checkpoint1 = "../bertQA/bertQA/result/official1"
    checkpoint2 = "../bertQA/bertQA/result/official2"
    device1 = torch.device("cuda:{}".format(args.device))
    device2 = torch.device("cuda:{}".format(args.device))
    null_score_diff_threshold = 0
    max_answer_length = 30

    model = bertmodel.BertForQuestionAnswering.from_pretrained(checkpoint).to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
    output_null_log_odds_file = None

    answer_append_sentences = joblib.load('sampled_perturb_answer_sentences.pkl')
    question_append_sentences = joblib.load('sampled_perturb_question_sentences.pkl')

    features_with_answers = joblib.load('features_with_answers.pkl')
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    if args.model == 'word_qa':
        """
            Choose the attack method
        """
        # cw_word_attack_target()
        # cw_word_attack()
        if args.targeted:
            cw_random_word_attack_position_targeted()
        else:
            cw_random_word_attack_untargeted()
        # model1 = bertmodel.BertForQuestionAnswering.from_pretrained(checkpoint1).to(device1)
        # model1.eval()
        # model2 = bertmodel.BertForQuestionAnswering.from_pretrained(checkpoint2).to(device2)
        # model2.eval()
        # cw_word_attack_ensemble()
        adv_data = joblib.load(root_dir + '/adv_text.pkl')
        adv_dict = get_adv_dic(adv_data)
        # generate_adv_dataset(dev_file, adv_dic=adv_dict)
        generate_adv_dataset_for_random_words(dev_file, adv_dic=adv_dict)
        dev_file = root_dir + '/adv_text_{}'.format(dev_file)
        eval(load_cache=False)
    elif args.model == 'tree_qa':
        # cw_tree_attack()
        cw_tree_attack_target()
        adv_data = joblib.load(root_dir + '/adv_text.pkl')
        adv_dict = get_adv_dic(adv_data)
        generate_adv_dataset(dev_file, adv_dic=adv_dict)
        dev_file = root_dir + '/adv_text_{}'.format(dev_file)
        eval(load_cache=False)
    else:
        eval(load_cache=False)
