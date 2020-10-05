#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import nltk
import numpy as np
import torch
import joblib
from pytorch_transformers import BertTokenizer
from pytorch_transformers.tokenization_bert import  whitespace_tokenize
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


# In[15]:


import bertmodel
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from utils_squad import convert_examples_to_features, RawResult, write_predictions, SquadExample


# In[12]:


import json
def read_squad_examples(input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
            
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    return examples

def load_and_cache_examples(input_file, tokenizer):
    cached_features_file = 'Ecached_dev_{}_{}'.format(model_name, str(max_seq_length))
    """ 
    if os.path.exists(cached_features_file):
        #print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    """
    print("Creating features from dataset file at %s", input_file)
    examples = read_squad_examples(input_file=input_file,
                                    is_training=True,
                                   version_2_with_negative=False)
    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            max_seq_length=max_seq_length,
                                            doc_stride=doc_stride,
                                            max_query_length=max_query_length,
                                            is_training=True)
    print("Saving features into cached file %s", cached_features_file)
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


# In[16]:


max_seq_length = 384
model_name = "bert-base-cased"
do_lower_case = False
dev_file = "none_n1000_k1_s0.json"
doc_stride = 128
max_query_length = 64
null_score_diff_threshold = 0
max_answer_length = 30

tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
output_null_log_odds_file = None
device = torch.device("cuda:0")


# In[17]:


dataset, examples, features = load_and_cache_examples(dev_file, tokenizer)
eval_sampler = SequentialSampler(dataset)


# In[7]:


eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=1)


# In[8]:


from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz")
predictor._model = predictor._model.to(device)


# In[9]:


srls = {}
for batch in tqdm(eval_dataloader, desc="Evaluating"):
    example_indices = batch[3]
    for example_idx in example_indices:
        eval_feature = features[example_idx.item()]
        unique_id = int(eval_feature.unique_id)
        srls[eval_feature.example_index] = predictor.predict(sentence=examples[eval_feature.example_index].question_text)


# In[10]:


interrogative = {"which", "what", "who", "whom", "whose", "where", "when", "why", "how"}
ARG0 = {'B-R-ARG0', 'B-ARG0', 'I-C-ARG0', 'I-R-ARG0', 'I-ARG0', 'B-C-ARG0'}
ARG1 = {'B-R-ARG1', 'B-ARG1', 'I-C-ARG1', 'I-R-ARG1', 'I-ARG1', 'B-C-ARG1'}
ARG2 = {'I-C-ARG2', 'B-R-ARG2', 'I-R-ARG2', 'B-ARG2', 'B-C-ARG2', 'I-ARG2'}
verb = {'B-V'}
AuxiliaryVerb = {'did', 'do', 'does', "don't", 'have', 'has', 'had'}
find_and_concat = [{'B-R-ARGM-LOC', 'B-ARGM-LOC', 'I-R-ARGM-LOC', 'I-ARGM-LOC'},
                        {'B-R-ARGM-TMP', 'I-ARGM-TMP', 'B-ARGM-TMP', 'I-R-ARGM-TMP'},
                        {'I-ARGM-CAU', 'B-R-ARGM-CAU', 'B-ARGM-CAU'},
                        {'B-ARGM-EXT', 'I-ARGM-EXT'},
                        {'I-ARGM-DIR', 'B-ARGM-DIR'},
                        {'I-ARG4', 'I-C-ARG4', 'B-C-ARG4', 'B-ARG4'},
                        {'I-R-ARGM-MNR', 'B-ARGM-MNR', 'B-R-ARGM-MNR', 'I-ARGM-MNR', 'I-C-ARGM-MNR'},
                        {'B-R-ARG3', 'B-C-ARG3', 'B-ARG3', 'I-C-ARG3', 'I-ARG3'},
                        {'I-ARGM-PRP', 'B-ARGM-PRP'},
                        {'B-ARGM-COM', 'I-ARGM-COM'},
                        {'B-ARGM-GOL', 'I-ARGM-GOL'},
                        {'B-ARGM-PNC', 'I-ARGM-PNC'},
                        {'I-ARGM-PRD', 'B-ARGM-PRD'}]


# In[11]:


remained_tags = ARG0 | ARG1 | ARG2 | verb
ARGS = [ARG0, ARG1, ARG2, verb]
belong = {}
for arg in ARGS:
    for x in arg:
        belong[x] = arg


# In[17]:


def generate_question_append(srl_data, add_start=0):
    answer_words = ['[UNK]', '[UNK]', '[UNK]']
    srl = srl_data['verbs']
    question = srl_data['words']
    if len(question) == 1:
        print(question)
        return None, None, None
    level = 0
    while srl[level]['verb'] in AuxiliaryVerb or srl[level]['tags'][0] == 'O':
        # append to last
        level += 1
        if len(srl) <= level:
            i = 0
            tags = srl[0]['tags']
            while tags[i] not in verb:
                i += 1
            add_words = question[i + 1:-1]
            return add_words + answer_words + ['.'],                    add_start + len(add_words), add_start + len(add_words) + len(answer_words)
     
    
    if srl[0]['verb'] in AuxiliaryVerb or srl[0]['tags'][0] == 'O':
        verbs = srl[level]
        add_words = []
        # manner/loc/temp
        for i, word in enumerate(question):
            if verbs['tags'][i] in remained_tags:
                if word.lower() in interrogative:
                    j = i + 1
                    question_tag = verbs['tags'][i]
                    while verbs['tags'][j] in belong[question_tag]:
                        verbs['tags'][j] = 'O'
                        j += 1
                else:
                    add_words.append(word)
        return add_words + answer_words + ['.'], add_start + len(add_words), add_start + len(add_words) + len(answer_words) - 1

    else:
         # append to first
        tags = srl[level]['tags']
        question_tag = tags[0]
        i = 1
        while tags[i] == question_tag:
            i += 1
        add_words = answer_words + question[i:-1] + ['.']
        return add_words, add_start, add_start + len(answer_words) - 1


# In[25]:


from gensim.models import KeyedVectors
glove_model = KeyedVectors.load_word2vec_format("bertqa_vectors.kv")


# In[26]:


glove_model.most_similar(positive='during', topn=2)


# In[27]:


def perturb_words(add_words):
    perturbed = []
    for add_word in add_words:
        try:
            similar_words = glove_model.most_similar(positive=add_word, topn=3)
        except:
            similar_words = glove_model.most_similar(positive='[UNK]', topn=3)
        perturbed_word = None
        for i in range(3):
            if similar_words[i][0].lower() != add_word.lower():
                perturbed_word = similar_words[i][0]
                break
        if perturbed_word == None:
            print("Cannot find similar words")
            break
        perturbed.append(perturbed_word)
    return perturbed


# In[41]:


def generate_perturbed_question_append(srl_data, add_start=0):
    answer_words = ['[UNK]', '[UNK]', '[UNK]']
    srl = srl_data['verbs']
    question = srl_data['words']
    if len(question) == 1:
        print(question)
        return None, None, None
    level = 0
    while srl[level]['verb'] in AuxiliaryVerb or srl[level]['tags'][0] == 'O':
        # append to last
        level += 1
        if len(srl) <= level:
            i = 0
            tags = srl[0]['tags']
            while tags[i] not in verb:
                i += 1
            add_words = question[i + 1:-1]
            add_words = perturb_words(add_words)
            return add_words + answer_words + ['.'],  add_start + len(add_words), add_start + len(add_words) + len(answer_words)
     
    
    if srl[0]['verb'] in AuxiliaryVerb or srl[0]['tags'][0] == 'O':
        verbs = srl[level]
        add_words = []
        # manner/loc/temp
        for i, word in enumerate(question):
            if verbs['tags'][i] in remained_tags:
                if word.lower() in interrogative:
                    j = i + 1
                    question_tag = verbs['tags'][i]
                    while verbs['tags'][j] in belong[question_tag]:
                        verbs['tags'][j] = 'O'
                        j += 1
                else:
                    add_words.append(word)
        add_words = perturb_words(add_words)
        return add_words + answer_words + ['.'], add_start + len(add_words), add_start + len(add_words) + len(answer_words) - 1

    else:
         # append to first
        tags = srl[level]['tags']
        question_tag = tags[0]
        i = 1
        while tags[i] == question_tag:
            i += 1
        add_words = answer_words + perturb_words(question[i:-1]) + ['.']
        return add_words, add_start, add_start + len(answer_words) - 1


# In[18]:


question_append_sentences = {}
for k, v in srls.items():
    if len(v['verbs']) == 0:
        append_sent, target_start, target_end = None, None, None
    else:
        try:
            append_sent, target_start, target_end = generate_question_append(v)
        except:
            traceback.print_exc()
    question_append_sentences[k] = {
        'append_sent': append_sent,
        'target_start': target_start,
        'target_end': target_end,
        'srl': v
    }


# In[20]:


joblib.dump(question_append_sentences, 'sampled_unperturbed_question_append_sentences.pkl')


# In[42]:


perturbed_question_append_sentences = {}
for k, v in srls.items():
    if len(v['verbs']) == 0:
        append_sent, target_start, target_end = None, None, None
    else:
        try:
            append_sent, target_start, target_end = generate_perturbed_question_append(v)
        except:
            traceback.print_exc()
    perturbed_question_append_sentences[k] = {
        'append_sent': append_sent,
        'target_start': target_start,
        'target_end': target_end,
        'srl': v
    }


# In[43]:


for i in range(10):
    print(perturbed_question_append_sentences[i])


# In[44]:


joblib.dump(perturbed_question_append_sentences, 'sampled_perturbed_question_append_sentences.pkl')