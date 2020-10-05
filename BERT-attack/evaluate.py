from bertmodel import BertForQuestionAnswering
from pytorch_transformers import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import torch
import torch.nn as nn
import argparse
import os,sys
import numpy as np
from utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions)
from utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad
from tqdm import tqdm

max_seq_length=384
model_name="bert-base-cased"
do_lower_case=False
dev_file="dev-v1.1.json"
doc_stride=128
max_query_length=64
device=torch.device("cuda:{}".format(4))  
checkpoint="result/official"
null_score_diff_threshold=0
max_answer_length=30

def to_list(tensor):
    return tensor.detach().cpu().tolist()
    
def load_and_cache_examples(input_file, tokenizer):
    cached_features_file = 'Ecached_dev_{}_{}'.format(model_name,str(max_seq_length))
    """ 
    if os.path.exists(cached_features_file):
        #print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    """
    print("Creating features from dataset file at %s", input_file)
    examples = read_squad_examples(input_file=input_file,
                                            is_training=False,
                                            version_2_with_negative=False)
    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            max_seq_length=max_seq_length,
                                            doc_stride=doc_stride,
                                            max_query_length=max_query_length,
                                            is_training=False)
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
    

model = BertForQuestionAnswering.from_pretrained(checkpoint)
model.to(device)
tokenizer=BertTokenizer.from_pretrained(model_name,do_lower_case=do_lower_case)
batch_size=8
output_null_log_odds_file=None
dataset, examples, features = load_and_cache_examples(dev_file, tokenizer)

# Note that DistributedSampler samples randomly
eval_sampler = SequentialSampler(dataset) 
eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)

all_results = []
for batch in tqdm(eval_dataloader, desc="Evaluating"):
    model.eval()
    batch = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2]
                  }
        example_indices = batch[3]
       
        outputs = model(**inputs)

    for i, example_index in enumerate(example_indices):
        eval_feature = features[example_index.item()]
        unique_id = int(eval_feature.unique_id)
        result = RawResult(unique_id    = unique_id,
                           start_logits = to_list(outputs[0][i]),
                           end_logits   = to_list(outputs[1][i]))
        all_results.append(result)

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
 