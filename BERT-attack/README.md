# Attack BERT Model

This repo contains the code to attack both BERT-QA and BERT-classification.

## Prepare BERT

Our BERT is fine-tuned on different datasets based on [Transformers](https://github.com/huggingface/transformers).

### BERT classification
We put our BERT-classification code in `bert_classification` directory.  You can reproduce the results by first running:

```
cd bert_classification
python make_dataset.py
```
Then 
```
python main.py --dataset data_bert-base-uncased.pkl --task-name XXX --devi 3210 --lr 2e-5 --num-epochs 4 --batch-size 32 --dropout 0.1
```

### BERT QA
We put BERT-QA code in the root directory. You can fine-tune BERT on QA dataset by running: 
```
python run_squad.py --train_file train-v1.1.json --predict_file dev-v1.1.json --model_type bert --model_name_or_path bert-base-cased --output_dir official1 --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 3 --per_gpu_eval_batch_size 4 --learning_rate 5e-5 --adam_epsilon 1e-6 --num_train_epochs 3 --save_steps 3000 --logging_steps 3000 --eval_all_checkpoints
```

## Attack 

You may read our `attack_qa.py` and `attack_classifier.py` for more information. You may also try different attack scenarios (position targeted attack/answer targeted attack) and different attack methods (word-level/sentence-level) to see the effects.
