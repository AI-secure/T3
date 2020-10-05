from fastNLP import Instance, DataSet, Vocabulary, Const
import argparse
import pickle
import re
from pytorch_transformers import BertTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--train", default="yelp/full-trn-processed.pkl", dest="train", help=".pkl file to use")
parser.add_argument("--dev", default="yelp/full-val-processed.pkl", dest="dev", help=".pkl file to use")
parser.add_argument("--test", default="yelp/full-tst-processed.pkl", dest="test", help=".pkl file to use")
parser.add_argument("--model", dest="model", default="bert-base-uncased", help="tokenizer")
options = parser.parse_args()

train_data = pickle.load(open(options.train, "rb"))
dev_data = pickle.load(open(options.dev, "rb"))
test_data = pickle.load(open(options.test, "rb"))
tokenizer = BertTokenizer.from_pretrained(options.model)


def make_dataset(data):
    dataset = DataSet()
    tot = 0
    for x in data:

        seq = "[CLS] " + x["raw_text"]
        seq = tokenizer.encode(seq)
        """
        seq=["[CLS]"]+word_tokenize(x["raw_text"])
        seq=tokenizer.convert_tokens_to_ids(seq)
        """
        if len(seq) > 512:
            seq = seq[:512]
            tot += 1
            # print(x["raw_text"])
            # print()

        label = int(x["label"])
        ins = Instance(origin=x["raw_text"], seq=seq, label=label, seq_len=len(seq))
        dataset.append(ins)

    dataset.set_input("seq", "seq_len")
    dataset.set_target("label")
    print(dataset[5])
    print("number:", len(dataset), tot)
    print()
    return dataset


out = {}
out["train_set"] = make_dataset(train_data)
out["dev_set"] = make_dataset(dev_data)
out["test_set"] = make_dataset(test_data)

with open("data.pkl", "wb") as outfile:
    pickle.dump(out, outfile)
