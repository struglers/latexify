from os.path import join
import pickle as pkl
from collections import Counter
import argparse
import csv

START_TOKEN = 0
PAD_TOKEN = 1
END_TOKEN = 2
UNK_TOKEN = 3

# buid sign2id


class Vocab(object):
    def __init__(self):
        self.sign2id = {"<s>": START_TOKEN, "</s>": END_TOKEN,
                        "<pad>": PAD_TOKEN, "<unk>": UNK_TOKEN}
        self.id2sign = dict((idx, token)
                            for token, idx in self.sign2id.items())
        self.length = 4

    def add_sign(self, sign):
        if sign not in self.sign2id:
            self.sign2id[sign] = self.length
            self.id2sign[self.length] = sign
            self.length += 1

    def __len__(self):
        return self.length


def build_vocab(data_dir, min_count=10):
    """
    traverse training formulas to make vocab
    and store the vocab in the file
    """
    vocab = Vocab()
    counter = Counter()

    with open(join(data_dir, 'im2latex_train.csv'), 'r') as csvfile:
        f = csv.reader(csvfile)
        next(f, None) #Skip csv file heading
        for line in f:
            formula, _ = line
            counter.update(formula.split())

    for word, count in counter.most_common():
        if count >= min_count:
            vocab.add_sign(word)

    vocab_file = join(data_dir, 'vocab.pkl')
    print("Writing Vocab File in ", vocab_file)
    with open(vocab_file, 'wb') as w:
        pkl.dump(vocab, w)


def load_vocab(data_dir):
    with open(join(data_dir, 'vocab.pkl'), 'rb') as f:
        vocab = pkl.load(f)
    print("Load vocab including {} words!".format(len(vocab)))
    return vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Building vocab for Im2Latex")
    parser.add_argument("--data_path", type=str,
                        default="./data/", help="The dataset's dir")
    parser.add_argument("--min_count", type=int,
                        default=10, help="Minimum number of times a token must occur to be recorded in the vocabulary")
    args = parser.parse_args()
    vocab = build_vocab(args.data_path, min_count=args.min_count)
