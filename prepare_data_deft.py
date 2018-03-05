import gzip
import argparse
import logging
import json
import pickle as pkl
import spacy
import itertools

from tqdm import tqdm
from random import randint,shuffle
from collections import Counter


def count_lines(file):
    count = 0
    for _ in file:
        count += 1
    file.seek(0)
    return count


def data_generator(data):
    with open(args.input,"r") as f:
        for x in tqdm(f,desc="Reviews",total=count_lines(f)):
            yield json.loads(x)

def to_double_list(s):
    return [list(w) for w in s.split()]
        

def build_dataset(args):

    print("Building dataset from : {}".format(args.input))
    print("-> Building {} random splits".format(args.nb_splits))

    data = [(z[0],to_double_list(z[1]),z[2]) for z in tqdm(data_generator(args.input),desc="reading file")]

    print(data[0])
    shuffle(data)

    splits = [randint(0,args.nb_splits-1) for _ in range(0,len(data))]
    count = Counter(splits)

    print("Split distribution is the following:")
    print(count)

    return {"data":data,"splits":splits,"rows":("tweet_id","text","label")}


def main(args):
    ds = build_dataset(args)
    pkl.dump(ds,open(args.output,"wb"))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str, default="sentences.pkl")
    parser.add_argument("--nb_splits",type=int, default=5)
    args = parser.parse_args()

    main(args)