from fmtl import FMTL
from utils import *
import pickle as pkl
import argparse

parser = argparse.ArgumentParser(description='Hierarchical Attention Networks for Document Classification')
    
parser.add_argument("--output", type=str)
parser.add_argument('filename', type=str)
args = parser.parse_args()






def to_str(ll):
    return " ".join(["".join(lc) for lc in ll])


datadict = pkl.load(open(args.filename,"rb"))

for split in range(5):
    data_tl,(trainit,valit,testit) = FMTL_train_val_test(datadict["data"],datadict["splits"],split,validation=0.5,rows=datadict["rows"])
    print(datadict["rows"])
    #label_mapping = data_tl.get_field_dict("label",key_iter=trainit) #creates class mapping
    label_mapping = {"INCONNU":1}
    data_tl.set_mapping("label",label_mapping,unk=0) 

    train_file = f'{args.filename}_{split}.train'
    test_file = f'{args.filename}_{split}.test'
    

    print("Train set class stats:\n" + 10*"-")
    _,_ = data_tl.get_stats("label",trainit,True)


    # for _,txt,label in tqdm(data_tl.indexed_iter(trainit)):
    #     #print(to_str(txt))
    #     line_train = f"__label__{label} {to_str(txt)} \n"
    #     with open(train_file, 'a+') as f:
    #         f.write(line_train)

    # for _,txt,label in tqdm(data_tl.indexed_iter(testit)):
    #     line_test = f"__label__{label} {to_str(txt)} \n"
    #     with open(test_file, 'a+') as f:
    #         f.write(line_test)

    with open(train_file, 'w') as f:
        for _,txt,label in tqdm(data_tl.indexed_iter(trainit)):
            #print(to_str(txt))
            line_train = f"__label__{label} {to_str(txt)} \n"
            f.write(line_train)

    with open(test_file, 'w') as f:
        for _,txt,label in tqdm(data_tl.indexed_iter(testit)):
            line_test = f"__label__{label} {to_str(txt)} \n"
            f.write(line_test)

