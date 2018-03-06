import argparse
import pickle as pkl
from tqdm import tqdm
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
from Nets import CWAN
from fmtl import FMTL
from utils import *
import sys
import quiviz
import logging



def save(net,dic,path):
    """
    Saves a model's state and it's embedding dic by piggybacking torch's save function
    """
    dict_m = net.state_dict()
    dict_m["word_dic"] = dic    
    torch.save(dict_m,path)


def tuple_batch(l):
    """
    Prepare batch
    - Reorder reviews by length
    - Split reviews by sentences which are reordered by length
    - Build sentence ordering index to extract each sentences in training loop
    """
    def min_one(rev):
        if len(rev)==0:
            rev = [[1]]
        return rev


    _,review,rating = zip(*l)
    r_t = torch.Tensor(rating).long()
    
    list_rev = [min_one(x) for x in review]
    #word_len = [[len(w) for w in rev] for rev in list_rev ]
    chars = []
    sp = []
    len_s = []
    w_index = []
    len_c = 0

    #pour chaque tweet
    for rev in list_rev:
        chars_r = []
        iw = []
        sp_r = []
        wi = 0
        #chaque mot
        for w in rev:
            #chaque lettre
            for c in w:
                chars_r.append(c)
                sp_r.append(0)
                wi += 1
            chars_r.append(2) #space character
            sp_r.append(1)
            iw.append(wi)
            wi +=1 

        sp_r[-2] = 1 #last character.
        chars.append(chars_r[:-1])
        sp.append(sp_r[:-1])
        w_index.append(iw)
        len_s.append(len(sp_r[:-1]))
        if len(sp_r[:-1]) > len_c: #save max 
            len_c = len(sp_r[:-1])

    tweets = torch.zeros(len(chars),len_c).long()
    word_ends = torch.zeros(len(chars),len_c).long()

    len_s = torch.LongTensor(len_s)

    for i,data in enumerate(chars):
        tweets[i,0:len(data)] = torch.LongTensor(data)
        word_ends[i,0:len(data)] = torch.LongTensor(sp[i])

    max_w = torch.max(torch.sum(word_ends,-1))
    w_index_t = torch.zeros(len(chars),max_w) #index_select tensor

    for i,data in enumerate(w_index):
        w_index_t[i,0:len(data)] = (torch.LongTensor(data) + 1) #0 will be padding


    _,i = torch.sort(len_s,descending=True) #sort for RNN

    tweets = tweets[i]
    words_ends = word_ends[i]
    len_s = len_s[i].tolist()
    rating = torch.Tensor(rating)[i]
    w_index_t =  w_index_t[i]

    w_index_t += (w_index_t != 0).float() * (torch.arange(0,w_index_t.size(0)) * len_c).unsqueeze(-1)

    return tweets, word_ends, w_index_t, len_s, rating

@quiviz.log
def train(epoch,net,optimizer,dataset,criterion,cuda,optimize=False,msg="test"):
    net.train()
    epoch_loss = 0
    ok_all = 0
    data_tensors = new_tensors(4,cuda,types={0:torch.LongTensor, 1:torch.LongTensor, 2:torch.LongTensor, 3:torch.LongTensor}) #data-tensors

    with tqdm(total=len(dataset),desc="Training") as pbar:
        for iteration, (text,we,wind,lens,labels) in enumerate(dataset):

            data = tuple2var(data_tensors,(text,we,wind,labels))

            if optimize:
                optimizer.zero_grad()
            
            out = net(data[0],data[1],data[2],lens)

            ok,per,val_i = accuracy(out,data[3])
            ok_all += per.data[0]

            loss =  criterion(out, data[3]) 
            epoch_loss += loss.data[0]

            if optimize:
                loss.backward()
                optimizer.step()

            pbar.update(1)
            pbar.set_postfix({"acc":ok_all/(iteration+1),"CE":epoch_loss/(iteration+1)})

    return {f"{msg}_acc": ok_all/len(dataset)}



def load(args):

    datadict = pkl.load(open(args.filename,"rb"))
    data_tl,(trainit,valit,testit) = FMTL_train_val_test(datadict["data"],datadict["splits"],args.split,validation=0.5,rows=datadict["rows"])
    
    if args.biclass:
        label_mapping = {"INCONNU":1}
        num_class = 2
        data_tl.set_mapping("label",label_mapping,unk=0) 

    else:
        label_mapping = data_tl.get_field_dict("label",key_iter=trainit) #creates class mapping
        data_tl.set_mapping("label",label_mapping) 
        num_class = len(label_mapping)

    print(label_mapping)


    if args.load:
        state = torch.load(args.load)
        wdict = state["word_dic"]
    else:
        if args.emb:
            tensor,wdict = load_embeddings(args.emb,offset=3)
        else:     
            wdict = data_tl.get_field_dict("text",key_iter=trainit,offset=3, max_count=args.max_feat, iter_func=(lambda x: (w for s in x for w in s )))

        wdict["_pad_"] = 0
        wdict["_unk_"] = 1
        wdict["_sp_"] = 2
        data_tl.set_mapping("text",wdict,unk=1)

    print("Train set class stats:\n" + 10*"-")
    _,_ = data_tl.get_stats("label",trainit,True)

    if args.load:
        #print(state.keys())
        net = CWAN(ntoken=len(state["word_dic"]),emb_size=state["embed.weight"].size(1),hid_size=state["sent.rnn.weight_hh_l0"].size(1),num_class=state["lin_out.weight"].size(0))
        del state["word_dic"]
        net.load_state_dict(state)

    else:
        if args.emb:
            net = CWAN(ntoken=len(wdict),emb_size=len(tensor[1]),hid_size=args.hid_size,num_class=num_class)
            net.set_emb_tensor(torch.FloatTensor(tensor))
        else:
            net = CWAN(ntoken=len(wdict), emb_size=args.emb_size,hid_size=args.hid_size, num_class=num_class)

    if args.prebuild:
        data_tl = FMTL(list(x for x  in tqdm(data_tl,desc="prebuilding")),data_tl.rows)

    return data_tl,(trainit,valit,testit), net, wdict


def main(args):

    print(32*"-"+"\nHierarchical Attention Network:\n" + 32*"-")
    data_tl, (train_set, val_set, test_set), net, wdict = load(args)


    dataloader = DataLoader(data_tl.indexed_iter(train_set), batch_size=args.b_size, shuffle=True, num_workers=0, collate_fn=tuple_batch,pin_memory=True)
    dataloader_valid = DataLoader(data_tl.indexed_iter(val_set), batch_size=args.b_size, shuffle=False,  num_workers=3, collate_fn=tuple_batch)
    dataloader_test = DataLoader(data_tl.indexed_iter(test_set), batch_size=args.b_size, shuffle=False, num_workers=3, collate_fn=tuple_batch,drop_last=True)

    criterion = torch.nn.CrossEntropyLoss()      

    if args.cuda:
        net.cuda()

    print("-"*20)

    last_acc = 0

    optimizer = optim.Adam(net.parameters())
    torch.nn.utils.clip_grad_norm(net.parameters(), args.clip_grad)

    for epoch in range(1, args.epochs + 1):
        print("\n-------EPOCH {}-------".format(epoch))
        logging.info("\n-------EPOCH {}-------".format(epoch))

        train(epoch,net,optimizer,dataloader,criterion,args.cuda,optimize=True,msg="train")

        if args.snapshot:
            print("snapshot of model saved as {}".format(args.save+"_snapshot"))
            save(net,wdict,args.save+"_snapshot")

        val = train(epoch,net,optimizer, dataloader_valid,criterion, args.cuda,msg="val")
        new_acc = list(val.values())[0]

        if new_acc < last_acc:
            logging.info("---- EARLY STOPPING")
            sys.exit()
        last_acc = new_acc

        train(epoch,net,optimizer, dataloader_test,criterion, args.cuda,msg="test")

    if args.save:
        print("model saved to {}".format(args.save))
        save(net,wdict,args.save)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hierarchical Attention Networks for Document Classification')
    
    parser.add_argument("--emb-size",type=int,default=200)
    parser.add_argument("--hid-size",type=int,default=100)

    parser.add_argument("--max-feat", type=int,default=10000)
    parser.add_argument("--epochs", type=int,default=10)
    parser.add_argument("--clip-grad", type=float,default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum",type=float,default=0.9)
    parser.add_argument("--b-size", type=int, default=3)

    parser.add_argument("--emb", type=str)
    parser.add_argument("--max-words", type=int,default=-1)
    parser.add_argument("--max-sents",type=int,default=-1)

    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--load", type=str)
    parser.add_argument("--save", type=str)
    parser.add_argument("--snapshot", action='store_true')
    parser.add_argument("--prebuild",action="store_true")
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--biclass', action='store_true', help='do biclass')

    parser.add_argument("--output", type=str)
    parser.add_argument('filename', type=str)
    args = parser.parse_args()
    logging.info("========== NEW XP =============")
    main(args)
