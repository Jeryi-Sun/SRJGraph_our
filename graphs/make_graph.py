import argparse
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, help='experiment name', default='train_reco')

args = parser.parse_args()

root_data_path = '../../reco_search_data/'
def gen_neg_samples(items_with_popular, postive_item, user_rec_his, num_negs):
    count = 0
    neg_items = []
    while count < num_negs:
        neg_item = random.choice(items_with_popular)
        if  neg_item == postive_item or\
            neg_item in neg_items or\
            neg_item in user_rec_his:
            continue
        count += 1
        neg_items.append(neg_item)

    return neg_items

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
setup_seed(20210823)

def two_dim_list(x):
    for d in range(len(x)):
        x[d] = '\x02'.join([str(t) for t in x[d]])
    return "\x03".join(x)

def one_dim_list(x):
    return "\x03".join([str(t) for t in x])


def final_process(inter_data, user_vocab, reco_or_search, items_with_pops, sample_neg):
    final_data = []
    for i in tqdm(range(len(inter_data['u_id']))):
        user_id = inter_data['u_id'][i]
        item_id = inter_data['i_id'][i]
        if reco_or_search=='reco':
            query = ''
        else:
            assert reco_or_search == 'search'
            query = inter_data['query'][i]
        # user [25, 1, 10] item [1, 25, 1]
        label = inter_data['label'][i]
        if reco_or_search=='reco':
            final_data.append("\t".join([str(user_id), str(item_id), query, str(label)]))
        else:
            final_data.append("\t".join([str(user_id), str(item_id), one_dim_list(query), str(label)]))
        # for neg samples:
        if sample_neg:
            if reco_or_search=='reco':
                neg_items = gen_neg_samples(items_with_pops, item_id, user_vocab[user_id]['rec_his'], 4)
            else:
                neg_items = gen_neg_samples(items_with_pops, item_id, user_vocab[user_id]['click_items']+user_vocab[user_id]['rec_his'], 4)
            for n in neg_items:
                final_data.append("\t".join([str(user_id), str(n), '', str(0.0)]))
    return final_data

train_inter_data = pd.read_csv(os.path.join(root_data_path, 'train_inter.tsv'), sep='\t')
train_inter_data[['ts', 'i_id']] = train_inter_data[['ts', 'i_id']].astype(np.int64)
train_inter_data_dict = train_inter_data.to_dict('list')
train_src_inter_all = pd.read_csv(os.path.join(root_data_path, 'train_src_inter_all.tsv'), sep='\t').to_dict('list')
train_inter_src_data =  {'u_id':[], 'i_id':[], 'label':[], 'query':[]}
for i in range(len(train_src_inter_all['user_id'])):
    for item in eval(train_src_inter_all['click_items'][i]):
        train_inter_src_data['u_id'].append(train_src_inter_all['user_id'][i])
        train_inter_src_data['i_id'].append(item)
        train_inter_src_data['query'].append(eval(train_src_inter_all['keyword_seg'][i]))
        train_inter_src_data['label'].append(1.0)

test_inter_data = pd.read_csv(os.path.join(root_data_path, 'test_inter.tsv'), sep='\t')
test_inter_data[['ts', 'i_id']] = test_inter_data[['ts', 'i_id']].astype(np.int64)

valid_inter_data = pd.read_csv(os.path.join(root_data_path, 'valid_inter.tsv'), sep='\t')
valid_inter_data[['ts', 'i_id']] = valid_inter_data[['ts', 'i_id']].astype(np.int64)

with open(os.path.join(root_data_path, 's_session_vocab.pickle'), 'rb') as f:
    s_session_vocab = pickle.load(f)

with open(os.path.join(root_data_path, "user_vocab.pickle"), 'rb') as f:
    user_vocab = pickle.load(f)

for k in user_vocab.keys():
    user_vocab[k]['click_items'] = []
    for s_id in user_vocab[k]['src_his']:
        user_vocab[k]['click_items'] += s_session_vocab[s_id]['click_items']


if args.name=='train_reco_1':
    train_final_data_reco_1 = final_process(train_inter_data[:2500000], user_vocab, 'reco', train_inter_data['i_id'], True)
    with open(os.path.join(root_data_path, "train_data_reco_1.txt"), 'w') as f:
        for line in train_final_data_reco_1:
            f.writelines(line)
            f.write('\n')

elif args.name=='train_reco_2':
    train_final_data_reco_2 = final_process(train_inter_data[2500000:5000000], user_vocab, 'reco', train_inter_data['i_id'], True)
    with open(os.path.join(root_data_path, "train_data_reco_2.txt"), 'w') as f:
        for line in train_final_data_reco_2:
            f.writelines(line)
            f.write('\n')
elif args.name=='train_reco_3':
    train_final_data_reco_3 = final_process(train_inter_data[5000000:7500000], user_vocab, 'reco', train_inter_data['i_id'], True)
    with open(os.path.join(root_data_path, "train_data_reco_3.txt"), 'w') as f:
        for line in train_final_data_reco_3:
            f.writelines(line)
            f.write('\n')

elif args.name=='train_reco_4':
    train_final_data_reco_4 = final_process(train_inter_data[7500000:], user_vocab, 'reco', train_inter_data['i_id'], True)
    with open(os.path.join(root_data_path, "train_data_reco_4.txt"), 'w') as f:
        for line in train_final_data_reco_4:
            f.writelines(line)
            f.write('\n')


elif args.name=='train_src':
    train_final_data_src = final_process(train_inter_src_data, user_vocab, 'search', train_inter_data['i_id'], True)
    with open(os.path.join(root_data_path, "train_final_data_src.txt"), 'w') as f:
        for line in train_final_data_src:
            f.writelines(line)
            f.write('\n')

elif args.name=='test':
    test_final_data = final_process(test_inter_data, user_vocab, 'reco', train_inter_data['i_id'], False)
    with open(os.path.join(root_data_path, "test_final_data.txt"), 'w') as f:
        for line in test_final_data:
            f.writelines(line)
            f.write('\n')

elif args.name=='valid':
    valid_final_data = final_process(valid_inter_data, user_vocab, 'reco', train_inter_data['i_id'], False)
    with open(os.path.join(root_data_path, "valid_final_data.txt"), 'w') as f:
        for line in valid_final_data:
            f.writelines(line)
            f.write('\n')

