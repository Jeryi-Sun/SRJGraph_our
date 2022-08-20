# coding=utf-8
import _pickle as cPickle
import copy
import json
import pickle
import logging
from bz2 import BZ2File
import numpy as np


def get_pvtime(user_neighbor_items, i):
    return user_neighbor_items[i]['time']

def split_list(x, max_len, seg='\x03'):
    if len(x.strip()) == 0:
        return [0] * max_len, 0
    x_split = x.split(seg)[:max_len]
    return [int(i) for i in x_split] + [0] * (max_len - len(x_split)), len(x_split)

def split_list_query(x, first_len, second_len, first_seg='\x02', second_seg='\x03'):
    out_query = []
    x_split = x.split(first_seg)[:first_len]
    x_split = x_split + [''] * (first_len - len(x_split))
    for _x in x_split:
        if len(_x.strip()) == 0:
            _x = [0] * second_len
        else:
            _x = _x.split(second_seg)[:second_len]
            _x = [int(i) for i in _x] + [0] * (second_len - len(_x))
        out_query.append(_x)
    return out_query, first_len * second_len


class Dataset():
    def __init__(self, args, train_files=[], dev_files=[], test_files=[], features={}, need_neighbor=False):
        self.logger = logging.getLogger(args.model_name)
        self.args = args
        self.batch_size, self.max_q_len, self.max_hist_len = args.batch_size, args.max_q_len, args.max_hist_len
        self.user_neighbor_nums, self.item_neighbor_nums = copy.deepcopy(args.user_neighbor_nums), copy.deepcopy(args.item_neighbor_nums)
        self.train_files = train_files
        self.dev_files = dev_files
        self.test_files = test_files
        self.features = features
        self.need_neighbor = need_neighbor
        self.user_feat_table, self.item_feat_table, self.i_id_vocab, self.u_id_vocab, self.user_feat_table_str, self.item_feat_table_str = self.load_feat_tables()
        self.logger.info('Loading graph and feature tables done.')

    def load_feat_tables(self):
        user_feat_table = cPickle.load(open(self.args.user_feat_table_path, 'rb'))
        item_feat_table = cPickle.load(open(self.args.item_feat_table_path, 'rb'))
        i_id_vocab = cPickle.load(open((self.args.i_id_vocab_path), 'rb'))
        u_id_vocab = cPickle.load(open((self.args.u_id_vocab_path), 'rb'))
        with open(self.args.user_feat_table_str_path, 'rb') as f:
            user_feat_table_str = pickle.load(f)
        with open(self.args.item_feat_table_str_path, 'rb') as f:
            item_feat_table_str = pickle.load(f)
        return user_feat_table, item_feat_table, i_id_vocab, u_id_vocab, user_feat_table_str, item_feat_table_str

    def get_mini_batch(self, mode):
        if mode == 'train':
            data_files = self.train_files
            batch_size = self.batch_size
        elif mode == 'dev':
            data_files = self.dev_files
            batch_size = self.batch_size * 8
        elif mode == 'test':
            data_files = self.test_files
            batch_size = self.batch_size * 8
        else:
            raise Exception('Invalid mode when getting data samples:', mode)

        # for f in data_files:
        #     if self.need_neighbor:
        #         assert f.endswith('.bz2')
        #     if f.endswith('.tsv'):
        #         assert not self.need_neighbor

        # columns = ['info', 'user_id', 'item_id', 'click', 'query', 'user_gender', 'user_age', 'item_brand',
        #            'item_seller', 'item_cate', 'item_cate_level1', 'item_price', 'hist', 'hist_brand', 'hist_seller',
        #            'hist_cate', 'hist_cate_level1', 'hist_price']
        # neighbor_colunms = ['ui', 'uiu', 'uiui', 'uiuiu', 'ui_query', 'uiu_query', 'uiui_query', 'uiuiu_query',
        #                     'ui_brand', 'ui_seller', 'ui_cate', 'ui_cate_level1', 'ui_price',
        #                     'uiu_gender', 'uiu_age',
        #                     'uiui_brand', 'uiui_seller', 'uiui_cate', 'uiui_cate_level1', 'uiui_price',
        #                     'uiuiu_gender', 'uiuiu_age',
        #                     'iu', 'iui', 'iuiu', 'iuiui', 'iu_query', 'iui_query', 'iuiu_query', 'iuiui_query',
        #                     'iu_gender', 'iu_age',
        #                     'iui_brand', 'iui_seller', 'iui_cate', 'iui_cate_level1', 'iui_price',
        #                     'iuiu_gender', 'iuiu_age',
        #                     'iuiui_brand', 'iuiui_seller', 'iuiui_cate', 'iuiui_cate_level1', 'iuiui_price']
        columns = ['task', 'user_id', 'user_sex', 'user_age', 'user_search_active', 'item_id',
                    'item_category', 'item_author_id', 'item_photo_len', 'item_upload_type', 'query', 'label']
        neighbor_colunms = ['ui', 'uiu', 'uiui', 'ui_query', 'uiu_query', 'uiui_query',
                             'ui_category', 'ui_author_id', 'ui_photo_len', 'ui_upload_type',
                            'uiu_sex', 'uiu_age', 'uiu_search_active',
                             'uiui_category', 'uiui_author_id', 'uiui_photo_len', 'uiui_upload_type',
                            'iu', 'iui', 'iuiu', 'iu_query', 'iui_query', 'iuiu_query',
                            'iu_sex', 'iu_age', 'iu_search_active',
                             'iui_category', 'iui_author_id', 'iui_photo_len', 'iui_upload_type',
                            'iuiu_sex', 'iuiu_age', 'iuiu_search_active']
        if not self.need_neighbor:
            neighbor_colunms = []
        raw_features_search = dict(zip(columns + neighbor_colunms, [[] for _ in range(len(columns) + len(neighbor_colunms))]))
        raw_features_recommend = copy.deepcopy(raw_features_search)

        data = []
        for f in data_files:
            for line in open(f):
                # if f.endswith('.bz2'):
                #     inputs = json.loads(line)
                # else:
                ## 先切分inter 再去找user 与 item 对应的特征 然后切分
                arr = line.strip('\n').split('\t')
                user_id, item_id, query, label = arr
                user_id, item_id, label = map(int, map(float, [user_id, item_id, label]))
                user_arr = self.user_feat_table_str[user_id].strip('\n').split('\t')
                item_arr = self.item_feat_table_str[item_id].strip('\n').split('\t')

                user_sex, user_age, user_search_active = user_arr[:3]
                _, item_category, item_author_id, item_photo_len, item_upload_type= item_arr[:5]

                user_sex, user_age, user_search_active, item_category, item_author_id, item_photo_len, item_upload_type  = \
                    map(int, map(float, [user_sex, user_age, user_search_active, item_category, item_author_id, item_photo_len, item_upload_type]))
                user_id = self.u_id_vocab[user_id]
                item_id = self.i_id_vocab[item_id]
                query_split, query_len = split_list(query, self.max_q_len)
                inputs = [user_id, user_sex, user_age, user_search_active, item_id, item_category, item_author_id, item_photo_len, item_upload_type, label,
                          query_split]
                if self.need_neighbor:
                    assert len(user_arr) == 9 and len(item_arr)==11
                    ui, uiu, uiui, ui_query, uiu_query, uiui_query = user_arr[3:]
                    iu, iui, iuiu, iu_query, iui_query, iuiu_query  = arr[5:]

                    if len(self.user_neighbor_nums) >= 1 and np.prod(self.user_neighbor_nums[:1]) > 0:
                        ui, _ = split_list(ui, np.prod(self.user_neighbor_nums[:1]))
                        ui_query, _ = split_list_query(ui_query, np.prod(self.user_neighbor_nums[:1]), self.max_q_len)
                    if len(self.user_neighbor_nums) >= 2 and np.prod(self.user_neighbor_nums[:2]) > 0:
                        uiu, _ = split_list(uiu, np.prod(self.user_neighbor_nums[:2]))
                        uiu_query, _ = split_list_query(uiu_query, np.prod(self.user_neighbor_nums[:2]), self.max_q_len)
                    if len(self.user_neighbor_nums) >= 3 and np.prod(self.user_neighbor_nums[:3]) > 0:
                        uiui, _ = split_list(uiui, np.prod(self.user_neighbor_nums[:3]))
                        uiui_query, _ = split_list_query(uiui_query, np.prod(self.user_neighbor_nums[:3]), self.max_q_len)

                    if len(self.item_neighbor_nums) >= 1 and np.prod(self.item_neighbor_nums[:1]) > 0:
                        iu, _ = split_list(iu, np.prod(self.item_neighbor_nums[:1]))
                        iu_query, _ = split_list_query(iu_query, np.prod(self.item_neighbor_nums[:1]), self.max_q_len)
                    if len(self.item_neighbor_nums) >= 2 and np.prod(self.item_neighbor_nums[:2]) > 0:
                        iui, _ = split_list(iui, np.prod(self.item_neighbor_nums[:2]))
                        iui_query, _ = split_list_query(iui_query, np.prod(self.item_neighbor_nums[:2]), self.max_q_len)
                    if len(self.item_neighbor_nums) >= 3 and np.prod(self.item_neighbor_nums[:3]) > 0:
                        iuiu, _ = split_list(iuiu, np.prod(self.item_neighbor_nums[:3]))
                        iuiu_query, _ = split_list_query(iuiu_query, np.prod(self.item_neighbor_nums[:3]), self.max_q_len)

                    inputs += [ui, uiu, uiui, ui_query, uiu_query, uiui_query, iu, iui, iuiu, iu_query, iui_query, iuiu_query]
                else:
                    inputs += [[]] * 12
                inputs += [1 if sum(query_split) else 0]
                data.append(inputs)

                if len(data) >= self.batch_size * 256:
                    for inputs in data:
                        if sum(inputs[4]) > 0: # query_split
                            raw_features_search = self.update_raw_features(raw_features_search, inputs)
                        else:
                            raw_features_recommend = self.update_raw_features(raw_features_recommend, inputs)

                        if len(raw_features_recommend['task']) >= batch_size:
                            outs = self._gen_outs(raw_features_recommend, columns)
                            if self.need_neighbor:
                                neighbor_outs = self._gen_outs(raw_features_recommend, neighbor_colunms)
                                outs.update(neighbor_outs)
                            for column in raw_features_recommend:
                                raw_features_recommend[column] = []
                            yield outs

                        if len(raw_features_search['task']) >= batch_size:
                            outs = self._gen_outs(raw_features_search, columns)
                            if self.need_neighbor:
                                neighbor_outs = self._gen_outs(raw_features_search, neighbor_colunms)
                                outs.update(neighbor_outs)
                            for column in raw_features_search:
                                raw_features_search[column] = []
                            yield outs
                    data = []

        if len(data):
            for inputs in data:
                if sum(inputs[10]) > 0:  # query_split
                    raw_features_search = self.update_raw_features(raw_features_search, inputs)
                else:
                    raw_features_recommend = self.update_raw_features(raw_features_recommend, inputs)

                if len(raw_features_recommend['task']) >= batch_size:
                    outs = self._gen_outs(raw_features_recommend, columns)
                    if self.need_neighbor:
                        neighbor_outs = self._gen_outs(raw_features_recommend, neighbor_colunms)
                        outs.update(neighbor_outs)
                    for column in raw_features_recommend:
                        raw_features_recommend[column] = []
                    yield outs

                if len(raw_features_search['task']) >= batch_size:
                    outs = self._gen_outs(raw_features_search, columns)
                    if self.need_neighbor:
                        neighbor_outs = self._gen_outs(raw_features_search, neighbor_colunms)
                        outs.update(neighbor_outs)
                    for column in raw_features_search:
                        raw_features_search[column] = []
                    yield outs
            data = []

        if len(raw_features_recommend['task']):
            outs = self._gen_outs(raw_features_recommend, columns)
            if self.need_neighbor:
                neighbor_outs = self._gen_outs(raw_features_recommend, neighbor_colunms)
                outs.update(neighbor_outs)
            for column in raw_features_recommend:
                raw_features_recommend[column] = []
            yield outs

        if len(raw_features_search['task']):
            outs = self._gen_outs(raw_features_search, columns)
            if self.need_neighbor:
                neighbor_outs = self._gen_outs(raw_features_search, neighbor_colunms)
                outs.update(neighbor_outs)
            for column in raw_features_search:
                raw_features_search[column] = []
            yield outs

    def update_raw_features(self, raw_features, inputs):
        assert len(inputs) == 24
        raw_features['task'].append(inputs[23])
        raw_features['user_id'].append(inputs[0])
        raw_features['user_sex'].append(inputs[1])
        raw_features['user_age'].append(inputs[2])
        raw_features['user_search_active'].append(inputs[3])

        raw_features['item_id'].append(inputs[4])
        raw_features['item_category'].append(inputs[5]),
        raw_features['item_author_id'].append(inputs[6])
        raw_features['item_photo_len'].append(inputs[7])
        raw_features['item_upload_type'].append(inputs[8])

        raw_features['query'].append(inputs[10])
        raw_features['label'].append(inputs[9])


        if self.need_neighbor:
            if len(self.user_neighbor_nums) >= 1 and np.prod(self.user_neighbor_nums[:1]) > 0:
                raw_features['ui'].append([self.i_id_vocab[i] if i else i for i in inputs[11]])
                raw_features['ui_query'].append(inputs[14])
                raw_features['ui_category'].append([self.item_feat_table[x]['cate'] if x else 0 for x in inputs[11]])
                raw_features['ui_author_id'].append([self.item_feat_table[x]['aut_id'] if x else 0 for x in inputs[11]])
                raw_features['ui_photo_len'].append([self.item_feat_table[x]['len_type'] if x else 0 for x in inputs[11]])
                raw_features['ui_upload_type'].append([self.item_feat_table[x]['type1'] if x else 0 for x in inputs[11]])

            if len(self.user_neighbor_nums) >= 2 and np.prod(self.user_neighbor_nums[:2]) > 0:
                raw_features['uiu'].append([self.u_id_vocab[i] if i else i for i in inputs[12]])
                raw_features['uiu_query'].append(inputs[15])
                raw_features['uiu_sex'].append([self.user_feat_table[x]['gender'] if x else 0 for x in inputs[12]])
                raw_features['uiu_age'].append([self.user_feat_table[x]['age'] if x else 0 for x in inputs[12]])
                raw_features["uiu_search_active"].append([self.user_feat_table[x]['src_level'] if x else 0 for x in inputs[12]])

            if len(self.user_neighbor_nums) >= 3 and np.prod(self.user_neighbor_nums[:3]) > 0:
                raw_features['uiui'].append([self.i_id_vocab[i] if i else i for i in inputs[13]])
                raw_features['uiui_query'].append(inputs[16])
                raw_features['uiui_category'].append([self.item_feat_table[x]['cate'] if x else 0 for x in inputs[13]])
                raw_features['uiui_author_id'].append([self.item_feat_table[x]['aut_id'] if x else 0 for x in inputs[13]])
                raw_features['uiui_photo_len'].append([self.item_feat_table[x]['len_type'] if x else 0 for x in inputs[13]])
                raw_features['uiui_upload_type'].append([self.item_feat_table[x]['type1'] if x else 0 for x in inputs[13]])


            if len(self.item_neighbor_nums) >= 1 and np.prod(self.item_neighbor_nums[:1]) > 0:
                raw_features['iu'].append([self.u_id_vocab[i] if i else i for i in inputs[17]])
                raw_features['iu_query'].append(inputs[20])
                raw_features['iu_sex'].append([self.user_feat_table[x]['gender'] if x else 0 for x in inputs[17]])
                raw_features['iu_age'].append([self.user_feat_table[x]['age'] if x else 0 for x in inputs[17]])
                raw_features["iu_search_active"].append([self.user_feat_table[x]['src_level'] if x else 0 for x in inputs[17]])

            if len(self.item_neighbor_nums) >= 2 and np.prod(self.item_neighbor_nums[:2]) > 0:
                raw_features['iui'].append([self.i_id_vocab[i] if i else i for i in inputs[18]])
                raw_features['iui_query'].append(inputs[21])
                raw_features['iui_category'].append([self.item_feat_table[x]['cate'] if x else 0 for x in inputs[18]])
                raw_features['iui_author_id'].append([self.item_feat_table[x]['aut_id'] if x else 0 for x in inputs[18]])
                raw_features['iui_photo_len'].append([self.item_feat_table[x]['len_type'] if x else 0 for x in inputs[18]])
                raw_features['iui_upload_type'].append([self.item_feat_table[x]['type1'] if x else 0 for x in inputs[18]])

            if len(self.item_neighbor_nums) >= 3 and np.prod(self.item_neighbor_nums[:3]) > 0:
                raw_features['iuiu'].append([self.u_id_vocab[i] if i else i for i in inputs[19]])
                raw_features['iuiu_query'].append(inputs[22])
                raw_features['iuiu_sex'].append([self.user_feat_table[x]['gender'] if x else 0 for x in inputs[19]])
                raw_features['iuiu_age'].append([self.user_feat_table[x]['age'] if x else 0 for x in inputs[19]])
                raw_features["iuiu_search_active"].append([self.user_feat_table[x]['src_level'] if x else 0 for x in inputs[19]])


        return raw_features

    def _gen_outs(self, raw_features, columns):
        current_batch_size = len(raw_features['task'])
        outs, new_columns = [], []
        for column in columns:
            if len(raw_features[column]) != current_batch_size:
                continue
            new_columns.append(column)
            if '_query' in column:
                outs.append(np.reshape(np.array(raw_features[column], dtype=np.int64), (current_batch_size, -1)))
            else:
                outs.append(np.array(raw_features[column], dtype=np.int64))
        return dict(zip(new_columns, outs))


if __name__ == '__main__':
    data_type = 'all'

    class config():
        def __init__(self):
            self.batch_size = 5
            self.max_q_len = 10
            self.max_hist_len = 25
            self.user_neighbor_nums = [25, 1, 10, 1]
            self.item_neighbor_nums = [1, 25, 1, 10]
            self.log_path = None
            self.user_feat_table_path = '../dataset/dataset_0918/user_feat_table_all.pkl'
            self.item_feat_table_path = '../dataset/dataset_0918/item_feat_table_all.pkl'
            self.model_name = 'dataset'

    args = config()

    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # d = Dataset(args,
    #             train_files=['../dataset/dataset_0918/all_cate_hist_graph_done/train_cate_hist_graph.tsv.bz2'],
    #             dev_files=['../dataset/dataset_0918/all_cate_hist_graph_done/dev_cate_hist_graph.tsv.bz2'],
    #             test_files=['../dataset/dataset_0918/all_cate_hist_graph_done/test_cate_hist_graph.tsv.bz2'],
    #             need_neighbor=True)
    #
    # cnt = 0
    # for x in d.get_mini_batch('dev'):
    #     for column, _x in x.items():
    #         print column + ':', _x
    #     print
    #     cnt += 1
    #     if cnt == 20:
    #         break


    # d = Dataset(args,
    #             train_files=['../dataset/dataset_0918/all_cate_hist/train_cate_hist.tsv'],
    #             dev_files=['../dataset/dataset_0918/all_cate_hist/dev_cate_hist.tsv'],
    #             test_files=['../dataset/dataset_0918/all_cate_hist/test_cate_hist.tsv'],
    #             need_neighbor=False)

    d = Dataset(args,
                # train_files=['../dataset/dataset_0918/all_cate_hist/train_cate_hist.tsv'],
                dev_files=['../dataset/dataset_0918/all_cate_hist_graph_parts/dev_cate_hist_graph_0.tsv'],
                # test_files=['../dataset/dataset_0918/all_cate_hist/test_cate_hist.tsv'],
                need_neighbor=True)

    cnt = 0
    for x in d.get_mini_batch('dev'):
        # for column, _x in x.items():
        #     print column + ':', _x
        # print
        cnt += 1
        if cnt % 5000 == 0:
            print(cnt)
