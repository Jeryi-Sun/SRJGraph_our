# -*- coding:utf-8 -*-
"""
Author:
    Yukun Zheng, zyk265182@alibaba-inc.com
"""
import logging
import os
from tqdm import tqdm
import numpy as np
import time
import tensorflow as tf
from sklearn.metrics import roc_auc_score, log_loss
from tensorboardX import SummaryWriter
from tensorflow.python.keras import backend as K, Model
from tensorflow.python.keras.layers import Concatenate, Lambda, Dense, Multiply, Add, BatchNormalization
from tensorflow.python.keras.optimizers import Adam, Adadelta, Adagrad, SGD, RMSprop

from deepctr.feature_column import build_input_features
from deepctr.inputs import create_embedding_matrix, embedding_lookup
from deepctr.layers import DNN, PredictionLayer
from inputs.features_columns import get_all_feature_columns
from layers.layers import Transformer
from utils.metrics import judger as judge

class BaseModel(object):
    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger(args.model_name)
        self.dataset = None
        self.model_name = args.model_name
        self.eval_step = args.eval_step
        self.save_step = args.save_step
        self.train_epoch = args.train_epoch
        self.batch_size = args.batch_size
        self.max_q_len = args.max_q_len
        self.max_hist_len = args.max_hist_len
        self.dropout_rate = args.dropout_rate
        self.need_neighbors = None
        self.cotrain = args.cotrain

        self.user_embed_size = args.user_embed_size
        self.item_embed_size = args.item_embed_size
        self.query_embed_size = args.query_embed_size

        self.user_sex_embed_size = args.user_sex_embed_size
        self.user_age_embed_size = args.user_age_embed_size
        self.user_search_active_embed_size = args.user_search_active_embed_size
        self.item_text_words_embed_size = args.item_text_words_embed_size
        self.item_category_embed_size = args.item_category_embed_size
        self.item_author_id_embed_size = args.item_author_id_embed_size
        self.item_photo_len_embed_size = args.item_photo_len_embed_size
        self.item_upload_type_embed_size = args.item_upload_type_embed_size


        self.user_vocab_size = args.user_vocab_size
        self.item_vocab_size = args.item_vocab_size
        self.query_vocab_size = args.query_vocab_size
        self.user_sex_vocab_size = args.user_sex_vocab_size
        self.user_age_vocab_size = args.user_age_vocab_size
        self.user_search_active_vocab_size = args.user_search_active_vocab_size
        self.item_text_words_vocab_size= args.item_text_words_vocab_size
        self.item_category_vocab_size = args.item_category_vocab_size
        self.item_author_id_vocab_size = args.item_author_id_vocab_size
        self.item_photo_len_vocab_size = args.item_photo_len_vocab_size
        self.item_upload_type_vocab_size = args.item_upload_type_vocab_size
        self.need_neighbors = True
        self.user_neighbor_nums = args.user_neighbor_nums
        self.item_neighbor_nums = args.item_neighbor_nums

        self.padding_id = 0
        self.global_step = args.global_step
        self.best_eval_auc_search = -1.
        self.best_eval_auc_recommend = -1.
        self.patience = args.patience
        self.current_patience = args.patience
        self.dnn_hidden_units = args.hidden_size
        self.learning_rate = args.learning_rate
        self.judger = judge()
        self.column_groups = None
        self.feature_columns_dict = None
        self.features = None
        self.feature_embed_dict = {}
        self.group_embed_dict = {}
        self.deep_input_emb = None
        self.model = None

        self.summary_dir = os.path.join(self.args.summary_dir, self.model_name)
        if not os.path.exists(self.summary_dir):
            os.mkdir(self.summary_dir)
        self.writer = SummaryWriter(logdir=self.summary_dir)

    def initialize_model(self):
        if self.args.load_model_path:
            self.logger.info('Restoring model from {} ...'.format(self.args.load_model_path))
            self.model = self.load_model(self.args.load_model_path)
            self.predict()
        else:
            self.logger.info('Creating model...')
            self.embedding_layer()
            self.create_model()
            self.model = self.prediction_layer()
            self.compile_model(self.args.opt, self.learning_rate)

    def embedding_layer(self):
        feature_columns = []
        for group in self.column_groups:
            feature_columns += self.get_feature_columns(self.column_groups[group])
        self.features = build_input_features(feature_columns)
        self.feature_columns_dict = {fc.name: fc for fc in feature_columns}
        embedding_dict = create_embedding_matrix(feature_columns, l2_reg=0., prefix="", seq_mask_zero=False)
        # query_transformer = Transformer(att_embedding_size=self.query_embed_size, head_num=1, dropout_rate=self.dropout_rate)
        for group in self.column_groups:
            group_feature_columns = []
            for x in self.column_groups[group]:
                if x in self.feature_columns_dict:
                    group_feature_columns.append(self.feature_columns_dict[x])
            if len(group_feature_columns) == 0:
                continue
            if 'query' in group:
                group_embed = embedding_lookup(embedding_dict, self.features, group_feature_columns, to_list=True)
                # group_embed_reshape = Lambda(K.reshape, arguments={'shape': (tf.shape(self.features['user_id'])[0], -1, self.max_q_len, self.query_embed_size)})(group_embed[0])
                # query_mask = Lambda(tf.reshape, arguments={'shape': (tf.shape(self.features['user_id'])[0], -1, self.max_q_len)})(self.features[self.column_groups[group][0]])
                # query_mask = Lambda(tf.not_equal, arguments={'y': 0})(query_mask)
                # group_embed_reshape = query_transformer(inputs=[group_embed_reshape, group_embed_reshape], mask=[query_mask, query_mask])
                # self.group_embed_dict[group] = Lambda(K.reshape, arguments={'shape': (-1, 1, self.query_embed_size)})(group_embed_reshape)
                # self.group_embed_dict[group] = Lambda(K.sum, arguments={'axis': 1, 'keepdims': True})(group_embed_reshape)
                group_embed_reshape = Lambda(K.reshape, arguments={'shape': (-1, self.max_q_len, self.query_embed_size)})(group_embed[0])
                self.group_embed_dict[group] = Lambda(K.sum, arguments={'axis': 1, 'keepdims': True})(group_embed_reshape)
            else:
                group_embed = embedding_lookup(embedding_dict, self.features, group_feature_columns, to_list=True)
                if len(group_embed) > 1:
                    self.group_embed_dict[group] = Concatenate(axis=-1)(group_embed)
                else:
                    self.group_embed_dict[group] = group_embed[0]

            assert len(group_feature_columns) == len(group_embed)
            for fc, fc_embed in zip(group_feature_columns, group_embed):
                self.feature_embed_dict[fc.name] = fc_embed

    def prediction_layer(self):
        dnn_activation = 'dice'
        l2_reg_dnn = 0.
        dnn_use_bn = False

        '''search'''
        deep_input_emb_search = BatchNormalization()(self.deep_input_emb, training=None)
        output_search = DNN(self.dnn_hidden_units, dnn_activation, l2_reg_dnn,
                     self.dropout_rate, dnn_use_bn)(deep_input_emb_search)
        final_logit_search = Dense(1, use_bias=False)(output_search)
        output_search = PredictionLayer()(final_logit_search)

        '''recommend'''
        deep_input_emb_recommend = BatchNormalization()(self.deep_input_emb, training=None)
        output_recommend = DNN(self.dnn_hidden_units, dnn_activation, l2_reg_dnn,
                     self.dropout_rate, dnn_use_bn)(deep_input_emb_recommend)
        final_logit_recommend = Dense(1, use_bias=False)(output_recommend)
        output_recommend = PredictionLayer()(final_logit_recommend)

        query_sum = Lambda(tf.reduce_sum, arguments={'axis': 1, 'keepdims': True})(self.features['query'])
        query_mask_search = Lambda(tf.not_equal, arguments={'y': 0})(query_sum)
        query_mask_recommend = Lambda(tf.equal, arguments={'y': 0})(query_sum)
        query_mask_search = Lambda(tf.cast, arguments={'dtype': 'float32'})(query_mask_search)
        query_mask_recommend = Lambda(tf.cast, arguments={'dtype': 'float32'})(query_mask_recommend)

        output_search = Multiply()([output_search, query_mask_search])
        output_recommend = Multiply()([output_recommend, query_mask_recommend])
        output_all = Add()([output_search, output_recommend])

        model = Model(inputs=list(self.features.values()), outputs=output_all)
        return model

    def create_model(self):
        pass

    def load_model(self, model_path):
        pass

    def save_model(self, prefix=''):
        out_dir = os.path.join(self.args.ckpt_dir, self.model_name)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if prefix == '':
            save_model_name = '{}_{}.h5'.format(self.model_name, self.global_step)
        else:
            save_model_name = '{}_{}_{}.h5'.format(self.model_name, self.global_step, prefix)
        self.model.save(os.path.join(out_dir, save_model_name))

    def compile_model(self, opt, lr):
        if opt == 'adam':
            opt = Adam(lr=lr, epsilon=1e-8)
        elif opt == 'adadelta':
            opt = Adadelta(lr=lr, epsilon=1e-8)
        elif opt == 'tf_adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate=lr)
        elif opt == 'adagrad':
            opt = Adagrad(lr=lr, epsilon=1e-8)
        elif opt == 'rmsprop':
            opt = RMSprop(lr=lr)
        else:
            opt = SGD(lr=lr, momentum=0.1)
        self.model.compile(optimizer=opt, loss='binary_crossentropy')

    def get_feature_columns(self, columns):
        all_feature_columns = get_all_feature_columns(self.args.embed_init,
                                                      self.user_embed_size, self.item_embed_size, self.query_embed_size,
                                                      self.user_sex_embed_size, self.user_age_embed_size,
                                                      self.user_search_active_embed_size,
                                                      self.item_text_words_embed_size, self.item_category_embed_size,
                                                      self.item_author_id_embed_size, self.item_photo_len_embed_size,
                                                      self.item_upload_type_embed_size,

                                                      self.user_vocab_size, self.item_vocab_size, self.query_vocab_size,
                                                      self.user_sex_vocab_size, self.user_age_vocab_size,
                                                      self.user_search_active_vocab_size,
                                                      self.item_text_words_vocab_size, self.item_category_vocab_size,
                                                      self.item_author_id_vocab_size, self.item_photo_len_vocab_size,
                                                      self.item_upload_type_vocab_size,
                                                      self.max_q_len,
                                                      need_neighbors = self.need_neighbors,
                                                      user_neighbor_nums=self.user_neighbor_nums,
                                                      item_neighbor_nums=self.item_neighbor_nums)
        requested_feature_columns = []
        for feat in all_feature_columns:
            if feat.name in columns:
                requested_feature_columns.append(feat)
        return requested_feature_columns

    def train_one_epoch(self):
        #time_start=time.time()
        for mini_batch in tqdm(self.dataset.get_mini_batch('train')):
            #time_end_1=time.time()
            #print('read time cost',time_end_1-time_start,'s') 
            x = {}
            #time_end_2 = time.time()
            for feat in self.feature_columns_dict:
                x[feat] = mini_batch[feat]
            y = mini_batch['label']
            #print('process time cost',time_end_2-time_end_1,'s') 
            train_loss = self.model.train_on_batch(x, y)
            self.global_step += 1
            #time_end_3 = time.time()
            #print('model time cost',time_end_3-time_end_2,'s')  
            data_type = 'search' if sum(x['query'][0].tolist()) > 0 else 'recommend'
            self.writer.add_scalar('train/loss_{}'.format(data_type), train_loss, self.global_step)
            if self.global_step % self.eval_step == 0:
                self.evaluate(predict_tag=True)
            if self.global_step % self.save_step == 0:
                self.save_model()
            if self.current_patience == 0:
                break             
            #time_start=time.time()
            #print('else time cost',time_start-time_end_3,'s') 

    def train(self):
        self.logger.info('Training...')
        for i in range(self.train_epoch):
            self.train_one_epoch()
            if self.current_patience == 0:
                break
            self.predict()
            self.evaluate()
            # self.save_model(prefix='epoch_{}'.format(i))
            if self.current_patience == 0:
                break

    def evaluate(self, predict_tag=False):
        self.logger.info('Evaluating...')
        #rn_search, rn_recommend = [], []
        rn = []
        group_size = 100
        eval_y_true, eval_y_pred = [], []
        group_preds, group_labels = [], [] 
        # eval_y_true_search, eval_y_pred_search = [], []
        # eval_y_true_recommend, eval_y_pred_recommend = [], []
        for mini_batch in tqdm(self.dataset.get_mini_batch('dev')):
            x = {}
            for feat in self.feature_columns_dict:
                x[feat] = mini_batch[feat]
            y = mini_batch['label'].tolist()
            preds = self.model.predict_on_batch(x).reshape([-1]).tolist()
            eval_y_true += y
            eval_y_pred += preds
            group_preds.extend(np.reshape(preds, (-1, group_size)))
            group_labels.extend(np.reshape(y, (-1, group_size)))
            for user, item in zip(mini_batch['user_id'].tolist(), mini_batch['item_id'].tolist()):
                rn.append([user, item])
            
            # if sum(x['query'][0].tolist()) > 0:
            #     eval_y_true_search += y
            #     eval_y_pred_search += preds
            #     for user, item in zip(mini_batch['user_id'].tolist(), mini_batch['item_id'].tolist()):
            #         rn_search.append([user, item])
            # else:
            #     eval_y_true_recommend += y
            #     eval_y_pred_recommend += preds
            #     for user, item in zip(mini_batch['user_id'].tolist(), mini_batch['item_id'].tolist()):
            #         rn_recommend.append([user, item])

        best_tag = False

        if len(eval_y_true) > 0:
            eval_loss = log_loss(eval_y_true, eval_y_pred, eps=1e-8, labels=[0, 1])
            eval_auc_recommend = roc_auc_score(eval_y_true, eval_y_pred)
            res_pairwise = self.judger.cal_metric(
            group_preds, group_labels, ['ndcg@5;10', 'hit@1;5;10','mrr'])
            self.logger.info('global_step: {}, dev/loss: {}, dev/auc: {}, dev/ndcg@5: {}, dev/ndcg@10: {}, dev/hit@1: {},  \
                dev/hit@5: {}, dev/hit@10: {}, dev/mrr: {}'.format(self.global_step, eval_loss, eval_auc_recommend, res_pairwise["ndcg@5"], res_pairwise["ndcg@10"], 
                res_pairwise["hit@1"], res_pairwise["hit@5"], res_pairwise["hit@10"], res_pairwise["mrr"]))
            print('global_step: {}, dev/loss: {}, dev/auc: {}, dev/ndcg@5: {}, dev/ndcg@10: {}, dev/hit@1: {},  \
                dev/hit@5: {}, dev/hit@10: {}, dev/mrr: {}'.format(self.global_step, eval_loss, eval_auc_recommend, res_pairwise["ndcg@5"], res_pairwise["ndcg@10"], 
                res_pairwise["hit@1"], res_pairwise["hit@5"], res_pairwise["hit@10"], res_pairwise["mrr"]))
            self.writer.add_scalar('dev/loss', eval_loss, self.global_step)
            self.writer.add_scalar('dev/auc', eval_auc_recommend, self.global_step)
            self.save_results(rn, eval_y_true, eval_y_pred, 'predict_{}.txt'.format(self.global_step))
            if eval_auc_recommend > self.best_eval_auc_recommend:
                self.best_eval_auc_recommend = eval_auc_recommend
                self.current_patience = self.patience
                best_tag = True
            else:
                self.current_patience -= 1

        # if len(eval_y_true_recommend) > 0:
        #     eval_loss_recommend = log_loss(eval_y_true_recommend, eval_y_pred_recommend, eps=1e-8, labels=[0, 1])
        #     eval_auc_recommend = roc_auc_score(eval_y_true_recommend, eval_y_pred_recommend)
        #     self.logger.info(
        #         'global_step: {}, dev/loss_recommend: {}, dev/auc_recommend: {}'.format(self.global_step, eval_loss_recommend, eval_auc_recommend))
        #     self.writer.add_scalar('dev/loss_recommend', eval_loss_recommend, self.global_step)
        #     self.writer.add_scalar('dev/auc_recommend', eval_auc_recommend, self.global_step)
        #     self.save_results(rn_recommend, eval_y_true_recommend, eval_y_pred_recommend, 'dev_recommend_{}.txt'.format(self.global_step))
        #     if eval_auc_recommend > self.best_eval_auc_recommend:
        #         self.best_eval_auc_recommend = eval_auc_recommend
        #         self.current_patience = self.patience
        #         best_tag = True
        #     else:
        #         self.current_patience -= 1

        # if len(eval_y_true_search) > 0:
        #     eval_loss_search = log_loss(eval_y_true_search, eval_y_pred_search, eps=1e-8, labels=[0, 1])
        #     eval_auc_search = roc_auc_score(eval_y_true_search, eval_y_pred_search)
        #     self.logger.info('global_step: {}, dev/loss_search: {}, dev/auc_search: {}'.format(self.global_step, eval_loss_search, eval_auc_search))
        #     self.writer.add_scalar('dev/loss_search', eval_loss_search, self.global_step)
        #     self.writer.add_scalar('dev/auc_search', eval_auc_search, self.global_step)
        #     self.save_results(rn_search, eval_y_true_search, eval_y_pred_search, 'dev_search_{}.txt'.format(self.global_step))

        #     if eval_auc_search > self.best_eval_auc_search:
        #         self.best_eval_auc_search = eval_auc_search
        #         self.current_patience = self.patience
        #         best_tag = True
        #     else:
        #         self.current_patience -= 1

        if predict_tag and best_tag and self.args.test_files:
            self.predict()

    def predict(self):
        self.logger.info('Predicting...')
        rn = []
        group_size = 100
        # rn_search, rn_recommend = [], []
        # eval_y_true_search, eval_y_pred_search = [], []
        # eval_y_true_recommend, eval_y_pred_recommend = [], []
        eval_y_true, eval_y_pred = [], [] 
        group_preds, group_labels = [], [] 
        for mini_batch in tqdm(self.dataset.get_mini_batch('test')):
            x = {}
            for feat in self.feature_columns_dict:
                x[feat] = mini_batch[feat]
            y = mini_batch['label'].tolist()
            preds = self.model.predict_on_batch(x).reshape([-1]).tolist()

            eval_y_true += y
            eval_y_pred += preds
            group_preds.extend(np.reshape(preds, (-1, group_size)))
            group_labels.extend(np.reshape(y, (-1, group_size)))
            for user, item in zip(mini_batch['user_id'].tolist(), mini_batch['item_id'].tolist()):
                rn.append([user, item])

        if len(eval_y_true) > 0:
            eval_loss = log_loss(eval_y_true, eval_y_pred, eps=1e-8, labels=[0, 1])
            eval_auc = roc_auc_score(eval_y_true, eval_y_pred)
            res_pairwise = self.judger.cal_metric(
            group_preds, group_labels, ['ndcg@5;10', 'hit@1;5;10','mrr'])
            self.logger.info('global_step: {}, test/loss: {}, test/auc: {}, test/ndcg@5: {}, test/ndcg@10: {}, test/hit@1: {},  \
                test/hit@5: {}, test/hit@10: {}, test/mrr: {}'.format(self.global_step, eval_loss, eval_auc, res_pairwise["ndcg@5"], res_pairwise["ndcg@10"], 
                res_pairwise["hit@1"], res_pairwise["hit@5"], res_pairwise["hit@10"], res_pairwise["mrr"]))
            print('global_step: {}, test/loss: {}, test/auc: {}, test/ndcg@5: {}, test/ndcg@10: {}, test/hit@1: {},  \
                test/hit@5: {}, test/hit@10: {}, test/mrr: {}'.format(self.global_step, eval_loss, eval_auc, res_pairwise["ndcg@5"], res_pairwise["ndcg@10"], 
                res_pairwise["hit@1"], res_pairwise["hit@5"], res_pairwise["hit@10"], res_pairwise["mrr"]))
            self.writer.add_scalar('test/loss', eval_loss, self.global_step)
            self.writer.add_scalar('test/auc', eval_auc, self.global_step)
            self.save_results(rn, eval_y_true, eval_y_pred, 'predict_{}.txt'.format(self.global_step))

            # if sum(x['query'][0].tolist()) > 0:
            #     eval_y_true_search += y
            #     eval_y_pred_search += preds
            #     for user, item in zip(mini_batch['user_id'].tolist(), mini_batch['item_id'].tolist()):
            #         rn_search.append([user, item])
            # else:
            #     eval_y_true_recommend += y
            #     eval_y_pred_recommend += preds
            #     for user, item in zip( mini_batch['user_id'].tolist(), mini_batch['item_id'].tolist()):
            #         rn_recommend.append([ user, item])

        # if len(eval_y_true_recommend) > 0:
        #     eval_loss_recommend = log_loss(eval_y_true_recommend, eval_y_pred_recommend, eps=1e-8, labels=[0, 1])
        #     eval_auc_recommend = roc_auc_score(eval_y_true_recommend, eval_y_pred_recommend)
        #     self.logger.info('global_step: {}, test/loss_recommend: {}, test/auc_recommend: {}'.format(self.global_step, eval_loss_recommend, eval_auc_recommend))
        #     self.writer.add_scalar('test/loss_recommend', eval_loss_recommend, self.global_step)
        #     self.writer.add_scalar('test/auc_recommend', eval_auc_recommend, self.global_step)
        #     self.save_results(rn_recommend, eval_y_true_recommend, eval_y_pred_recommend, 'predict_recommend_{}.txt'.format(self.global_step))

        # if len(eval_y_true_search) > 0:
        #     eval_loss_search = log_loss(eval_y_true_search, eval_y_pred_search, eps=1e-8, labels=[0, 1])
        #     eval_auc_search = roc_auc_score(eval_y_true_search, eval_y_pred_search)
        #     self.logger.info('global_step: {}, test/loss_search: {}, test/auc_search: {}'.format(self.global_step, eval_loss_search, eval_auc_search))
        #     self.writer.add_scalar('test/loss_search', eval_loss_search, self.global_step)
        #     self.writer.add_scalar('test/auc_search', eval_auc_search, self.global_step)
        #     self.save_results(rn_search, eval_y_true_search, eval_y_pred_search, 'predict_search_{}.txt'.format(self.global_step))

    def save_results(self, rn, y_true, y_pred, result_file):
        out_dir = os.path.join(self.args.result_dir, self.model_name)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out = open(os.path.join(out_dir, result_file), 'w')
        for _rn, y_t, y_p in zip(rn, y_true, y_pred):
            user, item = _rn
            out.write('{}\t{}\t{}\t{}\n'.format(user, item, y_t, y_p))
        out.close()
