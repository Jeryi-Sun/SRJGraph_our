# -*- coding:utf-8 -*-
# @Organization: Alibaba
# @Author: Yukun Zheng
# @Email: zyk265182@alibaba-inc.com
# @Time: 2020/9/21 14:04

import numpy as np
from deepctr.feature_column import SparseFeat, VarLenSparseFeat


def get_all_feature_columns(emb_init,
                            user_embed_size, item_embed_size, query_embed_size, user_sex_embed_size, user_age_embed_size,
                            user_search_active_embed_size,
                            item_text_words_embed_size, item_category_embed_size,
                            item_author_id_embed_size, item_photo_len_embed_size, item_upload_type_embed_size,

                            user_vocab_size, item_vocab_size, query_vocab_size, user_sex_vocab_size, user_age_vocab_size,
                            user_search_active_vocab_size,
                            item_text_words_vocab_size, item_category_vocab_size,
                            item_author_id_vocab_size, item_photo_len_vocab_size, item_upload_type_vocab_size,
                            max_q_len,
                            need_neighbors=False, user_neighbor_nums=None, item_neighbor_nums=None):
    all_feature_columns = [SparseFeat('task', vocabulary_size=2, embedding_dim=user_embed_size,
                                      embedding_name='task', dtype='int64', embeddings_initializer=emb_init,
                                      use_hash=False),
                           SparseFeat('user_id', vocabulary_size=user_vocab_size, embedding_dim=user_embed_size,
                                      embedding_name='user_id', dtype='int64', embeddings_initializer=emb_init,
                                      use_hash=False),
                           SparseFeat('item_id', vocabulary_size=item_vocab_size, embedding_dim=item_embed_size,
                                      embedding_name='item_id', dtype='int64', embeddings_initializer=emb_init,
                                      use_hash=False),
                           VarLenSparseFeat(
                               SparseFeat('query', vocabulary_size=query_vocab_size, embedding_dim=query_embed_size,
                                          embedding_name='query_term', dtype='int64', embeddings_initializer=emb_init,
                                          use_hash=False),
                               maxlen=max_q_len),
                           SparseFeat('user_sex', vocabulary_size=user_sex_vocab_size,
                                      embedding_dim=user_sex_embed_size,
                                      embedding_name='user_sex', dtype='int64', embeddings_initializer=emb_init,
                                      use_hash=False),
                           SparseFeat('user_age', vocabulary_size=user_age_vocab_size,
                                      embedding_dim=user_age_embed_size,
                                      embedding_name='user_age', dtype='int64', embeddings_initializer=emb_init,
                                      use_hash=False),
                           SparseFeat('user_search_active', vocabulary_size=user_search_active_vocab_size,
                                      embedding_dim=user_search_active_embed_size,
                                      embedding_name='user_search_active', dtype='int64', embeddings_initializer=emb_init,
                                      use_hash=False),

                           VarLenSparseFeat(
                               SparseFeat('item_text_words', vocabulary_size=item_text_words_vocab_size, embedding_dim=item_text_words_embed_size,
                                          embedding_name='item_text_words_term', dtype='int64', embeddings_initializer=emb_init,
                                          use_hash=False),
                               maxlen=max_q_len),

                           SparseFeat('item_category', vocabulary_size=item_category_vocab_size,
                                      embedding_dim=item_category_embed_size,
                                      embedding_name='item_category', dtype='int64', embeddings_initializer=emb_init,
                                      use_hash=False),
                           SparseFeat('item_author_id', vocabulary_size=item_author_id_vocab_size,
                                      embedding_dim=item_author_id_embed_size,
                                      embedding_name='item_author_id', dtype='int64', embeddings_initializer=emb_init,
                                      use_hash=False),
                           SparseFeat('item_photo_len', vocabulary_size=item_photo_len_vocab_size,
                                      embedding_dim=item_photo_len_embed_size,
                                      embedding_name='item_photo_len', dtype='int64', embeddings_initializer=emb_init,
                                      use_hash=False),
                           SparseFeat('item_upload_type', vocabulary_size=item_upload_type_vocab_size,
                                      embedding_dim=item_upload_type_embed_size,
                                      embedding_name='item_upload_type', dtype='int64', embeddings_initializer=emb_init,
                                      use_hash=False)
                          ]
    if not need_neighbors:
        return all_feature_columns

    if len(user_neighbor_nums) >= 1 and np.prod(user_neighbor_nums[:1]) > 0:
        all_feature_columns += [
            VarLenSparseFeat(
                SparseFeat('ui', vocabulary_size=item_vocab_size,
                           embedding_dim=item_embed_size,
                           embedding_name='item_id', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:1])),
            VarLenSparseFeat(
                SparseFeat('ui_query', vocabulary_size=query_vocab_size,
                           embedding_dim=query_embed_size,
                           embedding_name='query_term', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=max_q_len*np.prod(user_neighbor_nums[:1])),

            VarLenSparseFeat(
                SparseFeat('ui_item_text_words', vocabulary_size=item_text_words_vocab_size,
                           embedding_dim=item_text_words_embed_size,
                           embedding_name='item_text_words_term', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=max_q_len*np.prod(user_neighbor_nums[:1])),

            VarLenSparseFeat(
                SparseFeat('ui_category', vocabulary_size=item_category_vocab_size,
                           embedding_dim=item_category_embed_size,
                           embedding_name='item_category', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:1])),
            VarLenSparseFeat(
                SparseFeat('ui_author_id', vocabulary_size=item_author_id_vocab_size,
                           embedding_dim=item_author_id_embed_size,
                           embedding_name='item_author_id', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:1])),
            VarLenSparseFeat(
                SparseFeat('ui_photo_len', vocabulary_size=item_photo_len_vocab_size,
                           embedding_dim=item_photo_len_embed_size,
                           embedding_name='item_photo_len', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:1])),
            VarLenSparseFeat(
                SparseFeat('ui_upload_type', vocabulary_size=item_upload_type_vocab_size,
                           embedding_dim=item_upload_type_embed_size,
                           embedding_name='item_upload_type', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:1]))
        ]
    if len(user_neighbor_nums) >= 2 and np.prod(user_neighbor_nums[:2]) > 0:
        all_feature_columns += [
            VarLenSparseFeat(
                SparseFeat('uiu', vocabulary_size=user_vocab_size,
                           embedding_dim=user_embed_size,
                           embedding_name='user_id', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:2])),
            VarLenSparseFeat(
                SparseFeat('uiu_query', vocabulary_size=query_vocab_size,
                           embedding_dim=query_embed_size,
                           embedding_name='query_term', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=max_q_len*np.prod(user_neighbor_nums[:2])),
            VarLenSparseFeat(
                SparseFeat('uiu_sex', vocabulary_size=user_sex_vocab_size,
                           embedding_dim=user_sex_embed_size,
                           embedding_name='user_sex', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:2])),
            VarLenSparseFeat(
                SparseFeat('uiu_age', vocabulary_size=user_age_vocab_size,
                           embedding_dim=user_age_embed_size,
                           embedding_name='user_age', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:2])),
            VarLenSparseFeat(
                SparseFeat('uiu_search_active', vocabulary_size=user_search_active_vocab_size,
                           embedding_dim=user_search_active_embed_size,
                           embedding_name='user_search_active', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:2]))
        ]
    if len(user_neighbor_nums) >= 3 and np.prod(user_neighbor_nums[:3]) > 0:
        all_feature_columns += [
            VarLenSparseFeat(
                SparseFeat('uiui', vocabulary_size=item_vocab_size,
                           embedding_dim=item_embed_size,
                           embedding_name='item_id', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('uiui_query', vocabulary_size=query_vocab_size,
                           embedding_dim=query_embed_size,
                           embedding_name='query_term', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=max_q_len*np.prod(user_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('uiui_item_text_words', vocabulary_size=item_text_words_vocab_size,
                           embedding_dim=item_text_words_embed_size,
                           embedding_name='item_text_words_term', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('uiui_category', vocabulary_size=item_category_vocab_size,
                           embedding_dim=item_category_embed_size,
                           embedding_name='item_category', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('uiui_author_id', vocabulary_size=item_author_id_vocab_size,
                           embedding_dim=item_author_id_embed_size,
                           embedding_name='item_author_id', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('uiui_photo_len', vocabulary_size=item_photo_len_vocab_size,
                           embedding_dim=item_photo_len_embed_size,
                           embedding_name='item_photo_len', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('uiui_upload_type', vocabulary_size=item_upload_type_vocab_size,
                           embedding_dim=item_upload_type_embed_size,
                           embedding_name='item_upload_type', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:3]))
        ]
    if len(item_neighbor_nums) >= 1 and np.prod(item_neighbor_nums[:1]) > 0:
        all_feature_columns += [
            VarLenSparseFeat(
                SparseFeat('iu', vocabulary_size=user_vocab_size,
                           embedding_dim=user_embed_size,
                           embedding_name='user_id', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:1])),
            VarLenSparseFeat(
                SparseFeat('iu_query', vocabulary_size=query_vocab_size,
                           embedding_dim=query_embed_size,
                           embedding_name='query_term', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=max_q_len*np.prod(item_neighbor_nums[:1])),
            VarLenSparseFeat(
                SparseFeat('iu_sex', vocabulary_size=user_sex_vocab_size,
                           embedding_dim=user_sex_embed_size,
                           embedding_name='user_sex', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:1])),
            VarLenSparseFeat(
                SparseFeat('iu_age', vocabulary_size=user_age_vocab_size,
                           embedding_dim=user_age_embed_size,
                           embedding_name='user_age', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:1])),
            VarLenSparseFeat(
                SparseFeat('iu_search_active', vocabulary_size=user_search_active_vocab_size,
                           embedding_dim=user_search_active_embed_size,
                           embedding_name='user_search_active', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:1]))
        ]
    if len(item_neighbor_nums) >= 2 and np.prod(item_neighbor_nums[:2]) > 0:
        all_feature_columns += [
            VarLenSparseFeat(
                SparseFeat('iui', vocabulary_size=item_vocab_size,
                           embedding_dim=item_embed_size,
                           embedding_name='item_id', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:2])),
            VarLenSparseFeat(
                SparseFeat('iui_query', vocabulary_size=query_vocab_size,
                           embedding_dim=query_embed_size,
                           embedding_name='query_term', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=max_q_len*np.prod(item_neighbor_nums[:2])),
            VarLenSparseFeat(
                SparseFeat('iui_item_text_words', vocabulary_size=item_text_words_vocab_size,
                           embedding_dim=item_text_words_embed_size,
                           embedding_name='item_text_words_term', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('iui_category', vocabulary_size=item_category_vocab_size,
                           embedding_dim=item_category_embed_size,
                           embedding_name='item_category', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen= np.prod(item_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('iui_author_id', vocabulary_size=item_author_id_vocab_size,
                           embedding_dim=item_author_id_embed_size,
                           embedding_name='item_author_id', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('iui_photo_len', vocabulary_size=item_photo_len_vocab_size,
                           embedding_dim=item_photo_len_embed_size,
                           embedding_name='item_photo_len', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('iui_upload_type', vocabulary_size=item_upload_type_vocab_size,
                           embedding_dim=item_upload_type_embed_size,
                           embedding_name='item_upload_type', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:3]))
        ]
    if len(item_neighbor_nums) >= 3 and np.prod(item_neighbor_nums[:3]) > 0:
        all_feature_columns += [
            VarLenSparseFeat(
                SparseFeat('iuiu', vocabulary_size=user_vocab_size,
                           embedding_dim=user_embed_size,
                           embedding_name='user_id', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('iuiu_query', vocabulary_size=query_vocab_size,
                           embedding_dim=query_embed_size,
                           embedding_name='query_term', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=max_q_len*np.prod(item_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('iuiu_age', vocabulary_size=user_age_vocab_size,
                           embedding_dim=user_age_embed_size,
                           embedding_name='user_age', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('iuiu_sex', vocabulary_size=user_sex_vocab_size,
                           embedding_dim=user_sex_embed_size,
                           embedding_name='user_sex', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('iuiu_search_active', vocabulary_size=user_search_active_vocab_size,
                           embedding_dim=user_search_active_embed_size,
                           embedding_name='user_search_active', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:3]))
        ]
    return all_feature_columns


if __name__ == '__main__':
    all_feature_columns = get_all_feature_columns('', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True,
                                                  [1, 1, 1, 1], [1, 1, 1, 1])
    print(len(all_feature_columns))

    all_feature_columns = get_all_feature_columns('', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, False)
    print(len(all_feature_columns))
