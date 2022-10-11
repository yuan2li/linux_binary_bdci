# import tensorflow as tf
# print tf.__version__
from re import A
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import sys
sys.path.append("..")
import numpy as np
import time
from graphnnSiamese import graphnn
from utils import *
import os
import argparse
import json
import pandas as pd
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0',
                    help='visible gpu device')
parser.add_argument('--fea_dim', type=int, default=7,
                    help='feature dimension')
parser.add_argument('--embed_dim', type=int, default=64,
                    help='embedding dimension')
parser.add_argument('--embed_depth', type=int, default=2,
                    help='embedding network depth')
parser.add_argument('--output_dim', type=int, default=64,
                    help='output layer dimension')
parser.add_argument('--iter_level', type=int, default=5,
                    help='iteration times')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--epoch', type=int, default=100,
                    help='epoch number')
parser.add_argument('--batch_size', type=int, default=5,
                    help='batch size')
parser.add_argument('--load_path', type=str,
                    default='../saved_model/graphnn-model_best',
                    help='path for model loading, "#LATEST#" for the latest checkpoint')
parser.add_argument('--log_path', type=str, default=None,
                    help='path for training log')

parser.add_argument('--batch_num', type=str, default='0',
                    help='No of the batch data')
parser.add_argument('datapath', help='path to storage files')
parser.add_argument('featurefile', help='features of test set')


def get_adjacent_matrix(succs):
    # initialize the adjacent matrix of CFG
    adjacent_matrix = [[0 for _ in range(len(succs))] for _ in range(len(succs))]
    for i, nodes in enumerate(succs):
        for node in nodes:
            adjacent_matrix[i][node] = 1
    return adjacent_matrix


def get_sim_attr(fid, fid_dic):
    dic = fid_dic[fid]
    fl = np.asarray(dic['features'])
    fl = np.expand_dims(fl, axis=0)
    am = np.asarray(dic['adj_mat'])
    am = np.expand_dims(am, axis=0)
    return fl, am


if __name__ == '__main__':
    args = parser.parse_args()
    args.dtype = tf.float32
    print("=================================")
    print(args)
    print("=================================")

    data_path = args.datapath # E:/Dataset/linux_binary/data
    feature_file = args.featurefile # test.feature.json test.dict.json
    batch_num = args.batch_num

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    Dtype = args.dtype
    NODE_FEATURE_DIM = args.fea_dim
    EMBED_DIM = args.embed_dim
    EMBED_DEPTH = args.embed_depth
    OUTPUT_DIM = args.output_dim
    ITERATION_LEVEL = args.iter_level
    LEARNING_RATE = args.lr
    MAX_EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    LOAD_PATH = args.load_path
    LOG_PATH = args.log_path

    SHOW_FREQ = 1
    TEST_FREQ = 1
    SAVE_FREQ = 5

    F_NAME = data_path + os.sep + feature_file

    # Model
    gnn = graphnn(
        N_x=NODE_FEATURE_DIM,
        Dtype=Dtype,
        N_embed=EMBED_DIM,
        depth_embed=EMBED_DEPTH,
        N_o=OUTPUT_DIM,
        ITER_LEVEL=ITERATION_LEVEL,
        lr=LEARNING_RATE,
        device = '/gpu:0'
    )
    gnn.init(LOAD_PATH, LOG_PATH)

    # fid_dic = {}
    # with open(F_NAME) as f:
    #     for line in f:
    #         func = json.loads(line.strip())
    #         func['adj_mat'] = get_adjacent_matrix(func['succs'])
    #         fid_dic[str(func['fname'])] = func

    with open(F_NAME) as f:
        fid_dic = json.load(f)

    test_question_df = pd.read_csv(data_path + os.sep + "test.question." + batch_num + ".csv", header=None)
    test_question = np.array(test_question_df)
    target_fids = test_question[:, 0]

    sub = []
    with tqdm(total=len(target_fids), ncols=100, desc="similarity matching") as pbar:
        for i, t_fid in enumerate(target_fids):
            target_func_fealis, target_func_am = get_sim_attr(str(t_fid), fid_dic)
            res = []
            wait_fids = test_question[i, 1:]
            for w_fid in wait_fids:
                func_fealis, func_am = get_sim_attr(str(w_fid), fid_dic)
                sim = gnn.calc_diff(id1=str(t_fid), id2=str(w_fid), 
                                X1=target_func_fealis, X2=func_fealis,
                                mask1=target_func_am, mask2=func_am)
                sim_v = np.round(sim.tolist()[0], 2)
                res.append((w_fid, sim_v))
            res.sort(key=lambda x:x[1], reverse=True)
            res = [str(t_fid)] + [str(r[0]) + ':' +str(r[1]) for r in res[:10]]
            sub.append(res)
            pbar.update(1)

    sub_cols = ['fid'] + ['fid' + str(i) + ':' + 'sim' + str(i) for i in range(10)]
    sub_df = pd.DataFrame(sub, columns=sub_cols)
    sub_file = data_path + os.sep + 'submission.' + batch_num + '.' + time.strftime("%Y%m%d%H%M%S", time.localtime()) + '.csv'
    sub_df.to_csv(sub_file, index=None)
