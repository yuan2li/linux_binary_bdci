# import tensorflow as tf
# print tf.__version__
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from datetime import datetime
from graphnnSiamese import graphnn
from utils import *
import os
import argparse
import json

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
parser.add_argument('--load_path', type=str, default=None,
        help='path for model loading, "#LATEST#" for the latest checkpoint')
parser.add_argument('--save_path', type=str,
        default='./saved_model/graphnn-model', help='path for model saving')
parser.add_argument('--log_path', type=str, default=None,
        help='path for training log')
parser.add_argument('datapath', help='path to storage files')
parser.add_argument('featurefile', help='features of training set')


if __name__ == '__main__':
    args = parser.parse_args()
    args.dtype = tf.float32
    print("=================================")
    print(args)
    print("=================================")

    data_path = args.datapath # E:/Dataset/linux_binary/data
    feature_file = args.featurefile # train.feature.json

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
    SAVE_PATH = args.save_path
    LOG_PATH = args.log_path

    SHOW_FREQ = 1
    TEST_FREQ = 1
    SAVE_FREQ = 5

    FUNC_NAME_DICT = {}

    # Process the input graphs
    F_NAME = data_path + os.sep + feature_file
    FUNC_NAME_DICT, CLASS_NUM = get_f_dict(data_path)
     

    Gs, classes = read_graph(F_NAME, FUNC_NAME_DICT, CLASS_NUM)
    print("{} graphs, {} functions".format(len(Gs), len(classes)))


    class_file = data_path + os.sep + 'class_perm.npy'
    if os.path.isfile(class_file):
        perm = np.load(class_file)
    else:
        perm = np.random.permutation(len(classes))
        np.save(class_file, perm)
    if len(perm) < len(classes):
        perm = np.random.permutation(len(classes))
        np.save(class_file, perm)

    Gs_train, classes_train, Gs_dev, classes_dev, Gs_test, classes_test =\
            partition_data(Gs,classes,[0.8,0.1,0.1],perm)


    print("Train: {} graphs, {} functions".format(len(Gs_train), len(classes_train)))
    print("Dev: {} graphs, {} functions".format(len(Gs_dev), len(classes_dev)))
    print("Test: {} graphs, {} functions".format(len(Gs_test), len(classes_test)))

    # Fix the pairs for validation
    valid_file = data_path + os.sep + 'valid.json'
    if os.path.isfile(valid_file):
        with open(valid_file) as inf:
            valid_ids = json.load(inf)
        valid_epoch = generate_epoch_pair(
                Gs_dev, classes_dev, BATCH_SIZE, load_id=valid_ids)
    else:
        valid_epoch, valid_ids = generate_epoch_pair(
                Gs_dev, classes_dev, BATCH_SIZE, output_id=True)
        with open(valid_file, 'w') as outf:
            json.dump(valid_ids, outf)


    # Model
#     gpu_device = [dev.strip() for dev in args.device.split(',')]
    gnn = graphnn(
            N_x = NODE_FEATURE_DIM,
            Dtype = Dtype, 
            N_embed = EMBED_DIM,
            depth_embed = EMBED_DEPTH,
            N_o = OUTPUT_DIM,
            ITER_LEVEL = ITERATION_LEVEL,
            lr = LEARNING_RATE,
        #     device = ["/gpu:" + i for i in gpu_device]
        )
    gnn.init(LOAD_PATH, LOG_PATH)

    # Train
    auc, fpr, tpr, thres = get_auc_epoch(gnn, Gs_train, classes_train,
            BATCH_SIZE, load_data=valid_epoch)
    gnn.say("Initial training auc = {0} @ {1}".format(auc, datetime.now()))
    auc0, fpr, tpr, thres = get_auc_epoch(gnn, Gs_dev, classes_dev,
            BATCH_SIZE, load_data=valid_epoch)
    gnn.say("Initial validation auc = {0} @ {1}".format(auc0, datetime.now()))

    best_auc = 0
    for i in range(1, MAX_EPOCH+1):
        l = train_epoch(gnn, Gs_train, classes_train, BATCH_SIZE)
        gnn.say("EPOCH {3}/{0}, loss = {1} @ {2}".format(
            MAX_EPOCH, l, datetime.now(), i))

        if (i % TEST_FREQ == 0):
            auc, fpr, tpr, thres = get_auc_epoch(gnn, Gs_train, classes_train,
                    BATCH_SIZE, load_data=valid_epoch)
            gnn.say("Testing model: training auc = {0} @ {1}".format(
                auc, datetime.now()))
            auc, fpr, tpr, thres = get_auc_epoch(gnn, Gs_dev, classes_dev,
                    BATCH_SIZE, load_data=valid_epoch)
            gnn.say("Testing model: validation auc = {0} @ {1}".format(
                auc, datetime.now()))

            if auc > best_auc:
                path = gnn.save(SAVE_PATH+'_best')
                best_auc = auc
                gnn.say("Model saved in {}".format(path))

        if (i % SAVE_FREQ == 0):
            path = gnn.save(SAVE_PATH, i)
            gnn.say("Model saved in {}".format(path))
