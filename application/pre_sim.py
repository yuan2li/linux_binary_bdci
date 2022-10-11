import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('datapath', help='path to storage files')
parser.add_argument('featurefile', help='features of test set')

def get_adjacent_matrix(succs):
    # initialize the adjacent matrix of CFG
    adjacent_matrix = [[0 for _ in range(len(succs))] for _ in range(len(succs))]
    for i, nodes in enumerate(succs):
        for node in nodes:
            adjacent_matrix[i][node] = 1
    return adjacent_matrix

if __name__ == '__main__':
    args = parser.parse_args()
    print("=================================")
    print(args)
    print("=================================")

    data_path = args.datapath # E:/Dataset/linux_binary/data
    feature_file = args.featurefile # test.feature.json

    F_NAME = data_path + os.sep + feature_file
    F_NAME_DICT = data_path + os.sep + 'test.dict.json'

    fid_dic = {}
    with open(F_NAME) as f:
        for line in f:
            func = json.loads(line.strip())
            func['adj_mat'] = get_adjacent_matrix(func['succs'])
            fid_dic[str(func['fname'])] = {'features': func['features'], 'adj_mat': func['adj_mat']}
    with open(F_NAME_DICT, 'w') as f:
        json.dump(fid_dic, f, ensure_ascii=False)
