import pickle
import numpy as np
from scipy.sparse import csr_matrix

import sys, os


def get_label2id(label_map_file):
    label_map = open(label_map_file).readlines()
    label_map = [l.split() for l in label_map]

    num_labels = len(label_map)
    label2id = {label_map[i][0]: i for i in range(num_labels)}

    parents = []
    for lbl in label_map:
        parents += lbl[1:]
    parents = sorted(list(set(parents)))

    for parent in parents:
        if parent not in label2id:
            label2id[parent] = num_labels
            num_labels += 1

    return label2id


def multiclass2multilabel(source_file, save_file, label2id):
    images, labels, ids = pickle.load(open(source_file, 'rb'))
    num_images = 0
    for image_mat in images:
        num_images += image_mat.shape[0]
    assert num_images == len(labels)
    assert num_images == len(ids)

    label_map = open(label_map_file).readlines()
    label_map = [l.split() for l in label_map]

    num_labels = len(label2id)
    assert num_labels == len(set(label2id.values()))

    print('# of images: '+str(num_images))
    print('# of labels: '+str(num_labels))

    row_ind = []
    col_ind = []
    data = []
    for i, label in enumerate(labels):
        assert label_map[label][0] == ids[i].split('_')[0]
        for l in label_map[label]:
            row_ind.append(i)
            col_ind.append(label2id[l])
            data.append(np.float32(1.0))
    label_mat = csr_matrix((data,(row_ind, col_ind)), shape=(num_images, num_labels), dtype=np.float32)

    pickle.dump((images, label_mat, (ids, label2id)), open(save_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    source_train_file = sys.argv[1]
    source_val_file = sys.argv[2]
    source_test_file = sys.argv[3]
    save_train_file = sys.argv[4]
    save_val_file = sys.argv[5]
    save_test_file = sys.argv[6]
    label_map_file = sys.argv[7]

    label2id = get_label2id(label_map_file)
    multiclass2multilabel(source_train_file, save_train_file, label2id)
    multiclass2multilabel(source_val_file, save_val_file, label2id)
    multiclass2multilabel(source_test_file, save_test_file, label2id)

