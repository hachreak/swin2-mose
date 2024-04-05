import os
import pandas as pd
import numpy as np
import argparse

from functools import partial
from sklearn.model_selection import train_test_split


def get_list_files(dir_name):
    for dirpath, dirs, files in os.walk(dir_name):
        for filename in files:
            if filename.endswith('index.csv'):
                fname = os.path.join(dirpath, filename)
                yield fname


def get_info_by_id(dataframes, id_):
    return dataframes.iloc[np.where(dataframes['start'].values <= id_)[0][-1]]


def read_csv_file():
    read_csv = partial(pd.read_csv, sep='\s+')

    def f(fnames):
        for fname in fnames:
            df = read_csv(fname)
            yield fname, df
    return f


def split_dataset(dir_name, test_size=.2, seed=42):
    fnames = list(get_list_files(dir_name))
    dirnames = []
    dataframes = []
    pairs_counter = 0
    for fname, df in read_csv_file()(fnames):
        #  df = read_csv(fname)
        dirname = os.path.split(os.path.split(fname)[0])[1]
        # collect csv rows
        dataframes.append(df)
        # count how many examples on this csv file
        count_examples = df['nb_patches'].sum()
        # collect directory where the patch is
        dirnames.extend([dirname] * count_examples)
        # count total (x, y) pairs
        pairs_counter += count_examples
    dataframes = pd.concat(dataframes)
    y = np.array(dirnames)
    # index list to shuffle and split in training / test dataset
    ids = np.array(list(range(pairs_counter)))
    # nb_patches cumulative column (useful to get file from the index)
    start_v = [0]
    start_v[1:] = dataframes['nb_patches'].cumsum().iloc[:-1]
    dataframes['start'] = start_v
    # split them
    X_train, X_test, y_train, y_test = train_test_split(
        ids, y, test_size=test_size, random_state=seed, stratify=y)
    return dataframes, X_train, X_test, y_train, y_test


def collect_frames_by_ids(dataframes, ids, y):
    rows = [get_info_by_id(dataframes, id_).to_dict() for id_ in ids]
    df = pd.DataFrame(rows)
    df['index'] = ids - df['start']
    df['place'] = y
    return df


def parse_configs():
    parser = argparse.ArgumentParser(description='Sen2Ven splitter')

    parser.add_argument('--seed', type=int, default=123, metavar='N',
                        help='random seed (default: 123)')
    parser.add_argument('--test_size', type=float, default=.2, metavar='N',
                        help='testi size (default 0.2)')
    parser.add_argument('--input', type=str, default='./data/sen2venus',
                        help='Directory where original data are stored.')
    parser.add_argument('--output', type=str, default='./data/sen2venus/split',
                        help='Directory where save the output files.')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # parse input arguments
    args = parse_configs()
    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))

    dataframes, X_train, X_test, y_train, y_test = split_dataset(
        dir_name=args.input, test_size=args.test_size, seed=args.seed)

    df_train = collect_frames_by_ids(dataframes, X_train, y_train)
    df_test = collect_frames_by_ids(dataframes, X_test, y_test)

    os.makedirs(args.output, exist_ok=True)
    # save train csv file
    train_file = os.path.join(args.output, 'train.csv')
    print('save train file: {}'.format(train_file))
    df_train.to_csv(train_file)
    # save test csv file
    test_file = os.path.join(args.output, 'test.csv')
    print('save train file: {}'.format(test_file))
    df_test.to_csv(test_file)
