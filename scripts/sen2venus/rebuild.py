import pandas as pd
import os
import argparse
import torch

from functools import lru_cache


def parse_configs():
    parser = argparse.ArgumentParser(description='Sen2Ven rebuilder')

    parser.add_argument('--data', type=str, default='./data/sen2venus',
                        help='Directory where original data are stored.')
    parser.add_argument('--output', type=str, default='./data/sen2venus/split',
                        help='Directory where save the output files'
                             ' and load generated csv files.')

    args = parser.parse_args()
    return args


@lru_cache(maxsize=5)
def cached_torch_load(filename):
    print('load pt file {}'.format(filename))
    return torch.load(filename)


def load_rows(row):
    cols = [name for name in row.keys() if name.startswith('tensor_')]
    return {
        col: cached_torch_load(os.path.join(
            args.data, row['place'], row[col]
        ))[row['index']].clone()
        for col in cols
    }


def get_filename(index, row):
    return '{:06d}_{}_{}.pt'.format(index, row['place'], row['date'])


def get_dataset_type(filename):
    _, name = os.path.split(filename)
    name, _ = os.path.splitext(name)
    return name


def rebuild(input_filename, args):
    dtype = get_dataset_type(input_filename)
    out_dir = os.path.join(args.output, dtype)
    os.makedirs(out_dir, exist_ok=True)

    print('load {}'.format(input_filename))
    df = pd.read_csv(input_filename)
    df = df.sort_values(['place', 'date'])
    for index in range(len(df)):
        row = df.iloc[index]
        tensors = load_rows(row)
        # build filename
        filename = get_filename(index, row)
        # save tensor
        fname = os.path.join(out_dir, filename)
        print('save {}'.format(fname))
        torch.save(tensors, fname)


if __name__ == "__main__":
    # parse input arguments
    args = parse_configs()
    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))

    filename = os.path.join(args.output, 'test.csv')
    print('rebuild test dataset..')
    rebuild(filename, args)
    filename = os.path.join(args.output, 'train.csv')
    print('rebuild train dataset..')
    rebuild(filename, args)
