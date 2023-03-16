# This code file is used to construct an expanded version of the STS and SST training datasets
import pandas as pd
import numpy as np
import argparse

def construct_datasets(args):
    # sst dataset: key columns are 'sentence', 'id', and 'sentiment'
    # read in cfimdb train and dev datasets and alter the sentiment column
    if args.sst:
        df = pd.read_csv('data/ids-cfimdb-train.csv', sep='\t', index_col=0)
        df = pd.concat([df, pd.read_csv('data/ids-cfimdb-dev.csv', sep='\t', index_col=0)])
        sent_zero = df['sentiment'] == 0
        sent_one = df['sentiment'] == 1

        df2 = df.copy()
        # construct polar dataset and save to drive
        df.loc[sent_one, 'sentiment'] = 4
        df = pd.concat([df, pd.read_csv('data/ids-sst-train.csv', sep = '\t', index_col=0)])
        df.to_csv('data/ids-sst-train-expanded-polar.csv', sep='\t')
        # construct non-polar dataset and save to drive
        df2.loc[sent_zero, 'sentiment'] = 1
        df2.loc[sent_one, 'sentiment'] = 3
        df2 = pd.concat([df2, pd.read_csv('data/ids-sst-train.csv', sep = '\t', index_col=0)])
        df2.to_csv('data/ids-sst-train-expanded-not-polar.csv', sep='\t')
        print('sst generated and saved')

    # sts dataset: key columns are 'id', 'sentence1', 'sentence2', 'similarity'
        # read in sick2014 dataset and process it
    if args.sts:
        df = pd.read_csv('extra_data/sick2014.csv', sep='\t')
        # change column names
        df = df.rename(columns={'pair_ID': 'id', 'sentence_A': 'sentence1', 'sentence_B': 'sentence2', 'relatedness_score': 'similarity'})
        df = df[['id', 'sentence1', 'sentence2', 'similarity']]
        # rescale similarity values
        df.loc[:,'similarity'] = df.loc[:,'similarity'] - 1
        df.loc[:,'similarity'] = df.loc[:,'similarity'] * 1.25
        # concatenate with sts train dataset, and save to drive
        df = pd.concat([df, pd.read_csv('data/sts-train.csv', sep='\t', index_col=0)])
        df.to_csv('data/sts-train-expanded.csv', sep='\t')
        print('sts generated and saved')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst", action='store_true')
    parser.add_argument("--para", action='store_true')
    parser.add_argument("--sts", action='store_true')
    args=parser.parse_args()
    return args

if __name__ == "__main__":
    print('running construct_datasets.py')
    args = get_args()
    construct_datasets(args)
