import os
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--train', help='Test file to check')
parser.add_argument('--test', help='Train file to check')
args = parser.parse_args()

train_df = pd.read_csv(args.train)
test_df = pd.read_csv(args.test)

ctr = 0
# currently only counts duplicates
for i, item in enumerate(test_df['FILE']):
    if item in train_df.values:
        print(item)
        ctr += 1

print(ctr)
