import numpy as np
import pandas as pd
from argparse import ArgumentParser

def arg_parse():
    parser = ArgumentParser()
    parser.add_argument('--in_file', default='output/out.csv', help='File with prediction values to be converted')
    parser.add_argument('--thresh', default=0.5, type=float, help='Threshold for predictions')
    parser.add_argument('--out_file', default='output/predicted.csv', help='Output file with predictions')
    args = parser.parse_args()
    return args

args = arg_parse()

df = pd.read_csv(args.in_file)
conditions = list(df.columns)[2:-2]

# convert df to proper labels
gender = np.asarray(df.GENDER)
gender = np.where(gender > args.thresh, 'male', 'female')
df.GENDER = gender

# go through all hcc codes and convert to labels
for c in conditions:
    condition = np.asarray(df[c])
    condition = np.where(condition > args.thresh, 'PRESENT', 'ABSENT')
    df[c] = condition

df.to_csv(args.out_file, index=False)
