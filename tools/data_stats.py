import pandas as pd
import numpy as np
from argparse import ArgumentParser
from PIL import Image

conditions = ['GENDER', 'HCC18', 'HCC22', 'HCC40', 'HCC48', 'HCC59', 'HCC85', 'HCC96', 'HCC108', 'HCC111', 'HCC138']

def arg_parse():
    parser = ArgumentParser()
    parser.add_argument('--data_path', default='data', help='Path to data csv')
    return parser.parse_args()

args = arg_parse()
df = pd.read_csv(args.data_path)

for c in conditions:
    print(df[c].value_counts())

age = np.asarray(df.AGE)
print(np.mean(age))
print(np.median(age))

#calculate mean and std of datset
# df = pd.read_csv(args.data_path)
# vals = np.zeros(2)
# for i in df['FILE']:
#     img = Image.open('data/raf_data/data/' + i).convert('L')
#     img = np.asarray(img)
#     vals[0] += img.mean()
#     vals[1] += img.std()
# print(vals/len(df))
