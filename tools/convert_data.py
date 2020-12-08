import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--file', help='File to update')
args = parser.parse_args()

f = pd.read_csv(args.file)
keep_col = ['ACC', 'GENDER', 'HCC18-CNN', 'HCC22-CNN', 'HCC40-CNN', 'HCC48-CNN', 'HCC59-CNN', 'HCC85-CNN', 'HCC96-CNN', 'HCC108-CNN', 'HCC111-CNN', 'HCC138-CNN', 'RAF', 'AGE']
new_f = f[keep_col]
new_f.to_csv("test.csv", index=False)
