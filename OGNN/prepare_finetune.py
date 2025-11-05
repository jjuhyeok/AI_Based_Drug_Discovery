from sklearn.model_selection import KFold
import numpy as np
import os
import pandas as pd

data = pd.read_csv('data/train.csv')

# ID 컬럼이 없다면 생성
if 'ID' not in data.columns:
    data['ID'] = ['TRAIN_' + str(i) for i in range(len(data))]

# 이후 기존 코드 그대로 사용
data = data[data['ID'].str.contains('TRAIN')].reset_index(drop=True)

if not os.path.exists('split_fold'):
    os.mkdir('split_fold')

for seed in [0, 1000, 2000, 1113, 2023]:
    ksplit = KFold(n_splits=5, shuffle=True, random_state=seed)

    for k, (t_idx, v_idx) in enumerate(ksplit.split(data)):
        train_ids = data.loc[t_idx]['ID'].values
        valid_ids = data.loc[v_idx]['ID'].values
        np.save(f'split_fold/train_{seed}_{k + 1}.npy', train_ids)
        np.save(f'split_fold/val_{seed}_{k + 1}.npy', valid_ids)

