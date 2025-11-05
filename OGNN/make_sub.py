import os
import pandas as pd
import numpy as np
from glob import glob

# test.csv 로드
test = pd.read_csv('/content/drive/MyDrive/OGNN/data/test.csv')
assert 'ID' in test.columns, "❗ test.csv에 'ID' 컬럼이 있어야 합니다."

# npy 파일들 불러오기
pred_files = sorted(glob('/content/drive/MyDrive/OGNN/test_results/*Inhibition*.npy'))

# 저장 디렉토리
save_dir = '/content/drive/MyDrive/OGNN/results'
os.makedirs(save_dir, exist_ok=True)

# fold별 개별 예측 저장
for idx, file in enumerate(pred_files):
    preds = np.load(file)
    df = pd.DataFrame({
        'ID': test['ID'],
        'Inhibition': preds
    })
    save_path = os.path.join(save_dir, f'fold{idx + 1}_prediction.csv')
    df.to_csv(save_path, index=False)
    print(f"✅ {save_path} 저장 완료")
