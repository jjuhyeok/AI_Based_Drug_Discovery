import os
import re
import numpy as np
import pandas as pd
from glob import glob

# 1. 공통 경로 설정
base_dir = "."
result_dir = os.path.join(base_dir, "results")
output_dir = os.path.join(base_dir, "result")
test_csv_path = os.path.join(base_dir, "data", "test.csv")

# 2. 테스트 데이터 로드
test_df = pd.read_csv(test_csv_path)
assert 'ID' in test_df.columns, "❗ test.csv에 'ID' 컬럼이 있어야 합니다."

# 3. 점수 추출 함수 정의
def extract_score(filename):
    match = re.search(r'_(\d\.\d+)\.npy$', filename)
    return float(match.group(1)) if match else -1.0

# 4. Fold별 파일 탐색 및 저장
for fold in range(1, 6):
    pattern = os.path.join(result_dir, f"test_724_fold{fold}_*.npy")
    files = glob(pattern)

    if not files:
        print(f"❌ Fold {fold}에 해당하는 파일이 없습니다: {pattern}")
        continue

    # 성능 점수 기준으로 가장 높은 파일 선택
    file_path = max(files, key=extract_score)

    pred = np.load(file_path)

    if len(pred) != len(test_df):
        print(f"❗ Fold {fold} 예측값 길이와 test.csv 길이가 일치하지 않습니다.")
        continue

    # 결과 저장
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'Inhibition': pred
    })
    output_path = os.path.join(output_dir, f"fold{fold}_mixup_100_prediction.csv")
    submission.to_csv(output_path, index=False)

    print(f"✅ Fold {fold} 예측 결과 저장 완료: {output_path} (선택된 파일: {os.path.basename(file_path)})")
