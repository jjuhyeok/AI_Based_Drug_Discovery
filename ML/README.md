## 1. Feature_Engineering.ipynb
- Generates derived features using files from the `dacon` and `processed_data` folders located inside the `data_preprocessing` directory.
- **Output:** train_final.csv, test_final.csv

## 2. Aug.ipynb
- Performs data augmentation on the test dataset.
- **Output:** test_final_aug_724.csv

## 3. Emb.ipynb
- Creates embedding representations for training, testing, and augmented test datasets.
- **Output:** train_emb_final.parquet, test_emb_final.parquet, test_aug_724_emb_final.csv

## 4. ML.ipynb
- Requires importing `my_metrics.py` located in the same folder.
- Needs a CSV file containing the results of the OGNN model.
- **Output:** Final.csv
