# data_preprocessing README.md
------------------------------------------------------------------------------
Operating System: Windows 10
Python Version: 3.8.20
CPU: AMD64
CPU Cores: 12 physical, 16 logical
Total Memory: 15.69 GB
Available Memory: 6.62 GB
Number of GPUs: 0
------------------------------------------------------------------------------

# 1. data_preprocessing environment setup
    cd data_preprocessing
    conda env create -f environment.yml
    conda activate chem

# 2. folder explanation
    - dacon: Raw data from Dacon competition.
    - chembl_data: Raw data from ChEMBL database.
    - pubchem_data: Raw assay data from PubChem, split by assay type:
        - HTS: CYP3A4 inhibition (%) qHTS data.
        - midazolam: CYP3A4 inhibition (%) with Midazolam substrate.
        - testosterone: CYP3A4 inhibition (%) with Testosterone substrate.
    - data_preprocessing_ipynb: Jupyter Notebooks for data preprocessing workflows.
    - processed_data: Stores final, preprocessed datasets.
    - data_url.csv: Records data sources (URLs) for all datasets.
        
# 3. data preprocessing workflows
    Execute the Jupyter Notebooks in data_preprocessing_ipynb to preprocess raw datasets into clean formats in processed_data.

# 4. Make new_train_dataset using processed_data
 - Execute the 'make_new_train.ipynb' Jupyter notebook to generate 'new_train.csv'.

