import os
import pandas as pd

# Lấy đường dẫn thư mục chứa file dataflow.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# dirname: Trả về thư mục chứa file

# Load dữ liệu với đường dẫn đầy đủ
df_train = pd.read_csv(os.path.join(BASE_DIR, '../data/train.csv'), parse_dates=['Date'])
# parse_dates: Chuyển cột Date sang kiểu dữ liệu datetime
# os.path.join(BASE_DIR, '../data/train.csv') # Trả về đường dẫn đầy đủ của file train.csv
df_test = pd.read_csv(os.path.join(BASE_DIR, '../data/test.csv'), parse_dates=['Date'])
df_geography = pd.read_csv(os.path.join(BASE_DIR, '../data/geography.csv'))
df_product = pd.read_csv(os.path.join(BASE_DIR, '../data/product.csv'))
