import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.preprocessing import StandardScaler

# Lấy đường dẫn thư mục chứa file dataflow.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load dữ liệu với đường dẫn đầy đủ
df_train_origin = pd.read_csv(os.path.join(BASE_DIR, '../data/train.csv'), parse_dates=['Date'])
df_test_origin = pd.read_csv(os.path.join(BASE_DIR, '../data/test.csv'), parse_dates=['Date'])
df_geography_origin = pd.read_csv(os.path.join(BASE_DIR, '../data/geography.csv'))
df_product_origin = pd.read_csv(os.path.join(BASE_DIR, '../data/product.csv'))

df_train = df_train_origin.copy()
df_test = df_test_origin.copy()

df_train['Date'] = pd.to_datetime(df_train['Date'])
df_test['Date'] = pd.to_datetime(df_test['Date'])

# Lấy các cột cần thiết cho mô hình
df_model = df_train[['Date', 'Units', 'Revenue']]
df_test = df_test[['Date', 'Units', 'Revenue']]

# Nhóm theo ngày và tính tổng số lượng bán (Units) và doanh thu (Revenue) mỗi ngày
df_train = df_model.groupby('Date').agg({'Units': 'sum', 'Revenue': 'sum'}).reset_index()
df_test = df_test.groupby('Date').agg({'Units': 'sum', 'Revenue': 'sum'}).reset_index()

### XỬ LÝ DỮ LIỆU THIẾU ###
# Tạo dãy ngày đầy đủ từ ngày nhỏ nhất đến ngày lớn nhất trong dữ liệu
full_date_range = pd.date_range(start=df_train['Date'].min(), end=df_train['Date'].max(), freq='D')

# Đặt cột Date làm index để dễ thao tác
df_train.set_index('Date', inplace=True)

# Reindex DataFrame với dãy ngày đầy đủ. Các ngày thiếu sẽ có giá trị NaN
df_train = df_train.reindex(full_date_range)
df_train.index.name = 'Date'

# Sử dụng nội suy tuyến tính dựa trên thời gian để điền các giá trị thiếu
df_train['Units'] = df_train['Units'].interpolate(method='time')
df_train['Revenue'] = df_train['Revenue'].interpolate(method='time')

# Trong trường hợp giá trị tại đầu hoặc cuối chuỗi vẫn là NaN, sử dụng forward/backward fill
df_train['Units'] = df_train['Units'].ffill().bfill()
df_train['Revenue'] = df_train['Revenue'].ffill().bfill()

# Reset index để đưa Date trở lại làm cột thông thường
df_train = df_train.reset_index()

def add_time_features(df):
    """
    Thêm các đặc trưng thời gian, sử dụng cyclical encoding cho day_of_week và month.
    Giả định df có cột Date dạng datetime.
    """
    df['day_of_week'] = df['Date'].dt.dayofweek  # 0: Thứ Hai, 6: Chủ Nhật
    df['month'] = df['Date'].dt.month            # 1 đến 12

    # Cyclical encoding cho day_of_week
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Cyclical encoding cho month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Các cờ khác
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df['Date'].min(), end=df['Date'].max())
    df['is_holiday'] = df['Date'].isin(holidays).astype(int)
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    return df

# Hàm bổ sung các đặc trưng lag và rolling statistics cho df
def add_lag_features(df):
    # Giả sử df đã được sắp xếp theo Date và các cột 'Units', 'Revenue' đã được nội suy và chuẩn hóa
    df = df.sort_values('Date').reset_index(drop=True)
    # Rolling 7 ngày
    df['Units_mean_7'] = df['Units'].rolling(window=7, min_periods=1).mean()
    df['Units_std_7']  = df['Units'].rolling(window=7, min_periods=1).std().fillna(0)
    df['Revenue_mean_7'] = df['Revenue'].rolling(window=7, min_periods=1).mean()
    df['Revenue_std_7']  = df['Revenue'].rolling(window=7, min_periods=1).std().fillna(0)
    
    # Rolling 30 ngày
    df['Units_mean_30'] = df['Units'].rolling(window=30, min_periods=1).mean()
    df['Units_std_30']  = df['Units'].rolling(window=30, min_periods=1).std().fillna(0)
    df['Revenue_mean_30'] = df['Revenue'].rolling(window=30, min_periods=1).mean()
    df['Revenue_std_30']  = df['Revenue'].rolling(window=30, min_periods=1).std().fillna(0)
    return df

scaler = StandardScaler()

df_train = add_time_features(df_train)
df_test = add_time_features(df_test)

df_train[['Units', 'Revenue']] = scaler.fit_transform(df_train[['Units', 'Revenue']]) 
df_test[['Units', 'Revenue']] = scaler.transform(df_test[['Units', 'Revenue']])

df_train = add_lag_features(df_train)
df_test = add_lag_features(df_test)

class TimeSeriesDataset(Dataset):
    def __init__(self, df, window_size=30):
        """
        Mỗi mẫu gồm:
         - x_seq: chuỗi lịch sử (window_size ngày) với 2 biến liên tục đã chuẩn hóa: Units và Revenue.
         - x_cal: các đặc trưng lịch của ngày dự báo, bao gồm các cột ngoài ['Date', 'Units', 'Revenue'].
         - y: giá trị dự báo của ngày đó (Units, Revenue).
        """
        self.window_size = window_size
        self.df = df.sort_values('Date').reset_index(drop=True)
        # Các cột đặc trưng lịch là tất cả các cột ngoại trừ Date, Units, Revenue
        self.calendar_feature_cols = [col for col in self.df.columns if col not in ['Date', 'Units', 'Revenue']]
        self.seq_cols = ['Units', 'Revenue']
        
    def __len__(self):
        return len(self.df) - self.window_size
    
    def __getitem__(self, idx):
        x_seq = self.df.loc[idx:idx+self.window_size-1, self.seq_cols].values.astype(np.float32)
        x_cal = self.df.loc[idx+self.window_size, self.calendar_feature_cols].values.astype(np.float32)
        y = self.df.loc[idx+self.window_size, ['Units', 'Revenue']].values.astype(np.float32)
        return x_seq, x_cal, y

# Tạo dataset và DataLoader cho train (với validation tách ngẫu nhiên) và test
full_dataset = TimeSeriesDataset(df_train, window_size=30)
test_dataset = TimeSeriesDataset(df_test, window_size=30)

# Tách tập train và validation (20% validation)
seed = 42
torch.manual_seed(seed)
dataset_length = len(full_dataset)
val_size = int(0.2 * dataset_length)
train_size = dataset_length - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)