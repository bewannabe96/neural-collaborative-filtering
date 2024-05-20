import numpy as np
import pandas as pd
from torch.utils import data


class ValidationDataset(data.Dataset):
    @staticmethod
    def __convert_neg_item_ids_to_numpy(series, delim='|'):
        num_rows = series.shape[0]
        num_cols = len(series.iloc[0].split(delim))

        numpy_array = np.empty((num_rows, num_cols), dtype=np.int32)
        for i, row in enumerate(series):
            numpy_array[i, :] = np.fromiter((int(x) for x in row.split(delim)), dtype=int, count=num_cols)

        return numpy_array

    def __init__(self, filename, delim='|'):
        super(ValidationDataset, self).__init__()

        df = pd.read_csv(filename, names=['user_id', 'pos_item_id', 'neg_item_ids'], header=None)

        self.len = len(df)

        self.user_id = df['user_id'].to_numpy(dtype=np.int32)
        self.pos_item_id = df['pos_item_id'].to_numpy(dtype=np.int32)
        self.neg_item_ids = ValidationDataset.__convert_neg_item_ids_to_numpy(df['neg_item_ids'], delim)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.user_id[idx], self.pos_item_id[idx], self.neg_item_ids[idx]
