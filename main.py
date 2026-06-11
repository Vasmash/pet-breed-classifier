import torch
import numpy as np
from torch.utils.data import Dataset


# превращаем CSV данные в PyTorch датасет, который выдает пары (изображение, метки)
class UTKDataset(Dataset):
    def __init__(self, df, func=None):
        self.func = func
        
        data_arr = df.pixels.apply(lambda x: np.array(x.split(" "),dtype=float))
        arr = np.stack(data_arr)
        arr = arr / 255.0
        arr = arr.astype('float32')
        arr = arr.reshape(arr.shape[0], 48, 48, 1)
        self.data = arr
        
        self.age = np.array(df.bins[:])     
        self.gender = np.array(df.gender[:])
        self.eth = np.array(df.ethnicity[:])
    
    # переопределяем функцию длины
    def __len__(self):
        return len(self.data)

    # переопределяем функцию getitem
    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.func(data)
        
        labels = torch.tensor((self.age[idx], self.gender[idx], self.eth[idx]))
        
        return data, labels

