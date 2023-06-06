import numpy as np
import pandas as pd

path = "./Dataset/DoS/DataForBert.npy"
Header = ['ID', 'DLC', 'Byte1', 'Byte2', 'Byte3', 'Byte4', 'Byte5', 'Byte6', 'Byte7', 'Byte8']

def Normalization(X):
    mu = np.average(X, axis=0)
    sigma = np.std(X, axis=0)
    X = (X - mu) / sigma
    return X

def main(path):
    data = np.load(path)
    Data = []
    for data_item in data:
       data_item = data_item.split(" ")
       Data.append(data_item)
    data = pd.DataFrame(Data, columns=Header)
    # CAN ID ----> Hexadecimal to decimal
    data['ID'] = data['ID'].apply(lambda x: int(x, 16))
    data['DLC'] = data['DLC'].apply(lambda x: int(x, 10))
    # CAN Data Field -----> Hexadecimal to decimal
    for i in range(1, 9):
        data['Byte' + str(i)] = data['Byte' + str(i)].apply(lambda x: int(x, 16))
    data = data.values
    X_Normal = Normalization(data)
    np.save("./Dataset/DoS/Data.npy", X_Normal)

if __name__ == '__main__':
    main(path)