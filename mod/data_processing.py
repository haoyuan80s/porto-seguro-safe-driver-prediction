from .importing import *

#__all__ = ['load_data', 'load_noisy_data']

DATA_PATH = "./data"

def raw_data():
    x = os.listdir(f'{DATA_PATH}')
    if 'train_feather' in x:
        df_train = pd.read_feather(f'{DATA_PATH}/train_feather')
    else:
        df_train = pd.read_csv(f'{DATA_PATH}/train.csv')
    if 'test_feather' in x:
        df_test = pd.read_feather(f'{DATA_PATH}/test_feather')
    else:
        df_test = pd.read_csv(f'{DATA_PATH}/test.csv')
    return df_train, df_test

def train_val_test(split_ratio = 0.1):
    df_tr, df_te = raw_data()
    y = df_tr['target']
    df_tr.drop(columns='target', inplace = True)
    X_tr, X_val, y_tr, y_val = train_test_split(df_tr, y, test_size=split_ratio, random_state=42)
    print(f'X_tr.shape = {X_tr.shape}, X_val.shape = {X_val.shape}, X_te.shape = {df_te.shape}')
    return X_tr, y_tr, X_val, y_val, df_te

def feature_eng(X):
    dorp_list = ['id'] + [col for col in X.columns if 'calc' in col]
    one_hot_feature = []
    X.drop(columns = dorp_list, inplace = True)
    for col in tqdm(X.columns):
        if 'cat' in col: one_hot_feature.append(pd.get_dummies(X[col], prefix=col))    
        X[col] = rank_gauss(X[col])
    X = pd.concat([X, *one_hot_feature], axis = 1)
    for col in tqdm(X.columns):
        X[col] = X[col] - np.mean(X[col])
    return X

def load_data():
    return (
        pickle.load(open(f'{DATA_PATH}/X_tr.array', 'rb')),
        pickle.load(open(f'{DATA_PATH}/y_tr.array', 'rb')),
        pickle.load(open(f'{DATA_PATH}/X_val.array', 'rb')),
        pickle.load(open(f'{DATA_PATH}/y_val.array', 'rb')),
        pickle.load(open(f'{DATA_PATH}/X_te.array', 'rb'))
    )

def load_noisy_data():
    return [x[1] for x in np.load(f'{DATA_PATH}/noisy_data.npz').items()]

def rank_gauss(x):
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2 
    efi_x = erfinv(rank_x) 
    #efi_x -= efi_x.mean() # since we subtrace mean later
    return efi_x

def swap_noise(X, p = 0.15):
    X_ = X[:]
    m,n = X.shape
    for i in tqdm(range(m)):
        for j in range(n):
            if np.random.rand() < p:
                i_ = np.random.choice(m)
                X_[i,j] = X[i_,j]
    return X_
