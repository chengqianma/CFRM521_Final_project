import numpy as np
import pandas as pd


def read_file(train_data_path, train_label_path):

    train_data_df = pd.read_csv(train_data_path, index_col=False)
    train_label_df = pd.read_csv(train_label_path, index_col=False)

    # train_data_df = train_data_df.drop(columns=['Unnamed: 0'])

    train_data_df['customer_ID'] = train_data_df['customer_ID'].apply(lambda x: int(x[-16:], 16)).astype('int64')
    # train_data_df.S_2 = pd.to_datetime(train_data_df.S_2)
    # train_data_df.fillna(nan_value)
    train_label_df['customer_ID'] = train_label_df['customer_ID'].apply(lambda x: int(x[-16:], 16)).astype('int64')

    return train_data_df, train_label_df


def feature_engineer(train, PAD_CUSTOMER_TO_13_ROWS = True, targets = None):
        
    # REDUCE STRING COLUMNS 
    # from 64 bytes to 8 bytes, and 10 bytes to 3 bytes respectively
    # train['customer_ID'] = train['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
    train_data_df.S_2 = pd.to_datetime(train_data_df.S_2)
    train['year'] = (train.S_2.dt.year-2000).astype('int8')
    train['month'] = (train.S_2.dt.month).astype('int8')
    train['day'] = (train.S_2.dt.day).astype('int8')
    del train['S_2']
        
    # LABEL ENCODE CAT COLUMNS (and reduce to 1 byte)
    # with 0: padding, 1: nan, 2,3,4,etc: values
    d_63_map = {'CL':2, 'CO':3, 'CR':4, 'XL':5, 'XM':6, 'XZ':7}
    train['D_63'] = train.D_63.map(d_63_map).fillna(1).astype('int8')

    d_64_map = {'-1':2,'O':3, 'R':4, 'U':5}
    train['D_64'] = train.D_64.map(d_64_map).fillna(1).astype('int8')
    
    CATS = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_66', 'D_68']
    OFFSETS = [2,1,2,2,3,2,3,2,2] #2 minus minimal value in full train csv
    # then 0 will be padding, 1 will be NAN, 2,3,4,etc will be values
    for c,s in zip(CATS,OFFSETS):
        train[c] = train[c] + s
        train[c] = train[c].fillna(1).astype('int8')
    CATS += ['D_63','D_64']
    
    # ADD NEW FEATURES HERE
    # EXAMPLE: train['feature_189'] = etc etc etc
    # EXAMPLE: train['feature_190'] = etc etc etc
    # IF CATEGORICAL, THEN ADD TO CATS WITH: CATS += ['feaure_190'] etc etc etc
    
    # REDUCE MEMORY DTYPE
    SKIP = ['customer_ID','year','month','day']
    for c in train.columns:
        if c in SKIP: continue
        if str( train[c].dtype )=='int64':
            train[c] = train[c].astype('int32')
        if str( train[c].dtype )=='float64':
            train[c] = train[c].astype('float32')
            
    # PAD ROWS SO EACH CUSTOMER HAS 13 ROWS
    if PAD_CUSTOMER_TO_13_ROWS:
        tmp = train[['customer_ID']].groupby('customer_ID').customer_ID.agg('count')
        more = np.array([], dtype='int64') 
        for j in range(1,13):
            i = tmp.loc[tmp==j].index.values
            more = np.concatenate([more, np.repeat(i, 13 - j)])
        df = train.iloc[:len(more)].copy().fillna(0)
        df = df * 0 - 1 #pad numerical columns with -1
        df[CATS] = (df[CATS] * 0).astype('int8') #pad categorical columns with 0
        df['customer_ID'] = more
        train = pd.concat([train,df],axis=0,ignore_index=True)
        
    # ADD TARGETS (and reduce to 1 byte)
    if targets is not None:
        train = train.merge(targets,on='customer_ID',how='left')
        train.target = train.target.astype('int8')
        
    # FILL NAN
    train = train.fillna(-0.5) #this applies to numerical columns
    
    # SORT BY CUSTOMER THEN DATE
    train = train.sort_values(['customer_ID','year','month','day']).reset_index(drop=True)
    train = train.drop(['year','month','day'],axis=1)
    
    # REARRANGE COLUMNS WITH 11 CATS FIRST
    COLS = list(train.columns[1:])
    COLS = ['customer_ID'] + CATS + [c for c in COLS if c not in CATS]
    train = train[COLS]
    
    return train


if __name__ == "__main__":
    train_data_path = "/gscratch/ubicomp/cm74/AMEX/tiny_train_data.csv"
    train_label_path = "/gscratch/ubicomp/cm74/AMEX/train_labels.csv"
    train_data_df, train_label_df = read_file(train_data_path, train_label_path)
    print(train_data_df.head())
    processed_train_data = feature_engineer(train_data_df)
    print(processed_train_data.shape)
    # df = pd.read_csv(file_path)
    # tiny_df = df.head(n=(len(df) // 100))
    # tiny_df.to_csv("tiny_train_data.csv")
    # print(df.head())