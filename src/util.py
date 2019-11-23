from datetime import datetime

import pandas as pd


def transfer_dummy(df_train, df_val, cols, logger):
    logger.info('transfer_categorical_to_dummy...')
    df_all = pd.concat([df_train, df_val])
    for col in cols:
        col_dummies_titanic = pd.get_dummies(df_all[col])
        df_all = df_all.join(col_dummies_titanic)
        df_all.drop([col], axis=1, inplace=True)
    df_train = df_all.iloc[:df_train.shape[0], :]
    df_val = df_all.iloc[:df_val.shape[0], :]
    return df_train, df_val


def now_str(str_format='%Y%m%d%H%M'):
    return datetime.now().strftime(str_format)
