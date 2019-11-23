import numpy as np
import pandas as pd


def make_age_Feature(df_train, df_test):
    print('preprocessing_Age...')
    # get average, std, and number of NaN values in df_train
    average_age_titanic = df_train['Age'].mean()
    std_age_titanic = df_train['Age'].std()

    # get average, std, and number of NaN values in df_test
    average_age_test = df_test['Age'].mean()
    std_age_test = df_test['Age'].std()

    # generate random numbers between (mean - std) & (mean + std)
    def age_train_rand():
        return np.random.randint(average_age_titanic - std_age_titanic,
                                 average_age_titanic + std_age_titanic)

    def age_test_rand():
        return np.random.randint(average_age_test - std_age_test,
                                 average_age_test + std_age_test)

    # fill NaN values in Age column with random values generated
    df_train['Age'].fillna(age_train_rand(), inplace=True)
    df_test['Age'].fillna(age_test_rand(), inplace=True)

    # convert from float to int
    df_train['Age'] = df_train['Age'].astype(int)
    df_test['Age'] = df_test['Age'].astype(int)
