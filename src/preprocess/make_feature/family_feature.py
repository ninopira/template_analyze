
import numpy as np
import pandas as pd


def make_family_Feature(df_train, df_test):
    # Instead of having two columns Parch & SibSp,
    # we can have only one column represent if the passenger had any family member aboard or not,
    # Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not
    print('preprocessing_Family...')
    df_train['Family'] = df_train["Parch"] + df_train["SibSp"]
    df_train.loc[df_train['Family'] > 0, 'Family'] = 1
    df_train.loc[df_train['Family'] == 0, 'Family'] = 0

    df_test['Family'] = df_test["Parch"] + df_test["SibSp"]
    df_test.loc[df_test['Family'] > 0, 'Family'] = 1
    df_test.loc[df_test['Family'] == 0, 'Family'] = 0

    # drop Parch & SibSp
    df_train = df_train.drop(['SibSp', 'Parch'], axis=1)
    df_test = df_test.drop(['SibSp', 'Parch'], axis=1)
