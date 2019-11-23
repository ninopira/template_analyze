"""
https://www.kaggle.com/omarelgabry/a-journey-through-titanic
"""

import os

import click
import toml
import pandas as pd
from sklearn.model_selection import KFold

from make_feature import age_fearure, embarked_feature, family_feature, \
                         fare_feature, pclass_feature, sex_feature


def build_feature(df_train, df_test):
    embarked_feature.make_embarked_Feature(df_train)
    fare_feature.make_fare_Feature(df_train, df_test)
    age_fearure.make_age_Feature(df_train, df_test)
    family_feature.make_family_Feature(df_train, df_test)
    sex_feature.make_sex_Feature(df_train, df_test)
    pclass_feature.make_pclass_Feature(df_train, df_test)
    return df_train, df_test


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def preprocess_main(config_path):
    # set_config
    config = toml.load(config_path)
    seed = config['seed']
    n_splits = config['preprocess_params']['n_splits']
    preprocessed_dir_name = config['preprocess_params']['preprocessed_dir_name']
    preprocessed_base_dir_path = os.path.join('../../data/preprocessed_data', preprocessed_dir_name)

    # read_data
    df_train = pd.read_csv('../../data/raw_data/train.csv')
    df_test = pd.read_csv('../../data/raw_data/test.csv')
    print('size_of_train_is_{}'.format(df_train.shape))
    print('size_of_test_is_{}'.format(df_test.shape))

    # feature_engenieering
    df_train, df_test = build_feature(df_train, df_test)
    print('size_of_preprocessed_train_is_{}'.format(df_train.shape))
    print('size_of_preprocessed_test_is_{}'.format(df_test.shape))

    # write_all_train_test
    train_test_info_dict = {
        'train': {'df': df_train, 'csv_name': 'train.csv'},
        'test': {'df': df_test, 'csv_name': 'test.csv'},
    }
    preprocessed_dir_path = os.path.join(preprocessed_base_dir_path, 'all')
    os.makedirs(preprocessed_dir_path, exist_ok=True)
    for subject in train_test_info_dict:
        csv_path = os.path.join(preprocessed_dir_path, train_test_info_dict[subject]['csv_name'])
        train_test_info_dict[subject]['df'].to_csv(csv_path)
    print('all_train_shape', df_train.shape, 'all_test_shape', df_test.shape)

    # write_with_cv
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_, (train_idx, val_idx) in enumerate(folds.split(df_train.values)):
        preprocessed_dir_path = os.path.join(preprocessed_base_dir_path, '{}cv_{}'.format(n_splits, fold_+1))
        os.makedirs(preprocessed_dir_path, exist_ok=True)
        train = df_train.iloc[train_idx, :]
        val = df_train.iloc[val_idx, :]
        train.to_csv(os.path.join(preprocessed_dir_path, 'train.csv'), index=False)
        val.to_csv(os.path.join(preprocessed_dir_path, 'val.csv'), index=False)
        print('train_shape', train.shape, 'val_shape', val.shape)
        if not train.shape[1] == val.shape[1]:
            print('shape_of_train_{}_is_different_from_val_{}'.format(
                train.shape[1], val.shape[1]))


if __name__ == '__main__':
    preprocess_main()
