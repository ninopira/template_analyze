import os
import tempfile

import click
import lightgbm as lgb
import toml
import pandas as pd

from logger import build_logger
from model import nn
import util


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--cleanup', is_flag=True)
@click.option('--short_mode', is_flag=True)
def train(config_path, cleanup, short_mode):
    config = toml.load(config_path)
    # base_result_dir = config['result_dir'].format(now_str=now_str(str_format='%Y%m%d_%H%M%S'))
    base_result_dir = os.path.join(config['result_dir'], 'train')
    with tempfile.TemporaryDirectory() as f:
        if cleanup:
            print('cleanup mode')
            print('temporary directory: {}'.format(f))
            base_result_dir = f
        else:
            os.makedirs(base_result_dir, exist_ok=True)
        config['result_dir'] = base_result_dir

        # writing_config
        with open(os.path.join(config['result_dir'], 'config.toml'), 'a') as f:
            toml.dump(config, f)
        # building_logger
        logger_file_path = os.path.join(base_result_dir, 'log.txt')
        logger = build_logger(logger_file_path)
        # set_train_config
        preprocess_dir_name = config['preprocess_params']['preprocessed_dir_name']
        n_splits = config['preprocess_params']['n_splits']
        preprocess_base_dir_path = os.path.join('../data/preprocessed_data/', preprocess_dir_name)

        # fold分のdirがあるので、fold数分回す
        for fold in range(n_splits):
            fold_dir_name = '{}cv_{}'.format(n_splits, fold+1)
            logger.info('{}_start'.format(fold_dir_name))
            preprocess_dir_path = os.path.join(
                preprocess_base_dir_path, fold_dir_name)
            result_dir = os.path.join(base_result_dir, fold_dir_name)
            os.makedirs(result_dir, exist_ok=True)
            _train(config, preprocess_dir_path, result_dir, logger, short_mode=False)


def _train(config, preprocess_dir_path, result_dir, logger, short_mode):
    # read_data
    logger.info('reading_train_val_csv...')
    df_train = pd.read_csv(os.path.join(preprocess_dir_path, 'train.csv'))
    df_val = pd.read_csv(os.path.join(preprocess_dir_path, 'val.csv'))
    logger.debug('shape_of_train_is_{}'.format(df_train.shape))
    logger.debug('shape_of_val_is_{}'.format(df_val.shape))

    # set_config
    model_type = config['model_type']
    model_params = config['train_params']
    if short_mode:
        model_params['n_iter'] = 2

    # final_preprocess
    logger.info('drop_useless_cols...')
    useless_cols = ['PassengerId', 'Name', 'Ticket']
    categorical_cols = ['Cabin', 'Embarked', 'Person']
    df_train .drop(useless_cols, axis=1, inplace=True)
    df_val .drop(useless_cols, axis=1, inplace=True)
    logger.info('shape_of_train_is_{}'.format(df_train.shape))

    # Models not supportiong categorical_feature
    if model_type not in ['LightGBM', 'RF']:
        logger.info('transfer_categorical_to_dummy...')
        df_train, df_val = util.transfer_dummy(df_train, df_val, categorical_cols, logger)

    # chack_the_shape_of_train_val
    if not df_train.shape[1] == df_val.shape[1]:
        raise ValueError('shape_of_train_{}_is_different_from_val_{}'.format(
            df_train.shape[1], df_val.shape[1]))

    # separete_X_Y
    Y_train = df_train['Survived']
    X_train = df_train.drop('Survived', axis=1)
    Y_val = df_val['Survived']
    X_val = df_val.drop('Survived', axis=1)
    logger.info('shape_of_train_x_is_{}'.format(X_train.shape))
    logger.info('shape_of_val_x_is_{}'.format(X_val.shape))

    # NN
    if model_type == 'NN':
        # build_model
        logger.info('building_model...')
        model = nn.Mlp(
            input_size=X_train.shape[1], result_dir=result_dir, num_class=2)
        # train_mmodel
        logger.info('start_train...')
        history = model.fit(x=X_train,
                            y=Y_train,
                            batch_size=model_params['batch_size'],
                            epochs=model_params['n_iter'],
                            validation_data=(X_val, Y_val))
        logger.info('{}epoch_train_oss:{}'.format(config['n_iter'], history.history['loss'][-1]))
        logger.info('{}epoch_val_oss:{}'.format(config['n_iter'], history.history['val_loss'][-1]))
        model.save()

    # LigtGBM
    if model_type == 'LightGBM':
        X_train[categorical_cols] = X_train[categorical_cols].astype('category')
        X_val[categorical_cols] = X_val[categorical_cols].astype('category')
        Y_train = Y_train.astype('bool')
        Y_val = Y_val.astype('bool')
        trn_data = lgb.Dataset(data=X_train, label=Y_train, categorical_feature=categorical_cols)
        val_data = lgb.Dataset(data=X_val, label=Y_val, categorical_feature=categorical_cols)
        model_params['bagging_seed'] = config['seed']
        num_rounds = config['n_iter']

        model = lgb.train(params=model_params,
                          train_set=trn_data,
                          num_boost_round=num_rounds,
                          valid_sets=[trn_data, val_data],
                          verbose_eval=100,
                          early_stopping_rounds=100)
    logger.info('end_train')


if __name__ == '__main__':
    train()
