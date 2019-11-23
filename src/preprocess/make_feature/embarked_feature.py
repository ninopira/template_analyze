def make_embarked_Feature(df_train):
    """
    embarked列の前処理
    (1)naを'S'で穴埋め
    """
    print('preprocessing_embarked...')
    df_train = df_train.fillna({"Embarked": "S"})