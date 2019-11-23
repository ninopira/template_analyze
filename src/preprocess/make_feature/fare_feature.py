
def make_fare_Feature(df_train, df_test):
    print('preprocessing_Fare...')
    # only for df_test, since there is a missing "Fare" values
    df_test["Fare"].fillna(df_test["Fare"].median(), inplace=True)

    # convert from float to int
    df_train['Fare'] = df_train['Fare'].astype(int)
    df_test['Fare'] = df_test['Fare'].astype(int)