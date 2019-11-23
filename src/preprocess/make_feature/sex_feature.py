import pandas as pd


def make_sex_Feature(df_train, df_test):
    print('preprocessing_sex...')
    def get_person(passenger):
        age, sex = passenger
        return 'child' if age < 16 else sex
    df_train['Person'] = df_train[['Age', 'Sex']].apply(get_person, axis=1)
    df_test['Person'] = df_test[['Age', 'Sex']].apply(get_person, axis=1)

    # No need to use Sex column since we created Person column
    df_train.drop(['Sex'], axis=1, inplace=True)
    df_test.drop(['Sex'], axis=1, inplace=True)

    # # create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
    # person_dummies_titanic = pd.get_dummies(df_train['Person'])
    # person_dummies_titanic.columns = ['Child', 'Female', 'Male']
    # person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

    # person_dummies_test = pd.get_dummies(df_test['Person'])
    # person_dummies_test.columns = ['Child', 'Female', 'Male']
    # person_dummies_test.drop(['Male'], axis=1, inplace=True)

    # df_train = df_train.join(person_dummies_titanic)
    # df_test = df_test.join(person_dummies_test)

    # df_train.drop(['Person'], axis=1, inplace=True)
    # df_test.drop(['Person'], axis=1, inplace=True)
