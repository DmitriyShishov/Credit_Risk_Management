import pandas as pd
import dill
from datetime import datetime
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from lightgbm import LGBMClassifier


import warnings
warnings.filterwarnings('ignore')


def main_df_load(x):
    df = pd.DataFrame()

    for i in range(x):
        temp_df = pd.read_parquet(f'train_data/train_data_{i}.pq')
        for col in temp_df.columns:
            if np.all((temp_df[col] >= np.iinfo(np.int8).min) & (temp_df[col] <= np.iinfo(np.int8).max)):
                temp_df[col] = temp_df[col].astype('int8')

        df = pd.concat([df, temp_df])

        del temp_df
    return df


def drop_duplicates(df):
    df = df.drop_duplicates()
    return df


def drop_nan(df):
    df = df.dropna()
    return df


def encoding(df):
    cols_for_encoder = df.drop(['id', 'rn', 'is_zero_loans5', 'is_zero_loans530',
                                'is_zero_loans3060', 'is_zero_loans6090',
                                'is_zero_loans90', 'is_zero_util',
                                'is_zero_over2limit', 'is_zero_maxover2limit',
                                'pclose_flag', 'fclose_flag'], axis=1).columns.tolist()

    df = pd.get_dummies(df, columns=cols_for_encoder, dtype='int8')
    return df


def agg_func(df):
    max_columns = ['rn']
    sum_columns = df.drop(['id', 'rn'], axis=1).columns.tolist()

    agg_dict = {col: 'max' for col in max_columns}
    agg_dict.update({col: 'sum' for col in sum_columns})

    df = df.groupby('id', as_index=False).agg(agg_dict)

    for col in df.columns:
        if col != 'id':
            if np.all((df[col] >= np.iinfo(np.int8).min) & (df[col] <= np.iinfo(np.int8).max)):
                df[col] = df[col].astype('int8')
            elif np.all((df[col] >= np.iinfo(np.int16).min) & (df[col] <= np.iinfo(np.int16).max)):
                df[col] = df[col].astype('int16')
            else:
                df[col] = df[col].astype('int32')
    return df


def merge_targets(df):
    target_df = pd.read_csv('target/train_target.csv')
    df = df.merge(target_df, on='id', how='inner')
    del target_df
    return df


def new_features(df):
    for col in df.drop(['id', 'rn', 'flag'], axis=1).columns.tolist():
        df.loc[:, [f'{col}_to_rn']] = df[col] / df['rn']
        if np.all((df[f'{col}_to_rn'] >= np.finfo(np.float16).min) & (df[f'{col}_to_rn'] <= np.finfo(np.float16).max)):
            df[f'{col}_to_rn'] = df[f'{col}_to_rn'].astype('float16')
        else:
            df[f'{col}_to_rn'] = df[f'{col}_to_rn'].astype('float32')
    return df


def drop_columns(df):
    df = df.drop('id', axis=1)
    return df


def main():
    print('ML Junior FinalProject Pipeline')

    df = main_df_load(12)

    preprocessor_func = Pipeline(steps=[
        ('drop_duplicates', FunctionTransformer(drop_duplicates)),
        ('drop_nan', FunctionTransformer(drop_nan)),
        ('encoding', FunctionTransformer(encoding)),
        ('agg_func', FunctionTransformer(agg_func)),
        ('merge_targets', FunctionTransformer(merge_targets)),
        ('new_features', FunctionTransformer(new_features)),
        ('drop_columns', FunctionTransformer(drop_columns))
    ])
    lgbm_best_params = {
        'learning_rate': 0.08,
        'n_estimators': 300,
        'reg_alpha': 0.1,
        'reg_lambda': 0.4,
        'metric': 'auc',
        'class_weight': 'balanced',
        'verbose': 0
    }

    models = [
        LGBMClassifier(**lgbm_best_params)
    ]

    df = preprocessor_func.fit_transform(df)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    x_train = df_train.drop('flag', axis=1)
    y_train = df_train['flag']
    x_test = df_test.drop('flag', axis=1)
    y_test = df_test['flag']

    for model in models:
        pipe = Pipeline(steps=[
            ('classifier', model)
        ])

        pipe.fit(x_train, y_train)

        y_pred = pipe.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, pipe.predict_proba(x_test)[:, 1])

        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print(f"ROC AUC: {roc_auc:.3f}")

        with open('model/MLJ_final_project_model.pkl', 'wb') as file:
            dill.dump({
                'model': pipe,
                'metadata': {
                    'name': 'MLJ_final_project_model',
                    'author': 'Dmitry Shishov',
                    'version': 1,
                    'date': datetime.now(),
                    'type': type(pipe.named_steps["classifier"]).__name__,
                    'accuracy': roc_auc
                }
            }, file, recurse=True)

        print('Model is saved')


if __name__ == '__main__':
    main()
