"""Data preparation of credit default data.

Pre processing that can be carried out before splitting into training and test data:
    Ordered categorical columns are ordinal encoded.
    Unordered categorical columns are one hot encoded.

In general, the categories need to cover the entire data set.
In this data set the categories are pre defined.
"""

# Author: Graham Hay


import pandas as pd
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

INPUT_FILE = 'dataset_31_credit-g.csv'

FEATURES_NUMERIC = ['duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age',
                    'existing_credits', 'num_dependents']

FEATURES_ORDINAL = ['checking_status', 'savings_status', 'employment', 'own_telephone', 'foreign_worker']

FEATURES_NOMINAL = ['credit_history', 'purpose', 'personal_status', 'other_parties',
                    'property_magnitude', 'other_payment_plans', 'housing', 'job']


def preprocess_data():
    """Pre process the Credit default data.

    Load the credit default data, perform pre processing of categorical features
    and encode the target labels.

    Returns
    -------

    x : ndarray of shape (>2,)
        Expanded data set as a numpy array with additional columns for one hot encoded features.

    y : ndarray of shape (1,)
        Encoded target labels.

    features : list of feature names
        Expanded list of feature names with the additional one hot encoded columns.

    classes : list of class labels
        Class labels in order negative=0, positive=1

"""

    df = pd.read_csv(INPUT_FILE)
    print(INPUT_FILE)
    print('Input data shape:', df.shape)

    # LabelEncoder sorts the labels alphabetically, so 'bad' -> 0 and 'good' -> 1,
    # so the positive class is 'good' and the classes list should be classes = ['bad', 'good'].
    # We want 'bad' loans are the positive class but LabelEncoder sorts the labels alphabetically,
    # so flip the zeros and ones and the order of the class names.
    # NB: This requires the Decision tree to have class_weights='balanced' to work properly.
    classes = ['good', 'bad']
    class_label_encoder = LabelEncoder()
    class_label_encoder.fit(df['class'])
    y = 1 - class_label_encoder.transform(df['class'])

    # For 'good' as the positive class, use the following code instead.
    # classes = ['bad', 'good']
    # class_label_encoder = LabelEncoder()
    # class_label_encoder.fit(df['class'])
    # y = class_label_encoder.transform(df['class'])

    # Numeric features
    df_numeric = df[FEATURES_NUMERIC]

    # Ordinal (Categorical ordered) features
    df_ordinal = DataFrame()
    df_ordinal['checking_status'] = pd.Categorical(df['checking_status'], ordered=True,
                                    categories=["'no checking'", "'<0'", "'0<=X<200'", "'>=200'"]).codes
    df_ordinal['savings_status'] = pd.Categorical(df['savings_status'], ordered=True,
                                   categories=["'no known savings'", "'<100'", "'100<=X<500'",
                                                      "'500<=X<1000'", "'>=1000'"]).codes
    df_ordinal['employment'] = pd.Categorical(df['employment'], ordered=True,
                               categories=["unemployed", "'<1'", "'1<=X<4'", "'4<=X<7'", "'>=7'"]).codes
    df_ordinal['own_telephone'] = pd.Categorical(df['own_telephone'], ordered=True,
                                  categories=['none', 'yes']).codes
    df_ordinal['foreign_worker'] = pd.Categorical(df['foreign_worker'], ordered=True,
                                   categories=['no', 'yes']).codes

    # Nominal (Categorical unordered) features
    df_nominal = DataFrame()

    df_nominal['credit_history'] = pd.Categorical(df['credit_history'], ordered=True,
                                   categories=["'all paid'", "'critical/other existing credit'",
                                               "'delayed previously'", "'existing paid'",
                                               "'no credits/all paid'"])

    df_nominal['purpose'] = pd.Categorical(df['purpose'], ordered=True,
                            categories=["business", "'domestic appliance'", "education", "furniture/equipment",
                                        "'new car'", "other", "radio/tv", "repairs", "retraining",
                                        "'used car'"])

    # NB: 'female single' is not present in the data
    df_nominal['personal_status'] = pd.Categorical(df['personal_status'], ordered=True,
                                    categories=["'female single'", "'male single'", "'female div/dep/mar'",
                                                "'male mar/wid'", "'male div/sep'"])

    df_nominal['other_parties'] = pd.Categorical(df['other_parties'], ordered=True,
                                  categories=["none", "'co applicant'", "guarantor"])

    df_nominal['property_magnitude'] = pd.Categorical(df['property_magnitude'], ordered=True,
                                  categories=["car", "'life insurance'", "real estate", "'no known property'"])

    df_nominal['other_payment_plans'] = pd.Categorical(df['other_payment_plans'], ordered=True,
                                        categories=["bank", "stores", "none"])

    df_nominal['housing'] = pd.Categorical(df['housing'], ordered=True,
                            categories=["'for free'", "own", "rent"])

    df_nominal['job'] = pd.Categorical(df['job'], ordered=True,
                        categories=["'high qualif/self emp/mgmt'", "skilled",
                                    "'unemp/unskilled non res'", "'unskilled resident'"])

    df_nominal = pd.get_dummies(df_nominal)
    # df_nominal = pd.get_dummies(df[features_nominal])
    # df_nominal = pd.get_dummies(df, columns=features_nominal)

    # Dataframe
    x = pd.concat([df_numeric, df_ordinal, df_nominal], axis=1)

    feature_names = x.columns.tolist()

    return x.values, y, feature_names, classes


def make_scale_transformer():
    """Make a scaling column transformer the Credit default data as a numpy array.

    The numeric features are standardised and the ordinal features are normalised.
    All other features (nominal) are unchanged.

    Returns
    -------

    ct : ColumnTransformer
        Column transformer that takes a numpy array and scales the features.
"""
    ct = ColumnTransformer([('numeric', StandardScaler(), list(range(len(FEATURES_NUMERIC)))),
                            ('ordinal', MinMaxScaler(), list(range(len(FEATURES_NUMERIC),
                                                             len(FEATURES_NUMERIC)+len(FEATURES_ORDINAL))))],
                           remainder='passthrough')
    return ct

