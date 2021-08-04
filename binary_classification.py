import random
import numpy as np
import joblib
import os
import glob
import shutil
import pandas as pd
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import itertools
from itertools import accumulate
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif, f_regression,\
    VarianceThreshold, SelectKBest, chi2, SelectFromModel
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, train_test_split, ParameterGrid
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import auc, plot_roc_curve, accuracy_score, \
    confusion_matrix, roc_curve, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def imbalaced_ratio(n_samples, n_classes, n_samples_class_i) -> float:
    """
    seek for the weights that inversely proportional to class frequencies in the input data
    :param n_samples: number of overall samples
    :param n_classes: number of classes
    :param n_samples_class_i: number of samples in class i
    :return: ratio to multiply each class in the imbalanced data
    """
    ratio = n_samples / (n_classes * n_samples_class_i)
    return ratio


def scale_data(df_data_train: pd.DataFrame) -> pd.DataFrame:
    """
    fit sk-learn StandardScaler on data
    :param df_data_train: data-frame of the data to scale
    :return: data frame of the scaled data
    """
    train_data_array: np.ndarray = df_data_train.to_numpy()
    scaler = StandardScaler().fit(train_data_array)
    train_scaled_data_array: np.ndarray = scaler.transform(train_data_array)
    return scaler.mean_, scaler.scale_, pd.DataFrame(train_scaled_data_array, columns=df_data_train.columns)


def remove_no_variance_features(df_data: pd.DataFrame, t=(.99 * (1 - .99))):
    """
    scale data in [0,1] (sk-learn MinMaxScaler) and apply sk-learn VarianceThreshold to remove features (columns)
    with variance lower the threshold
    :param df_data: data-frame of the data to remove low variance features (columns)
    :param t: threshold for variance with default value (.99 * (1 - .99))
    :return: data-frame of data removed the low variance features and numpy array of the corresponding columns indices
    """
    sel = VarianceThreshold(threshold=t)
    scaler = MinMaxScaler()
    # scaler.fit(df_data.to_numpy())
    # scaled_data = scaler.fit_transform(df_data.to_numpy())
    scaled_data = scaler.fit_transform(df_data)
    var_data: np.ndarray = sel.fit_transform(scaled_data)
    mask_array = sel.get_support(indices=True)
    df: pd.DataFrame = pd.DataFrame(var_data, columns=mask_array)
    return df, mask_array


def features_score(df_data: pd.DataFrame, df_target: pd.DataFrame, method: str = chi2) -> pd.Series:
    """
    seek uni-variate relation strength between features and target by
    applying sk-learn SelectKBest to calculate importance of all features (columns) in the input data
    :param df_data: data-frame of the data
    :param df_target: data-frame of the target
    :param method: callable score_func
    :return: pandas series with the data features importance
    """
    feature_score = SelectKBest(score_func=method, k='all').\
        fit(df_data.to_numpy(), np.ravel(df_target.to_numpy())).scores_
    df_feature_score: pd.DataFrame = pd.Series(feature_score)
    return df_feature_score


def uni_feature_selection(features_indices: list, features_names: pd.Index,
                      df_data_train: pd.DataFrame, df_target_train: pd.DataFrame, method) -> pd.DataFrame:
    """
    create a data-frame with feature indices, names and relative importance that represents the uni-variate relation
    strength to the target according to method
    :param features_indices: list of columns indices corresponding to columns in df_data_train
    :param features_names: data-frame with features names in corresponding to features indices
    :param df_data_train: data-frame of data to calculate it's features (columns) importance. columns names are indices
    :param df_target_train: data-frame of data to explore uni-variate relation with each feature
    :param method: callable score_func
    :return: data-frame with feature indices, names and relative importance
    """
    col_0 = pd.Series(features_indices, name='feature_num')
    col_1 = pd.Series(features_names, name='feature')
    feature_importance = features_score(df_data_train, df_target_train, method)
    col_2 = pd.Series(feature_importance, name='score')
    df_feature_importance = pd.concat([col_0, col_1, col_2], axis=1)
    return df_feature_importance


def selected_features(df_select_rank: pd.DataFrame, top_univariate: int = 50) -> pd.DataFrame:
    """
    select the minimum between input number and the most influential features (at least in one univariate method this feature was selected)
    :param df_select_rank: data-frame which is sorted first in descending order of the sum number of methods
        that selected the feature and second in ascending order of their sum of ranks
    :param top_univariate: int that determine the number of first row of df_select_rank to return
    :return: data-frame with number rows of data-frame df_select_rank.
    """
    index_importance = df_select_rank['sum_ind'] >= 1
    number = np.minimum(top_univariate, np.sum(index_importance))
    return df_select_rank[['feature_num', 'feature']].head(number)


def select_rank(
        fs_ls: list, min_to_select: int = 10, weighing: bool = False) -> pd.DataFrame:
    """
    create a data-frame with indices, names and relative importance that represents the relation
    strength to the target and combining the corresponding feature importance, selection indicator, and rank
    :param fs_ls: list of features importance from methods
    :param min_to_select: minimum features to select in each method
    :return: df_select_rank: data-frame which is sorted first in descending order of the sum number of methods
        that selected the feature and second in ascending order of their sum of ranks
    """
    #create df_select_rank with columns['feature_num','feature','sum_ind','sum_rank']
    df: pd.DataFrame = fs_ls[0]
    df_select_rank = df.drop(['score'], axis=1)
    df_select_rank['sum_ind'] = 0
    df_select_rank['sum_rank'] = 0
    if weighing:
        weight = 1 / len(fs_ls)
    else:
        weight = 1

    for fs in fs_ls:
        df_ordered = fs.sort_values(by=['score'], ascending=False, ignore_index=True)
        sum_score = df_ordered['score'].sum()
        # select features with 95% of the accumulated score
        Threshold1 = 0.95 * sum_score
        accum_df = pd.DataFrame(accumulate(df_ordered['score']))
        index1 = accum_df.index
        condition1 = accum_df < Threshold1
        condition1_indices = index1[condition1.iloc[:, 0]]
        #make sure selected features are with at least 1% of total accumulated score
        Threshold2 = 0.01 * sum_score
        index2 = df_ordered['score'].index
        condition2 = df_ordered['score'] > Threshold2
        condition2_indices = index2[condition2]
        #intersection of two conditions
        condition_indices = condition1_indices.intersection(condition2_indices)
        condition = np.logical_and(np.array(condition1), np.array(pd.DataFrame(condition2)))
        df_ordered['rank'] = 1 + np.arange(len(condition2))
        # sanity check - minimum of features selected
        if len(condition_indices) >= min_to_select:
            df_ordered['ind'] = pd.DataFrame(condition)
            df_ordered = df_ordered.sort_values(by=['feature_num'], ascending=True, ignore_index=True)
        else:
            condition = df_ordered['rank'] <= min_to_select
            df_ordered['ind'] = pd.DataFrame(condition)
            df_ordered = df_ordered.sort_values(by=['feature_num'], ascending=True, ignore_index=True)

        df_select_rank['sum_ind'] += weight * df_ordered['ind']
        df_select_rank['sum_rank'] += weight * df_ordered['rank']

    df_select_rank.sort_values(by=['sum_ind', 'sum_rank'], ascending=[False, True], inplace=True, ignore_index=True)

    return df_select_rank

# inbar test conflic omer





def sequential_forward_selection(
        df_data_train: pd.DataFrame, features_names: pd.DataFrame, df_target_train: pd.DataFrame, clf_ls: list,
        k_feature_range: tuple) -> list:
    """
    create a data-frame with feature indices, names and relative importance that represents the multi-variate relation
    strength to the target according to sequential feature selection (SFS) methods.
    execute several SFS corresponding to the given classifiers list and combining the corresponding feature importance,
    selection indicator, and rank
    :param df_data_train: data-frame of data to calculate it's features (columns) importance. columns names are indices
    :param df_target_train: data-frame of data to explore multi-variate relation with each feature
    :param clf_ls: list of classifiers to explore forward selection corresponding features
    :param k_feature_range: interval range to limit the forward selection. could also be 'best' or 'parsimonious'
    :return: list of lists - a list per classifier. a classifier list includes the following SFS items:
    0: k_feature_idx_ - Feature Indices of the selected feature subsets,
    1: k_feature_names_ - Feature names of the selected feature subsets,
    2: k_score_ - Cross validation average score of the selected subset,
    3: subsets_ - dictionary with MLxtend sequential forward selection subsets,
    4: list of ordered indices of the features according to their introduction order to the model
    """
    list_ls: list = []

    def create_select_indicator(df_data_train: pd.DataFrame, list_ls: list) -> list:
        """
        combine the selection indicators and ranks of the features according to the given classifiers
        :param df_data_train: data-frame of data to calculate it's features (columns) importance. columns names are indices
        :param list_ls: list of lists - a list per classifier. a classifier list includes the following SFS items:
            0: k_feature_idx_ - Feature Indices of the selected feature subsets,
            1: k_feature_names_ - Feature names of the selected feature subsets,
            2: k_score_ - Cross validation average score of the selected subset,
            3: subsets_ - dictionary with MLxtend sequential forward selection subsets,
            4: list of ordered indices of the features according to their introduction order to the model
        :return: list of two data-frames.
            the first data-frame with feature indices, names and selector indicator column per classifier that represents
            whether the feature was selected according to that classifer.
            the second data-frame with feature indices, names and rank column per classifier that represents
            the order when the feature was introduced to the model according to that classifer.
        """
        col_array = df_data_train.columns.to_numpy(int)
        df: pd.DataFrame = pd.DataFrame(col_array).rename(columns={0: 'feature_num'})
        df_rank: pd.DataFrame = df
        df_rank['feature'] = features_names
        for ls in range(len(list_ls)):
            indx = np.asarray(list_ls[ls][1])
            col = pd.Series(np.where(np.in1d(col_array, indx) == False, 0, 1))
            rank_array = np.zeros(col_array.shape, dtype=int)
            counter: int = 1
            for item in list_ls[ls][4]:
                row_indx: int = df[df['feature_num'] == item].index.values.astype(int)[0]
                rank_array[row_indx] = counter
                counter = counter + 1
            df = pd.concat([df, col], axis=1)
            rank_col: pd.Series = pd.Series(rank_array)
            df_rank = pd.concat([df_rank, rank_col], axis=1)
        sum_col = pd.Series(df.iloc[:, 1:len(list_ls) + 1].sum(axis=1), name='sum_ind')
        sum_rank = pd.Series(df_rank.iloc[:, 1:len(list_ls) + 1].sum(axis=1), name='sum_rank')
        df = pd.concat([df, sum_col], axis=1)
        df_rank = pd.concat([df_rank, sum_rank], axis=1)
        return [df, df_rank]

    def ordered_features(subsets: dict) -> list:
        """
        get MLxtend sequential forward selection subsets dict and return a list with the features indices ordered
        according to their addition to the model
        :param subsets: dictionary with MLxtend sequential forward selection subsets
            A dictionary of selected feature subsets during the sequential selection,
            where the dictionary keys are the lengths k of these feature subsets. The dictionary values are
            dictionaries themselves with the following keys: 'feature_idx' (tuple of indices of the feature subset)
            'feature_names' (tuple of feature names of the feat. subset)
            'cv_scores' (list individual cross-validation scores)
            'avg_score' (average cross-validation score)
        :return: list of feature indices according to the order they were added to the model
        """
        df_subsets = pd.DataFrame.from_dict(subsets)
        ls: list = []
        ls.append([*set(df_subsets.iloc[3, 0]), ])
        for col_indx in range(len(df_subsets.columns) - 1):
            set1 = set(df_subsets.iloc[3, col_indx])
            set2 = set(df_subsets.iloc[3, col_indx + 1])
            diff = set2.difference(set1)
            ls.append([*diff, ])
        merged = list(itertools.chain.from_iterable(ls))
        return merged

    for clf in clf_ls:
        ls: list = []
        sfs = SFS(clf,
                  k_features=k_feature_range,
                  forward=True,
                  floating=False,
                  verbose=2,
                  cv=0)
        sfs_fit = sfs.fit(df_data_train, df_target_train)
        ls.append(sfs_fit.k_feature_idx_)
        ls.append(sfs_fit.k_feature_names_)
        ls.append(sfs_fit.k_score_)
        ls.append(sfs_fit.subsets_)
        ls.append(ordered_features(sfs_fit.subsets_))
        list_ls.append(ls)
    list_ls.append(create_select_indicator(df_data_train, list_ls))
    return list_ls

def remove_correlation_redundancy(df_data_train: pd.DataFrame, threshold: int = 0.75) -> list:
    """
    remove features with Pearson correlation higher than threshold
    :param df_data_train: data-frame of data that might have correlated features
    :param threshold: correlation threshold with default value 0.75
    :return: list with features indices that have correlation lower than threshold
    """
    corr_df = df_data_train.corr(method='pearson')
    corr_df['2nd_large'] = corr_df.apply(lambda row: row.nlargest(2).values[-1], axis=1)
    mask_threshold: np.ndarray = np.where(abs(corr_df.values) > threshold, 1, 0)
    corr_df['depend#'] = mask_threshold.sum(axis=1) - 1
    df_corr_count = pd.DataFrame(np.unique(corr_df['depend#'].values), columns=['depend_level'])
    bincount = np.bincount(corr_df['depend#'].values)
    df_corr_count['count'] = bincount[df_corr_count['depend_level']]
    sorted_corr_df = corr_df.sort_values(by=['depend#', '2nd_large'], ascending=[True, True])
    independent_set = set()
    ls: list = []
    for depend_level in df_corr_count['depend_level']:
        row_feature_indx = sorted_corr_df[sorted_corr_df['depend#'] == depend_level].index
        if depend_level == 0:
            independent_set = independent_set.union(row_feature_indx)
            continue
        for row in row_feature_indx:
            # get the features indices that has correlation greater than threshold with the feature in row
            row_series = sorted_corr_df.loc[row].drop(['depend#', '2nd_large'])
            col_feature_indx = row_series[abs(row_series) > 0.75].index
            corr_set = set(col_feature_indx)
            if independent_set.isdisjoint(corr_set):
                independent_set.add(row)
    ls.append([*independent_set, ])
    independent_indx_list = list(itertools.chain.from_iterable(ls))
    return independent_indx_list


def plot_importance(df_score: pd.DataFrame):
    """
    sort df_score according to the features importance and plot it's bar graph
    :param df_score: data-frame with feature indices, names and relative importance columns
    """
    df_score_sorted = df_score.sort_values(
        by=df_score.columns[2], ascending=False).reset_index().drop(columns=['index'])
    plt.figure()
    df_score_sorted.plot(kind='bar')
    plt.show()


def tree_tune_pruning(
        features_indices: list, features_names: pd.DataFrame, df_data: pd.DataFrame,
        df_target: pd.DataFrame, folds_num: int, min_to_select: int) -> pd.DataFrame:
    """
    create a data-frame with feature indices, names and relative importance that represents the multi-variate relation
    strength to the target according to a decision tree classifier.
    5-fold fitting a ccp tuned tree and combining the corresponding feature import, selection indicator, and rank
    :param folds_num: the number of fold to split the train data
    :param features_indices: list of columns indices corresponding to columns in df_data
    :param features_names: data-frame with features names in corresponding to features indices
    :param df_data: data-frame of data to calculate it's features (columns) importance. columns names are indices
    :param df_target: data-frame of data to explore multi-variate relation with each feature
    :return: data-frame with feature indices, names and relative importance and rank per fold
    along with the sum of the indicators and sum of rank
    """

    # split the data into folds_num stratified folds
    skf = StratifiedKFold(n_splits=folds_num, random_state=0)
    data_array: np.ndarray = df_data.to_numpy()
    target_array: np.ndarray = df_target.to_numpy()
    # define a decision tree classifier
    clf = DecisionTreeClassifier(random_state=0, criterion='entropy')
    # define an empty list that would facilitate the feature score and ranking per test fold
    ls_fold: list = []
    df_score_rank_fold = pd.DataFrame()
    # loop over the folds to train a decision tree and compute cost-complexity parameters
    # test each parameter on test, use the best score to get that tree feature importance
    # then use algorithm to indicate relevant features and their rank
    for fold_num, (train_index, test_index) in enumerate(skf.split(data_array, target_array)):
        data_array_train, data_array_test = data_array[train_index], data_array[test_index]
        target_array_train, target_array_test = target_array[train_index], target_array[test_index]
        path = clf.cost_complexity_pruning_path(data_array_train, target_array_train)
        cv_accuracy_ls: list = []
        ccp_alphas = path.ccp_alphas
        for ccp_alpha in ccp_alphas:
            clf_alpha = DecisionTreeClassifier(random_state=0, criterion='entropy',
                                               ccp_alpha=ccp_alpha)
            clf_alpha.fit(data_array_train, target_array_train)
            cv_accuracy_ls.append(clf_alpha.score(data_array_test, target_array_test))
        df_plot: pd.DataFrame = pd.concat([pd.Series(ccp_alphas, name='ccp_alpha'),
                                           pd.Series(cv_accuracy_ls, name='cv_accuracy')], axis=1)
        # df_plot.plot.scatter(x='ccp_alpha', y='cv_accuracy')
        df_plot.sort_values(by=['cv_accuracy', 'ccp_alpha'], ascending=[False, False], inplace=True)
        # use the lowest ccp_alpha in the highest cv_accuracy category
        # if this ccp_alpha is the max at the category it means that the entire tree is pruned
        # in this case take the 2nd lowest ccp_alpha that correspond to the 2nd highest cv_accuracy
        grouped = df_plot.groupby('cv_accuracy')
        if (grouped.max() - grouped.min()).iloc[-1, 0] != 0:
            best_ccp_alpha = grouped.min().iloc[-1, 0]
        else:
            best_ccp_alpha = grouped.min().iloc[-2, 0]

        clf = DecisionTreeClassifier(random_state=0, criterion='entropy',
                                     ccp_alpha=best_ccp_alpha)
        clf_fit = clf.fit(data_array_train, target_array_train)
        feature_importance = clf_fit.feature_importances_

        df_score_rank_fold['feature_num'] = features_indices
        df_score_rank_fold['feature'] = features_names
        df_score_rank_fold['score'] = feature_importance
        ls_fold.append(df_score_rank_fold)
    weighing = True
    df_score_rank = select_rank(ls_fold, min_to_select, weighing)
    return df_score_rank


def etl(gait_data: pd.DataFrame, headers: pd.Index, datasets_dir: str,
        is_test: bool, test_percentage: float, is_loo: bool,
        k: int, is_balance: bool):
    """
    :param gait_data: pd.DataFrame with features data and target (no missing values)
    :param headers: pd.Index with features headers
    :param datasets_dir: directory to save processed data and splits k-folds
    :param is_test: True = hold out data for test, False = no hold out data for test
    :param test_percentage: hold out percentage data for test
    :param is_loo: True = leave one out CV, False = k fold CV
    :param k: number of CV folds
    :param is_balance: True = balance the train data
    :return: train_dir: directory where train data is saved
    :return: testper_dir: directory where hold out test and train data is saved (relevant if is_test= True)
    :return: test_dir: directory where test data is saved (relevant if is_test= True)
    """
    if is_test:
        test_percentage: float = test_percentage
    else:
        test_percentage: float = 0.0
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
    # is_test directory
    if is_test:
        is_test_dir: str = os.path.join(datasets_dir, 'is_test_' + str(is_test))
        if not os.path.exists(is_test_dir):
            os.makedirs(is_test_dir)
        testper_dir: str = os.path.join(is_test_dir, str(test_percentage) + 'per_test_set')
        if not os.path.exists(testper_dir):
            os.makedirs(testper_dir)
        train_dir: str = os.path.join(testper_dir, 'train_set', str(k) + 'folds')
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        test_dir: str = os.path.join(testper_dir, 'test_set')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
    else:
        is_test_dir: str = os.path.join(datasets_dir, 'is_test_' + str(is_test))
        if not os.path.exists(is_test_dir):
            os.makedirs(is_test_dir)
        train_dir: str = os.path.join(is_test_dir, str(k) + 'folds')
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        testper_dir: str =''
        test_dir: str = ''
    # set seed for current run
    if not os.path.exists(os.path.join(train_dir, 'seed')):
        seed: int = random.randint(1, 4000)
        joblib.dump(seed, os.path.join(train_dir, 'seed'))
        random.seed(seed)
        print(seed)

    # split data into input and output elements
    x, y = gait_data[:, :-1], gait_data[:, -1]
    # label encode the target variable
    y = LabelEncoder().fit_transform(y)
    # save the original data sets
    df_data = pd.DataFrame(x)
    df_target = pd.DataFrame(y)
    joblib.dump(df_data, os.path.join(train_dir, 'df_data'))
    joblib.dump(df_target, os.path.join(train_dir, 'df_target'))
    joblib.dump(headers, os.path.join(train_dir, 'headers'))
    if is_test and len(os.listdir(test_dir)) == 0:
        # split to train(80%) test(20%) sets
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_percentage,
                                                            stratify=y, random_state=42)
        df_data_test = pd.DataFrame(X_test)
        df_target_test = pd.DataFrame(y_test)
        df_data_train = pd.DataFrame(X_train)
        df_target_train = pd.DataFrame(y_train)
        joblib.dump(
            df_data_test, os.path.join(testper_dir, 'test_set', 'df_data_test'))
        joblib.dump(
            df_target_test, os.path.join(testper_dir, 'test_set', 'df_target_test'))
        joblib.dump(
            df_data_train, os.path.join(testper_dir, 'train_set', 'df_data_train'))
        joblib.dump(
            df_target_train, os.path.join(testper_dir, 'train_set', 'df_target_train'))
    elif is_test and len(os.listdir(test_dir)) != 0:
        df_data_train = joblib.load(
            os.path.join(testper_dir, 'train_set', 'df_data_train'))
        df_target_train = joblib.load(
            os.path.join(testper_dir, 'train_set', 'df_target_train'))
    elif not is_test and len(os.listdir(train_dir)) < k + 5:
        # k+5 ---> there should be k folders (one for each fold)
        # and 5 more files (fs_config_dir, df_data, df_target, headers, seed)
        df_data_train = pd.DataFrame(x)
        df_target_train = pd.DataFrame(y)
        joblib.dump(
            df_data_train, os.path.join(train_dir, 'df_data_train'))
        joblib.dump(
            df_target_train, os.path.join(train_dir, 'df_target_train'))
    if len(os.listdir(train_dir)) < k + 5:
        # split to k folds cv or leave one oue
        df_data: pd.DataFrame = df_data_train
        df_target: pd.DataFrame = df_target_train
        if is_loo:
            cv_split = LeaveOneOut()
            cv_split.get_n_splits(x)
        else:
            cv_split = StratifiedKFold(n_splits=k, random_state=1495)

        for fold_num, (train_index, test_index) in enumerate(cv_split.split(df_data, df_target)):
            df_data_train: pd.DataFrame = df_data.loc[train_index, :]
            df_target_train: pd.DataFrame = df_target.loc[train_index]
            df_data_test: pd.DataFrame = df_data.loc[test_index, :]
            df_target_test: pd.DataFrame = df_target.loc[test_index]
            # balance the train data
            # label encode the target variable
            x = df_data_train
            y = LabelEncoder().fit_transform(np.ravel(df_target_train))
            # summarize distribution
            print('imbalanced data: fold=%d' % fold_num)
            counter = Counter(y)
            for u, v in counter.items():
                per = v / len(y) * 100
                print('Class=%d, n=%d (%.3f%%)' % (u, v, per))
            # # plot the distribution
            # # plt.bar(counter.keys(), counter.values())
            # # plt.show()
            if is_balance:
                n_samples: int = len(y)
                n_classes: int = len(counter.items())
                if np.argmax([list(counter.items())[0][1], list(counter.items())[1][1]], axis=0):
                    undersample_group: tuple = list(counter.items())[1]
                    oversample_group: tuple = list(counter.items())[0]
                else:
                    undersample_group: tuple = list(counter.items())[0]
                    oversample_group: tuple = list(counter.items())[1]

                # # balance dataset
                undersample_indx = np.random.choice(np.argwhere(y == undersample_group[0])[:, 0],
                                                    int(undersample_group[1] *
                                                        imbalaced_ratio(n_samples, n_classes, undersample_group[1])),
                                                    replace=False)
                oversample_indx = np.random.choice(np.argwhere(y == oversample_group[0])[:, 0],
                                                   int(oversample_group[1] *
                                                       imbalaced_ratio(n_samples, n_classes, oversample_group[1])))

                df_target_train_balanced = y[undersample_indx]
                df_target_train_balanced = pd.DataFrame(np.append(df_target_train_balanced, y[oversample_indx]))
                df_data_train_balanced = x.iloc[undersample_indx, :]
                df_data_train_balanced = pd.DataFrame(np.append(df_data_train_balanced,
                                                                x.iloc[oversample_indx, :], axis=0))
                # summarize new distribution
                counter = Counter(np.ravel(df_target_train_balanced))
                print('balanced data: fold=%d' % fold_num)
                for u, v in counter.items():
                    per = v / len(df_target_train_balanced) * 100
                    print('Class=%d, n=%d (%.3f%%)' % (u, v, per))
                # plot the distribution
                # plt.bar(counter.keys(), counter.values())
                # plt.show()
            else:
                df_target_train_balanced = pd.DataFrame(y)
                df_data_train_balanced = pd.DataFrame(x)

            if not os.path.exists(os.path.join(train_dir, 'fold_' + str(fold_num))):
                os.makedirs(os.path.join(train_dir, 'fold_' + str(fold_num)))
            joblib.dump(
                train_index, os.path.join(train_dir, 'fold_' + str(fold_num),
                                          'train_index'))
            joblib.dump(
                df_data_train_balanced, os.path.join(train_dir, 'fold_' + str(fold_num),
                                                     'df_data_train'))
            joblib.dump(
                df_target_train_balanced, os.path.join(train_dir, 'fold_' + str(fold_num),
                                                       'df_target_train'))
            joblib.dump(
                test_index, os.path.join(train_dir, 'fold_' + str(fold_num),
                                         'test_index'))
            joblib.dump(
                df_data_test, os.path.join(train_dir, 'fold_' + str(fold_num),
                                           'df_data_test'))
            joblib.dump(
                df_target_test, os.path.join(train_dir, 'fold_' + str(fold_num),
                                             'df_target_test'))
    pass
    return train_dir, testper_dir, test_dir


def cfg_fs(train_dir: str) -> str:
    """
    create directore with current feature selection configuration
    :param train_dir: directory where train data is saved
    :return: config_fs_dir: directory where feature selection configuration is saved
    """
    # parameters for feature selection tuning
    # select minimum features in each uni-variate analysis
    min_to_select: int = 10
    # select top features following uni-variate analysis
    top_univariate: int = 65
    top_multivariate: int = 65
    folds_num_tree: int = 5
    # feature selection configuration
    config_fs_dir = os.path.join(train_dir, 'fs_congif_' + 'top' + str(top_univariate) + 'univar' +
                                 '_' + 'top' + str(top_multivariate) + 'multivar' +
                                 '_' + str(folds_num_tree) + 'foldsnum')
    # create current feature selection configuration folder
    if not os.path.exists(config_fs_dir):
        os.makedirs(config_fs_dir)
        fs_dict_param = {'min_vars_to_select': min_to_select, 'top_univariate': top_univariate,
                              'top_multivariate': top_multivariate, 'folds_num_tree': folds_num_tree}
        joblib.dump(fs_dict_param, os.path.join(config_fs_dir, 'fs_dict_param'))
    return config_fs_dir


def feature_selection(train_dir: str, config_fs_dir: str, k: int):
    """
    runs uni and multi variate feature selection methods. saves for each fold the top discriminating features
    according to top_univariate and top_multivariate parameters
    :param train_dir: directory where train data is saved
    :param config_fs_dir: directory where feature selection configuration is saved
    :param k: number of CV folds
    """
    # read original data features names
    original_data_header: pd.Index = joblib.load(os.path.join(train_dir, 'headers'))
    # read feature selection parameters dictionary
    fs_dict_param: int = joblib.load(os.path.join(config_fs_dir, 'fs_dict_param'))
    min_to_select: int = fs_dict_param['min_vars_to_select']
    top_univariate: int = fs_dict_param['top_univariate']
    folds_num_tree: int = fs_dict_param['folds_num_tree']
    top_multivariate: int = fs_dict_param['top_multivariate']
    if len(os.listdir(config_fs_dir)) < k+1:
        # loop over k-folds
        for fold_num in range(k):
            print(str(fold_num))
            # create configuration folder for each fold
            config_dir_fold: str = os.path.join(config_fs_dir, 'fold_' + str(fold_num))
            if not os.path.exists(config_dir_fold):
                os.makedirs(config_dir_fold)
            # read train/test datasets
            df_data_train: pd.DataFrame = joblib.load(os.path.join(train_dir,
                                                                   'fold_' + str(fold_num), 'df_data_train'))
            df_target_train: pd.DataFrame = joblib.load(os.path.join(train_dir,
                                                                     'fold_' + str(fold_num), 'df_target_train'))
            df_data_test: pd.DataFrame = joblib.load(os.path.join(train_dir,
                                                                  'fold_' + str(fold_num), 'df_data_test'))
            df_target_test: pd.DataFrame = joblib.load(os.path.join(train_dir,
                                                                    'fold_' + str(fold_num), 'df_target_test'))

            # remove low variance features
            df_data_train_select, mask_array = remove_no_variance_features(df_data_train)
            data_header: pd.DataFrame = original_data_header[mask_array]
            df_data_train: pd.DataFrame = df_data_train.iloc[:, mask_array]
            df_data_test: pd.DataFrame = df_data_test.iloc[:, mask_array]
            # test univariate relations
            df_fclassif: pd.DataFrame = uni_feature_selection(mask_array, data_header,
                                                              df_data_train, df_target_train, f_classif)
            df_mutual: pd.DataFrame = uni_feature_selection(mask_array, data_header,
                                                            df_data_train, df_target_train, mutual_info_classif)
            # df_freg: pd.DataFrame = uni_feature_selection(mask_array, data_header, df_data_train,
            #                                               df_target_train, f_regression)
            df_fclassif.to_csv(os.path.join(config_dir_fold, 'df_fclassif.csv'))
            df_mutual.to_csv(os.path.join(config_dir_fold, 'df_mutual.csv'))
            # df_freg.to_csv(os.path.join(config_dir_fold, 'df_freg.csv'))

            # select top features from univariate feature selection
            ufs_ls = []
            ufs_ls.append(df_fclassif)
            ufs_ls.append(df_mutual)
            # ufs_ls.append(df_freg)
            df_select_rank = select_rank(ufs_ls, min_to_select)
            df_select_univariate = selected_features(df_select_rank, top_univariate)
            uni_mask: np.ndarray = df_select_univariate['feature_num'].to_numpy()
            df_train = df_data_train[uni_mask]
            df_test = df_data_test[uni_mask]
            data_header = original_data_header[uni_mask]

            # examine multivariate relations
            # decision tree based method
            df_select_rank_tree: pd.DataFrame = tree_tune_pruning(uni_mask, data_header, df_train,
                                                                  df_target_train, folds_num_tree, min_to_select)
            df_select_rank_tree = df_select_rank_tree.rename(columns=
                                                             {'sum_ind': 'sum_ind_tree', 'sum_rank': 'sum_rank_tree'})
            df_select_rank_tree.to_csv(os.path.join(config_dir_fold, 'tree.csv'))

            # execute several SFS corresponding to the given classifiers
            means_train, stds_train, df_train_scaled = scale_data(df_train)
            sfs_clf_ls = []
            svm_linear = SVC(C=1.0, kernel='linear', random_state=0)
            gnb = GaussianNB()
            lr = LogisticRegression(random_state=0)
            sfs_clf_ls.append(svm_linear)
            sfs_clf_ls.append(gnb)
            sfs_clf_ls.append(lr)
            sfs_ls: list = sequential_forward_selection(df_train_scaled, data_header, df_target_train,
                                                        sfs_clf_ls, 'parsimonious')
            df_sfs = pd.merge(sfs_ls[-1][0], sfs_ls[-1][1], how='left', on='feature_num')
            df_sfs.sort_values(by=['sum_ind', 'sum_rank'], ascending=[False, True], inplace=True)
            df_sfs.to_csv(os.path.join(config_dir_fold, 'fs.csv'))
            columns = ['feature_num', 'feature_x', 'sum_ind', 'sum_rank']
            df_sfs = df_sfs[columns]
            df_sfs = df_sfs.rename(columns={'feature_x': 'feature'})
            df_sfs = df_sfs.rename(columns={'sum_ind': 'sum_ind_sfs'})
            df_sfs = df_sfs.rename(columns={'sum_rank': 'sum_rank_sfs'})

            # combine multivariate feature selection
            df_select_rank_tree.sort_values(by=['feature_num'], ascending=[True], inplace=True)
            df_sfs.sort_values(by=['feature_num'], ascending=[True], inplace=True)
            dfs = [df_select_rank_tree, df_sfs]
            cols = ['feature_num', 'feature']
            df_multi: pd.DataFrame = pd.concat([d.set_index(cols) for d in dfs], axis=1).reset_index()
            df_multi['sum_ind'] = df_multi['sum_ind_tree'] + df_multi['sum_ind_sfs']
            df_multi['sum_rank'] = df_multi['sum_rank_tree'] + df_multi['sum_rank_sfs']
            df_multi.sort_values(by=['sum_ind', 'sum_rank'], ascending=[False, True], inplace=True)

            df_select_multi: pd.DataFrame = selected_features(df_multi, top_multivariate)
            df_select_multi.to_csv(os.path.join(config_dir_fold, 'multivar_selection.csv'))

            # selecting top features from multivariate methods
            multi_mask: np.ndarray = df_select_multi['feature_num'].to_numpy()
            # normalizing test by train mean and std
            df_train_scaled: pd.DataFrame = df_train_scaled[multi_mask]
            df_test_scaled: pd.DataFrame = (df_test - means_train) / stds_train
            df_test_scaled: pd.DataFrame = df_test_scaled[multi_mask]
            df_train: pd.DataFrame = df_train[multi_mask]
            df_test: pd.DataFrame = df_test[multi_mask]
            features_names = original_data_header[multi_mask]

            joblib.dump(
                df_train, os.path.join(config_dir_fold,
                                       'df_data_train_selected'))
            joblib.dump(
                df_target_train, os.path.join(config_dir_fold,
                                              'df_target_train_selected'))
            joblib.dump(
                df_test, os.path.join(config_dir_fold,
                                      'df_data_test_selected'))
            joblib.dump(
                df_target_test, os.path.join(config_dir_fold,
                                             'df_target_test'))
            joblib.dump(
                features_names, os.path.join(config_dir_fold,
                                             'selected_features_names'))
            joblib.dump(
                df_train_scaled, os.path.join(config_dir_fold,
                                       'df_train_scaled_selected'))
            joblib.dump(
                df_test_scaled, os.path.join(config_dir_fold,
                                      'df_test_scaled_selected'))


def lasso_feature_selection(train_dir: str, config_fs_dir: str, k: int):
    """
    runs uni and multi variate feature selection methods. saves for each fold the top discriminating features
    according to top_univariate and top_multivariate parameters
    :param train_dir: directory where train data is saved
    :param config_fs_dir: directory where feature selection configuration is saved
    :param k: number of CV folds
    """
    # read original data features names
    original_data_header: pd.Index = joblib.load(os.path.join(train_dir, 'headers'))
    # read feature selection parameters dictionary
    fs_dict_param: int = joblib.load(os.path.join(config_fs_dir, 'fs_dict_param'))
    min_to_select: int = fs_dict_param['min_vars_to_select']
    top_univariate: int = fs_dict_param['top_univariate']
    folds_num_tree: int = fs_dict_param['folds_num_tree']
    top_multivariate: int = fs_dict_param['top_multivariate']
    if len(os.listdir(config_fs_dir)) < k+1:
        # loop over k-folds
        for fold_num in range(k):
            print(str(fold_num))
            # create configuration folder for each fold
            config_dir_fold: str = os.path.join(config_fs_dir, 'fold_' + str(fold_num))
            if not os.path.exists(config_dir_fold):
                os.makedirs(config_dir_fold)
            # read train/test datasets
            df_data_train: pd.DataFrame = joblib.load(os.path.join(train_dir,
                                                                   'fold_' + str(fold_num), 'df_data_train'))
            df_target_train: pd.DataFrame = joblib.load(os.path.join(train_dir,
                                                                     'fold_' + str(fold_num), 'df_target_train'))
            df_data_test: pd.DataFrame = joblib.load(os.path.join(train_dir,
                                                                  'fold_' + str(fold_num), 'df_data_test'))
            df_target_test: pd.DataFrame = joblib.load(os.path.join(train_dir,
                                                                    'fold_' + str(fold_num), 'df_target_test'))

            # remove low variance features
            df_data_train_select, mask_array = remove_no_variance_features(df_data_train)
            data_header: pd.DataFrame = original_data_header[mask_array]
            df_data_train: pd.DataFrame = df_data_train.iloc[:, mask_array]
            df_data_test: pd.DataFrame = df_data_test.iloc[:, mask_array]
            # test univariate relations
            df_fclassif: pd.DataFrame = uni_feature_selection(mask_array, data_header,
                                                              df_data_train, df_target_train, f_classif)
            df_mutual: pd.DataFrame = uni_feature_selection(mask_array,data_header,
                                                            df_data_train, df_target_train, mutual_info_classif)
            df_freg: pd.DataFrame = uni_feature_selection(mask_array, data_header, df_data_train,
                                                          df_target_train, f_regression)
            df_fclassif.to_csv(os.path.join(config_dir_fold, 'df_fclassif.csv'))
            df_mutual.to_csv(os.path.join(config_dir_fold, 'df_mutual.csv'))
            df_freg.to_csv(os.path.join(config_dir_fold, 'df_freg.csv'))

            # select top features from univariate feature selection
            ufs_ls = []
            ufs_ls.append(df_fclassif)
            ufs_ls.append(df_mutual)
            ufs_ls.append(df_freg)
            df_select_rank = select_rank(ufs_ls, min_to_select)
            df_select_univariate = selected_features(df_select_rank, top_univariate)
            uni_mask: np.ndarray = df_select_univariate['feature_num'].to_numpy()
            df_train = df_data_train[uni_mask]
            df_test = df_data_test[uni_mask]
            data_header = original_data_header[uni_mask]

            # examine multivariate relations
            # decision tree based method
            df_select_rank_tree: pd.DataFrame = tree_tune_pruning(uni_mask, data_header, df_train,
                                                                  df_target_train, folds_num_tree, min_to_select)
            df_select_rank_tree = df_select_rank_tree.rename(columns=
                                                             {'sum_ind': 'sum_ind_tree', 'sum_rank': 'sum_rank_tree'})
            df_select_rank_tree.to_csv(os.path.join(config_dir_fold, 'tree.csv'))

            # execute several SFS corresponding to the given classifiers
            means_train, stds_train, df_train_scaled = scale_data(df_train)
            sfs_clf_ls = []
            svm_linear = SVC(C=1.0, kernel='linear', random_state=0)
            gnb = GaussianNB()
            lr = LogisticRegression(random_state=0)
            sfs_clf_ls.append(svm_linear)
            sfs_clf_ls.append(gnb)
            sfs_clf_ls.append(lr)
            sfs_ls: list = sequential_forward_selection(df_train_scaled, data_header, df_target_train,
                                                        sfs_clf_ls, 'parsimonious')
            df_sfs = pd.merge(sfs_ls[-1][0], sfs_ls[-1][1], how='left', on='feature_num')
            df_sfs.sort_values(by=['sum_ind', 'sum_rank'], ascending=[False, True], inplace=True)
            df_sfs.to_csv(os.path.join(config_dir_fold, 'fs.csv'))
            columns = ['feature_num', 'feature_x', 'sum_ind', 'sum_rank']
            df_sfs = df_sfs[columns]
            df_sfs = df_sfs.rename(columns={'feature_x': 'feature'})
            df_sfs = df_sfs.rename(columns={'sum_ind': 'sum_ind_sfs'})
            df_sfs = df_sfs.rename(columns={'sum_rank': 'sum_rank_sfs'})

            # combine multivariate feature selection
            df_select_rank_tree.sort_values(by=['feature_num'], ascending=[True], inplace=True)
            df_sfs.sort_values(by=['feature_num'], ascending=[True], inplace=True)
            dfs = [df_select_rank_tree, df_sfs]
            cols = ['feature_num', 'feature']
            df_multi: pd.DataFrame = pd.concat([d.set_index(cols) for d in dfs], axis=1).reset_index()
            df_multi['sum_ind'] = df_multi['sum_ind_tree'] + df_multi['sum_ind_sfs']
            df_multi['sum_rank'] = df_multi['sum_rank_tree'] + df_multi['sum_rank_sfs']
            df_multi.sort_values(by=['sum_ind', 'sum_rank'], ascending=[False, True], inplace=True)

            df_select_multi: pd.DataFrame = selected_features(df_multi, top_multivariate)
            df_select_multi.to_csv(os.path.join(config_dir_fold, 'multivar_selection.csv'))

            # selecting top features from multivariate methods
            multi_mask: np.ndarray = df_select_multi['feature_num'].to_numpy()
            # normalizing test by train mean and std
            df_train_scaled: pd.DataFrame = df_train_scaled[multi_mask]
            df_test_scaled: pd.DataFrame = (df_test - means_train) / stds_train
            df_test_scaled: pd.DataFrame = df_test_scaled[multi_mask]
            df_train: pd.DataFrame = df_train[multi_mask]
            df_test: pd.DataFrame = df_test[multi_mask]
            features_names = original_data_header[multi_mask]

            joblib.dump(
                df_train, os.path.join(config_dir_fold,
                                       'df_data_train_selected'))
            joblib.dump(
                df_target_train, os.path.join(config_dir_fold,
                                              'df_target_train_selected'))
            joblib.dump(
                df_test, os.path.join(config_dir_fold,
                                      'df_data_test_selected'))
            joblib.dump(
                df_target_test, os.path.join(config_dir_fold,
                                             'df_target_test'))
            joblib.dump(
                features_names, os.path.join(config_dir_fold,
                                             'selected_features_names'))
            joblib.dump(
                df_train_scaled, os.path.join(config_dir_fold,
                                       'df_train_scaled_selected'))
            joblib.dump(
                df_test_scaled, os.path.join(config_dir_fold,
                                      'df_test_scaled_selected'))
    # read original data features names
    original_data_header: pd.Index = joblib.load(os.path.join(train_dir, 'headers'))
    if len(os.listdir(config_fs_dir)) < k+1:
        # loop over k-folds
        for fold_num in range(k):
            print(str(fold_num))
            # create configuration folder for each fold
            config_dir_fold: str = os.path.join(config_fs_dir, 'fold_' + str(fold_num))
            if not os.path.exists(config_dir_fold):
                os.makedirs(config_dir_fold)
            # read train/test datasets
            df_data_train: pd.DataFrame = joblib.load(os.path.join(train_dir,
                                                                   'fold_' + str(fold_num), 'df_data_train'))
            df_target_train: pd.DataFrame = joblib.load(os.path.join(train_dir,
                                                                     'fold_' + str(fold_num), 'df_target_train'))
            df_data_test: pd.DataFrame = joblib.load(os.path.join(train_dir,
                                                                  'fold_' + str(fold_num), 'df_data_test'))
            df_target_test: pd.DataFrame = joblib.load(os.path.join(train_dir,
                                                                    'fold_' + str(fold_num), 'df_target_test'))
            # Selecting features using Lasso regularisation
            means_train, stds_train, df_train_scaled = scale_data(df_data_train)
            sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
            sel_.fit(df_train_scaled, df_target_train)
            sel_.get_support()
            selected_feat = df_train_scaled.columns[(sel_.get_support())]
            print('total features: {}'.format((df_train_scaled.shape[1])))
            print('selected features: {}'.format(len(selected_feat)))
            print('features with coefficients shrank to zero: {}'.format(
                np.sum(sel_.estimator_.coef_ == 0)))
            df_train_scaled = df_train_scaled.iloc[selected_feat]
            df_test_scaled: pd.DataFrame = (df_data_test - means_train) / stds_train
            df_test_scaled: pd.DataFrame = df_test_scaled[selected_feat]
            df_train: pd.DataFrame = df_data_train.iloc[selected_feat]
            df_test: pd.DataFrame = df_data_test[selected_feat]
            features_names = original_data_header[selected_feat]

            joblib.dump(
                df_train, os.path.join(config_dir_fold,
                                       'df_data_train_selected'))
            joblib.dump(
                df_target_train, os.path.join(config_dir_fold,
                                              'df_target_train_selected'))
            joblib.dump(
                df_test, os.path.join(config_dir_fold,
                                      'df_data_test_selected'))
            joblib.dump(
                df_target_test, os.path.join(config_dir_fold,
                                             'df_target_test'))
            joblib.dump(
                features_names, os.path.join(config_dir_fold,
                                             'selected_features_names'))
            joblib.dump(
                df_train_scaled, os.path.join(config_dir_fold,
                                       'df_train_scaled_selected'))
            joblib.dump(
                df_test_scaled, os.path.join(config_dir_fold,
                                      'df_test_scaled_selected'))


def cfg_model(train_dir: str, config_fs_dir: str) -> str:
    """
    create classifiers dictionary with hyperparameters for tuning
    :param train_dir: directory where train data is saved
    :param config_fs_dir: directory where feature selection configuration is saved
    :return: models_config_dir: directory where current models configuration is saved
    (saved with current date and time "%m-%d-%Y-%H-%M-%S")
    """
    # set random seed
    seed: int = joblib.load(os.path.join(train_dir, 'seed'))
    models = [
        'XGB',
        'GBC',
        'RFC',
        'SVC',
        'logisticRegression'
    ]
    clfs = [
        XGBClassifier(objective='binary:logistic', random_state=seed),
        GradientBoostingClassifier(random_state=seed),
        RandomForestClassifier(random_state=seed),
        SVC(random_state=seed, probability=True),
        LogisticRegression(penalty='elasticnet', solver='saga', random_state=seed)
    ]
    params = {
            models[0]: {'learning_rate': (0.01, 0.1, 0.3),
                        'n_estimators': (800, 1000), 'max_depth': (3, 5, 8),
                        'subsample': (0.5, 0.8), 'colsample_bytree': (0.5, 0.8), 'min_child_weight': (0, 1)},
            models[1]: {'loss': ('deviance', 'exponential'), 'learning_rate': (0.01, 0.1, 0.3),
                        'n_estimators': (800, 1000), 'max_depth': (3, 5, 8),
                        'subsample': (0.5, 0.8)},
            models[2]: {'criterion': ('gini', 'entropy'), 'n_estimators': (800, 1000), 'max_depth': (3, 5, 8)},
            models[3]: {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'), 'C': (0.1, 1, 10),
                        'gamma': (1, 0.1, 0.01, 0.001)},
            models[4]: {'l1_ratio': (0.1, 0.5, 0.8, 1)}
         }
    is_scale_data = {
        models[0]: False,
        models[1]: False,
        models[2]: False,
        models[3]: True,
        models[4]: True,
    }
    models_config_dir = os.path.join(config_fs_dir, 'models_config_' +
                                     datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
    if not os.path.exists(models_config_dir):
        os.makedirs(models_config_dir)
    joblib.dump(clfs, os.path.join(models_config_dir, 'clfs'))
    joblib.dump(params, os.path.join(models_config_dir, 'params'))
    joblib.dump(params, os.path.join(models_config_dir, 'models'))
    joblib.dump(is_scale_data, os.path.join(models_config_dir, 'is_scale_data'))
    return models_config_dir


def modeling(train_dir: str, config_fs_dir: str, models_config_dir:str, k: int, is_loo:bool,
             is_test:bool, testper_dir: str, test_dir: str):
    """
    runs all classification models defined in cfg_model and saves ROC, feature importance for each model
    :param train_dir:  directory where train data is saved
    :param config_fs_dir: directory where feature selection configuration is saved
    :param models_config_dir: directory where current models configuration is saved
    :param k: number of CV folds
    :param is_loo: True = leave one out CV, False = k fold CV
    :param is_test: True = hold out data for test, False = no hold out data for test
    :param testper_dir: directory with hold out percent of data for train and test (relevant only if is_test = True)
    :param test_dir: directory with hold out percent of data for test (relevant only if is_test = True)
    """
    features_df_ls: list = []
    # load classifiers configuration
    models = joblib.load(os.path.join(models_config_dir, 'models'))
    clfs = joblib.load(os.path.join(models_config_dir, 'clfs'))
    params = joblib.load(os.path.join(models_config_dir, 'params'))
    is_scale_data = joblib.load(os.path.join(models_config_dir, 'is_scale_data'))

    # loop over all classifiers
    for name, estimator in zip(models, clfs):
        # read original data features names
        df_features_importance: pd.DataFrame = pd.DataFrame(joblib.load(os.path.join(train_dir, 'headers')))
        df_features_importance = df_features_importance.rename(columns={0: 'feature'})
        print(name)
        classifier_dir: str = os.path.join(models_config_dir, name)
        if not os.path.exists(classifier_dir):
            os.makedirs(classifier_dir)
        clf = estimator
        grid = params[name]
        grid_auc = {}
        grid_gap = {}
        ind = 0
        for z in ParameterGrid(grid):
            print(z)
            hyperparam_dir: str = os.path.join(classifier_dir, 'grid' + str(ind))
            if not os.path.exists(hyperparam_dir):
                os.makedirs(hyperparam_dir)
            joblib.dump(z, os.path.join(hyperparam_dir, 'clf_params'))
            clf.set_params(**z)
            fig_ROC, ax = plt.subplots()
            all_test_accu = []
            all_train_accur = []
            if not is_loo:
                tprs = []
                aucs = []
                mean_fpr = np.linspace(0, 1, 100)
            else:
                all_y = []
                all_probs = []
            # model
            for fold_num in range(k):
                config_dir_fold: str = os.path.join(config_fs_dir, 'fold_' + str(fold_num))
                # read data
                if is_scale_data[name]:
                    data_train: pd.DataFrame = joblib.load(os.path.join(config_dir_fold,
                                                                    'df_train_scaled_selected'))
                    data_test: pd.DataFrame = joblib.load(os.path.join(config_dir_fold,
                                                                   'df_test_scaled_selected'))
                else:
                    data_train: pd.DataFrame = joblib.load(os.path.join(config_dir_fold,
                                                                    'df_data_train_selected'))
                    data_test: pd.DataFrame = joblib.load(os.path.join(config_dir_fold,
                                                                   'df_data_test_selected'))
                label_train: pd.DataFrame = joblib.load(os.path.join(config_dir_fold,
                                                                     'df_target_train_selected'))
                label_test: pd.DataFrame = joblib.load(os.path.join(config_dir_fold,
                                                                    'df_target_test'))
                df_features_fold: pd.DataFrame = pd.DataFrame(joblib.load(os.path.join(config_dir_fold,
                                                                                       'selected_features_names')))
                clf.fit(data_train, np.ravel(label_train))
                if not is_loo:
                    viz = plot_roc_curve(clf, data_test, np.ravel(label_test),
                                         name='ROC fold {}'.format(fold_num),
                                         alpha=0.3, lw=1, ax=ax)
                    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)
                    aucs.append(viz.roc_auc)
                    all_test_accu.append(clf.score(data_test, np.ravel(label_test)))
                    all_train_accur.append(clf.score(data_train, np.ravel(label_train)))
                else:
                    all_y.append(np.ravel(label_test))
                    all_probs.append(clf.fit(data_train, np.ravel(label_train)).predict_proba(data_test)[:, 1])
                    all_test_accu.append(clf.score(data_test, label_test))
                    all_train_accur.append(clf.score(data_train, np.ravel(label_train)))

                # create feature importance dataframe for each fold and merge it to one dataframe
                if hasattr(clf, 'feature_importances_'):
                    df_features_fold['importance']: pd.DataFrame = clf.feature_importances_
                elif hasattr(clf, 'coef_'):
                    df_features_fold['importance']: pd.DataFrame = pd.DataFrame(abs(clf.coef_)).transpose()
                if hasattr(clf, 'feature_importances_') | hasattr(clf, 'coef_'):
                    df_features_fold = df_features_fold.rename(columns={0: 'feature'})
                    df_merge: pd.DataFrame = df_features_importance.merge(df_features_fold, how='outer',
                                                                          left_on='feature', right_on='feature')
                    df_features_importance['fold' + str(fold_num)] = df_merge['importance']
            if not is_loo:
                ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                        label='Chance', alpha=.8)
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                roc_auc = auc(mean_fpr, mean_tpr)
                std_auc = np.std(aucs)
                ax.plot(mean_fpr, mean_tpr, color='b',
                        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_auc, std_auc),
                        lw=2, alpha=.8)
                std_tpr = np.std(tprs, axis=0)
                tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                                label=r'$\pm$ 1 std. dev.')
                ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
                       title="Receiver operating characteristic")
                ax.legend(loc="lower right")
                fig_ROC.savefig(os.path.join(hyperparam_dir, 'ROC_CV.png'))
                fig_accu = plt.figure()
                plt.scatter(all_test_accu, all_train_accur)
                plt.title('mean train accuracy: {:1.3f} mean test accuracy:{:1.3f}'
                          .format(np.mean(all_train_accur), np.mean(all_test_accu)))
                fig_accu.savefig(os.path.join(hyperparam_dir, 'Acuu_CV.png'))

            else:
                all_y = np.array(all_y)
                all_probs = np.array(all_probs)
                fpr, tpr, thresholds = roc_curve(all_y, all_probs)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, alpha=0.5, label='LOOCV ROC (AUC = %0.2f)' % roc_auc)
                ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance level', alpha=.8)
                ax.set(xlim=([-0.05, 1.05]), ylim=([-0.05, 1.05]),
                       xlabel='False Positive Rate', ylabel='True Positive Rate',
                       title="Receiver operating characteristic")
                ax.legend(loc="lower right")
                ax.grid()
                fig_ROC.savefig(os.path.join(hyperparam_dir, 'ROC_CV.png'))
                fig_accu = plt.figure()
                plt.scatter(all_test_accu, all_train_accur)
                plt.title('mean train accuracy: {:1.3f} mean test accuracy:{:1.3f}'
                          .format(np.mean(all_train_accur), np.mean(all_test_accu)))
                fig_accu.savefig(os.path.join(hyperparam_dir, 'Acuu_CV.png'))
            if hasattr(clf, 'feature_importances_') | hasattr(clf, 'coef_'):
                df_features_importance = df_features_importance.fillna(0)
                df_features_importance['sum_importance'] = df_features_importance.iloc[:, 1:].sum(axis=1)
                df_features_importance.sort_values(by=['sum_importance'], ascending=[False], inplace=True)
                df_features_importance.to_csv(os.path.join(hyperparam_dir, 'importance_CV.csv'))
                df_features_importance.sort_values(by=['sum_importance'], ascending=[True], inplace=True)
                mask = np.ravel(df_features_importance['sum_importance'] != 0)
                df_features_importance = df_features_importance.iloc[mask, :]
                df = pd.DataFrame({'feature': df_features_importance['feature'],
                                   'importance': np.ravel(df_features_importance['sum_importance'])})
                df.tail(20).plot.barh(x='feature', y='importance', figsize=(30, 18), fontsize=14)
                plt.savefig(os.path.join(hyperparam_dir, 'importance_CV.png'))
            grid_auc['grid' + str(ind)] = roc_auc
            grid_gap['grid' + str(ind)] = 1 - np.abs(np.mean(all_train_accur) - np.mean(all_test_accu))
            ind += 1
            plt.close('all')

        """create and save final model using all train set and test it with test set with best hyperparams
        """
        # read feature selection parameters dictionary
        fs_dict_param: int = joblib.load(os.path.join(config_fs_dir, 'fs_dict_param'))
        min_to_select: int = fs_dict_param['min_vars_to_select']
        top_univariate: int = fs_dict_param['top_univariate']
        folds_num_tree: int = fs_dict_param['folds_num_tree']
        top_multivariate: int = fs_dict_param['top_multivariate']
        if is_test:
            df_data_train: pd.DataFrame = joblib.load(os.path.join(testper_dir, 'train_set', 'df_data_train'))
            df_target_train: pd.DataFrame = joblib.load(os.path.join(testper_dir, 'train_set', 'df_target_train'))
            df_data_test: pd.DataFrame = joblib.load(os.path.join(test_dir, 'df_data_test'))
            df_target_test: pd.DataFrame = joblib.load(os.path.join(test_dir, 'df_target_test'))
        else:
            df_data_train: pd.DataFrame = joblib.load(os.path.join(train_dir, 'df_data'))
            df_target_train: pd.DataFrame = joblib.load(os.path.join(train_dir, 'df_target'))
        original_data_header = joblib.load(os.path.join(train_dir, 'headers'))
        # feature selection
        df_data_train_select, mask_array = remove_no_variance_features(df_data_train)
        data_header: pd.DataFrame = original_data_header[mask_array]
        df_data_train: pd.DataFrame = df_data_train.iloc[:, mask_array]
        if is_test:
            df_data_test: pd.DataFrame = df_data_test.iloc[:, mask_array]
        # test univariate relations
        df_fclassif: pd.DataFrame = uni_feature_selection(mask_array,
                                                         data_header, df_data_train, df_target_train, f_classif)
        df_mutual: pd.DataFrame = uni_feature_selection(mask_array,
                                                       data_header, df_data_train, df_target_train, mutual_info_classif)
        df_freg: pd.DataFrame = uni_feature_selection(mask_array,
                                                     data_header, df_data_train, df_target_train, f_regression)
        df_fclassif.to_csv(os.path.join(classifier_dir, 'df_fclassif.csv'))
        df_mutual.to_csv(os.path.join(classifier_dir, 'df_mutual.csv'))
        df_freg.to_csv(os.path.join(classifier_dir, 'df_freg.csv'))
        # select top features from univariate feature selection
        ufs_ls = []
        ufs_ls.append(df_fclassif)
        ufs_ls.append(df_mutual)
        ufs_ls.append(df_freg)
        df_select_rank = select_rank(ufs_ls, min_to_select)
        df_select_univariate = selected_features(df_select_rank, top_univariate)
        uni_mask: np.ndarray = df_select_univariate['feature_num'].to_numpy()
        df_data_train = df_data_train[uni_mask]
        if is_test:
            df_data_test = df_data_test[uni_mask]
        data_header = original_data_header[uni_mask]
        # examine multivariate relations
        # decision tree based method
        df_select_rank_tree: pd.DataFrame = tree_tune_pruning(uni_mask, data_header, df_data_train,
                                                              df_target_train, folds_num_tree, min_to_select)
        df_select_rank_tree = df_select_rank_tree.rename(columns=
                                                         {'sum_ind': 'sum_ind_tree', 'sum_rank': 'sum_rank_tree'})
        df_select_rank_tree.to_csv(os.path.join(classifier_dir, 'tree.csv'))
        # execute several SFS corresponding to the given classifiers
        means_train, stds_train, df_train_scaled = scale_data(df_data_train)
        sfs_clf_ls = []
        svm_linear = SVC(C=1.0, kernel='linear', random_state=0)
        gnb = GaussianNB()
        lr = LogisticRegression(random_state=0)
        sfs_clf_ls.append(svm_linear)
        sfs_clf_ls.append(gnb)
        sfs_clf_ls.append(lr)
        sfs_ls: list = sequential_forward_selection(df_train_scaled, data_header, df_target_train,
                                                    sfs_clf_ls, 'parsimonious')
        df_sfs = pd.merge(sfs_ls[-1][0], sfs_ls[-1][1], how='left', on='feature_num')
        df_sfs.sort_values(by=['sum_ind', 'sum_rank'], ascending=[False, True], inplace=True)
        df_sfs.to_csv(os.path.join(classifier_dir, 'fs.csv'))
        columns = ['feature_num', 'feature_x', 'sum_ind', 'sum_rank']
        df_sfs = df_sfs[columns]
        df_sfs = df_sfs.rename(columns={'feature_x': 'feature'})
        df_sfs = df_sfs.rename(columns={'sum_ind': 'sum_ind_sfs'})
        df_sfs = df_sfs.rename(columns={'sum_rank': 'sum_rank_sfs'})
        # combine multivariate feature selection
        df_select_rank_tree.sort_values(by=['feature_num'], ascending=[True], inplace=True)
        df_sfs.sort_values(by=['feature_num'], ascending=[True], inplace=True)
        dfs = [df_select_rank_tree, df_sfs]
        cols = ['feature_num', 'feature']
        df_multi: pd.DataFrame = pd.concat([d.set_index(cols) for d in dfs], axis=1).reset_index()
        df_multi['sum_ind'] = df_multi['sum_ind_tree'] + df_multi['sum_ind_sfs']
        df_multi['sum_rank'] = df_multi['sum_rank_tree'] + df_multi['sum_rank_sfs']
        df_multi.sort_values(by=['sum_ind', 'sum_rank'], ascending=[False, True], inplace=True)
        df_select_multi: pd.DataFrame = selected_features(df_multi, top_multivariate)
        df_select_multi.to_csv(os.path.join(classifier_dir, 'multivar_selection.csv'))
        # selecting top features from multivariate methods
        multi_mask: np.ndarray = df_select_multi['feature_num'].to_numpy()
        df_train_selected_features: pd.DataFrame = df_data_train[multi_mask]
        if is_test:
            df_test_selected_features: pd.DataFrame = df_data_test[multi_mask]
        if is_scale_data[name]:
            df_train_selected_features: pd.DataFrame = df_train_scaled[multi_mask]
            if is_test:
                df_test_selected: pd.DataFrame = (df_data_test - means_train) / stds_train
                df_test_selected_features: pd.DataFrame = df_test_selected[multi_mask]
        features_names = original_data_header[multi_mask]
        # fit classifier with all the train data with the best hyper params
        # the best params that maximize the weighted sum of AUC and maximize (1-gap) between train ant test accuracy
        weighted_average_grid = 0.5 * np.array(list(grid_auc.values())) + 0.5 * np.array(list(grid_gap.values()))
        best_grid_index = np.argmax(weighted_average_grid)
        best_grid = 'grid' + str(best_grid_index)
        src_dir = os.path.join(classifier_dir, best_grid)
        dst_dir = os.path.join(classifier_dir)
        for pngfile in glob.iglob(os.path.join(src_dir, "*.png")):
            shutil.copy(pngfile, dst_dir)
        clf_params = joblib.load(os.path.join(classifier_dir, best_grid, 'clf_params'))
        joblib.dump(clf_params, os.path.join(classifier_dir, 'clf_params'))
        print('the best hyper-parameters are:')
        print(clf_params)
        clf.set_params(**clf_params)
        clf.fit(df_train_selected_features, np.ravel(df_target_train))
        df_features_importance: pd.DataFrame = pd.DataFrame()
        df_features_importance['feature']: pd.DataFrame = features_names
        if hasattr(clf, 'feature_importances_'):
            df_features_importance['importance']: pd.DataFrame = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            df_features_importance['importance']: pd.DataFrame = pd.DataFrame(abs(clf.coef_)).transpose()
        if hasattr(clf, 'feature_importances_') | hasattr(clf, 'coef_'):
            df_features_importance.sort_values(by=['importance'], ascending=[True], inplace=True)
            df = pd.DataFrame({'feature': df_features_importance['feature'],
                               'importance': np.ravel(df_features_importance['importance'])})
            df.plot.barh(x='feature', y='importance', figsize=(30, 18), fontsize=14)
            df.to_csv(os.path.join(classifier_dir, 'features_importance.csv'))
            plt.savefig(os.path.join(classifier_dir, 'features_importance.png'))
        joblib.dump(clf, os.path.join(classifier_dir, 'final_model'))
        plt.close()
        # test model performance using test_set
        if is_test:
            y_pred = clf.predict(df_test_selected_features)
            acc_score = accuracy_score(np.ravel(df_target_test), y_pred)
            print(acc_score)
            conf_mat = confusion_matrix(np.ravel(df_target_test), y_pred)
            print(conf_mat)
            joblib.dump(acc_score, os.path.join(classifier_dir, 'accuracy_score'))
            joblib.dump(conf_mat, os.path.join(classifier_dir, 'confusion_matrix'))
            target_names = ['LR under 20', 'LR above 80']
            labels_order = [0, 1]
            plt.figure()
            sns.heatmap(
                confusion_matrix(df_target_test,
                                 y_pred, labels=labels_order), annot=True, fmt='d', cmap="binary", linewidths=.5)
            plt.savefig(os.path.join(classifier_dir, 'confusion matrix.png'))

            report = classification_report(df_target_test, y_pred, labels=labels_order,
                                           target_names=labels_order, output_dict=True)
            plt.figure()
            plt.title('confidence classification report')
            sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True,
                        cmap="Greys", linewidths=.5, vmin=0,
                        vmax=0.8)
            plt.savefig(os.path.join(classifier_dir, 'report.png'))


# data directory
data_dir: str = r'O:\Inbar\Beat ML\BeatReports\PPMI_format\processed_data'
file_name: str = 'LRunder20_LRabove80_matched.xlsx'
sheet_name: str = 'Sheet1'
# directory to save processed data and splits k-folds
datasets_dir: str = r'O:\Inbar\Beat ML\BeatReports\PPMI_format\Datasets\LRunder20_LRabove80_matched'
# read data as pd.DataFrame
df_beat_gait: pd.DataFrame = pd.read_excel(
    os.path.join(data_dir, file_name), sheet_name=sheet_name)
gait_data: pd.DataFrame = df_beat_gait.values
headers: pd.Index = df_beat_gait.columns[:-1]
# validation set is/no and percentage
is_test: bool = False
test_percentage: float = 0.2
# CV number of folds/leave one out
is_loo: bool = True
# number of folds if not loo
if not is_loo:
    k: int = 20
else:
    k: int = len(gait_data)
# balance data yes/no
is_balance: bool = False

if __name__ == '__main__':
    train_dir, testper_dir, test_dir = etl(gait_data, headers, datasets_dir, is_test,
                              test_percentage, is_loo, k, is_balance)
    config_fs_dir: str = cfg_fs(train_dir)
    # lasso_feature_selection(train_dir, config_fs_dir, k)
    feature_selection(train_dir, config_fs_dir, k)
    models_config_dir: str = cfg_model(train_dir, config_fs_dir)
    modeling(train_dir, config_fs_dir, models_config_dir, k, is_loo,
             is_test, testper_dir, test_dir)



