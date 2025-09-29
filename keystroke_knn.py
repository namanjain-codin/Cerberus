import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit


def _build_hist_feature_tables(train_df: pd.DataFrame, test_df: pd.DataFrame, num_bins: int = 10):
    # Derive PPD, RPD, HD features as in the notebook
    train = train_df.copy()
    test = test_df.copy()
    for i in range(1, 13):
        train[f'PPD-{i}'] = train[f'press-{i}'] - train[f'press-{i-1}']
        train[f'RPD-{i}'] = train[f'release-{i}'] - train[f'press-{i-1}']
        test[f'PPD-{i}'] = test[f'press-{i}'] - test[f'press-{i-1}']
        test[f'RPD-{i}'] = test[f'release-{i}'] - test[f'press-{i-1}']
    for i in range(13):
        train[f'HD-{i}'] = train[f'release-{i}'] - train[f'press-{i}']
        test[f'HD-{i}'] = test[f'release-{i}'] - test[f'press-{i}']

    # Long-format for HD, PPD, RPD with corresponding press times
    drop_cols_hd = [f'PPD-{i}' for i in range(1, 13)] + [f'RPD-{i}' for i in range(1, 13)] + [f'release-{i}' for i in range(13)]
    t_hd = train.drop(columns=drop_cols_hd).copy()
    t_hd['id'] = t_hd.index
    t_hd = pd.wide_to_long(t_hd, ['press-', 'HD-'], i='id', j='key_no').sort_values(by=['user', 'id', 'key_no'])

    drop_cols_ppd = [f'HD-{i}' for i in range(13)] + [f'RPD-{i}' for i in range(1, 13)] + [f'release-{i}' for i in range(13)] + ['press-0']
    t_ppd = train.drop(columns=drop_cols_ppd).copy()
    t_ppd['id'] = t_ppd.index
    t_ppd = pd.wide_to_long(t_ppd, ['press-', 'PPD-'], i='id', j='key_no').sort_values(by=['user', 'id', 'key_no'])

    drop_cols_rpd = [f'HD-{i}' for i in range(13)] + [f'PPD-{i}' for i in range(1, 13)] + [f'release-{i}' for i in range(13)] + ['press-0']
    t_rpd = train.drop(columns=drop_cols_rpd).copy()
    t_rpd['id'] = t_rpd.index
    t_rpd = pd.wide_to_long(t_rpd, ['press-', 'RPD-'], i='id', j='key_no').sort_values(by=['user', 'id', 'key_no'])

    train_combined = t_hd.join(t_rpd.drop(columns=['user', 'press-']), rsuffix='RPD_').join(
        t_ppd.drop(columns=['user', 'press-']), rsuffix='PPD_'
    )

    # Test
    thd = test.drop(columns=drop_cols_hd).copy()
    thd['id'] = thd.index
    thd = pd.wide_to_long(thd, ['press-', 'HD-'], i='id', j='key_no').sort_values(by=['id', 'key_no'])

    tppd = test.drop(columns=drop_cols_ppd).copy()
    tppd['id'] = tppd.index
    tppd = pd.wide_to_long(tppd, ['press-', 'PPD-'], i='id', j='key_no').sort_values(by=['id', 'key_no'])

    trpd = test.drop(columns=drop_cols_rpd).copy()
    trpd['id'] = trpd.index
    trpd = pd.wide_to_long(trpd, ['press-', 'RPD-'], i='id', j='key_no').sort_values(by=['id', 'key_no'])

    test_combined = thd.join(trpd.drop(columns=['press-']), rsuffix='RPD_').join(
        tppd.drop(columns=['press-']), rsuffix='PPD_'
    )

    # Binning
    labels = list(range(num_bins))
    train_combined['HDEnc'], HDBins = pd.qcut(train_combined['HD-'], retbins=True, labels=labels, q=num_bins)
    train_combined['PPDEnc'], RPDBins = pd.qcut(train_combined['PPD-'], retbins=True, labels=labels, q=num_bins)
    train_combined['RPDEnc'], PPDBins = pd.qcut(train_combined['RPD-'], retbins=True, labels=labels, q=num_bins)

    train_combined['HDEnc'] = train_combined['HDEnc'].astype(str).replace('nan', -1).astype(float)
    train_combined['PPDEnc'] = train_combined['PPDEnc'].astype(str).replace('nan', -1).astype(float)
    train_combined['RPDEnc'] = train_combined['RPDEnc'].astype(str).replace('nan', -1).astype(float)

    test_combined['HDEnc'] = pd.cut(test_combined['HD-'], labels=labels, bins=HDBins)
    test_combined['PPDEnc'] = pd.cut(test_combined['PPD-'], labels=labels, bins=RPDBins)
    test_combined['RPDEnc'] = pd.cut(test_combined['RPD-'], labels=labels, bins=PPDBins)
    test_combined['HDEnc'] = test_combined['HDEnc'].astype(str).replace('nan', -1).astype(float)
    test_combined['PPDEnc'] = test_combined['PPDEnc'].astype(str).replace('nan', -1).astype(float)
    test_combined['RPDEnc'] = test_combined['RPDEnc'].astype(str).replace('nan', -1).astype(float)

    # User-level average bin per keystroke signature
    train_avg = pd.DataFrame({
        'HD': train_combined.reset_index().groupby(['user', 'key_no'])['HDEnc'].mean(),
        'PPD': train_combined.reset_index().groupby(['user', 'key_no'])['PPDEnc'].mean(),
        'RPD': train_combined.reset_index().groupby(['user', 'key_no'])['RPDEnc'].mean(),
    })
    train_user_props = pd.DataFrame({
        'HD': train_avg.reset_index().groupby('user')['HD'].apply(np.array),
        'PPD': train_avg.reset_index().groupby('user')['PPD'].apply(np.array),
        'RPD': train_avg.reset_index().groupby('user')['RPD'].apply(np.array),
    })
    train_user_props = pd.DataFrame(train_user_props.HD.tolist(), index=train_user_props.index).add_prefix('HD_').join(
        pd.DataFrame(train_user_props.PPD.tolist(), index=train_user_props.index).add_prefix('PPD_')
    ).join(
        pd.DataFrame(train_user_props.RPD.tolist(), index=train_user_props.index).add_prefix('RPD_')
    )

    # All-samples table for CV
    t_all = pd.DataFrame({
        'HD': train_combined.reset_index().groupby(['user', 'id'])['HDEnc'].apply(np.array),
        'PPD': train_combined.reset_index().groupby(['user', 'id'])['PPDEnc'].apply(np.array),
        'RPD': train_combined.reset_index().groupby(['user', 'id'])['RPDEnc'].apply(np.array),
    })
    train_all_samples = pd.DataFrame(t_all.HD.tolist(), index=t_all.index).add_prefix('HD_').join(
        pd.DataFrame(t_all.PPD.tolist(), index=t_all.index).add_prefix('PPD_')
    ).join(
        pd.DataFrame(t_all.RPD.tolist(), index=t_all.index).add_prefix('RPD_')
    ).reset_index().set_index('user').drop(columns=['id'])

    # Test user props (by id)
    test_avg = pd.DataFrame({
        'HD': test_combined.reset_index().groupby(['id', 'key_no'])['HDEnc'].mean(),
        'PPD': test_combined.reset_index().groupby(['id', 'key_no'])['PPDEnc'].mean(),
        'RPD': test_combined.reset_index().groupby(['id', 'key_no'])['RPDEnc'].mean(),
    })
    test_user_props = pd.DataFrame({
        'HD': test_avg.reset_index().groupby('id')['HD'].apply(np.array),
        'PPD': test_avg.reset_index().groupby('id')['PPD'].apply(np.array),
        'RPD': test_avg.reset_index().groupby('id')['RPD'].apply(np.array),
    })
    test_user_props = pd.DataFrame(test_user_props.HD.tolist(), index=test_user_props.index).add_prefix('HD_').join(
        pd.DataFrame(test_user_props.PPD.tolist(), index=test_user_props.index).add_prefix('PPD_')
    ).join(
        pd.DataFrame(test_user_props.RPD.tolist(), index=test_user_props.index).add_prefix('RPD_')
    )

    # All-samples for test
    ta_all = pd.DataFrame({
        'HD': test_combined.reset_index().groupby(['id'])['HDEnc'].apply(np.array),
        'PPD': test_combined.reset_index().groupby(['id'])['PPDEnc'].apply(np.array),
        'RPD': test_combined.reset_index().groupby(['id'])['RPDEnc'].apply(np.array),
    })
    test_all_samples = pd.DataFrame(ta_all.HD.tolist(), index=ta_all.index).add_prefix('HD_').join(
        pd.DataFrame(ta_all.PPD.tolist(), index=ta_all.index).add_prefix('PPD_')
    ).join(
        pd.DataFrame(ta_all.RPD.tolist(), index=ta_all.index).add_prefix('RPD_')
    )

    return train_user_props, train_all_samples, test_user_props, test_all_samples


def knn_cross_val_accuracy(train_all_samples: pd.DataFrame, n_splits: int = 5, test_size: float = 0.2, n_neighbors: int = 1) -> float:
    trainX = train_all_samples.reset_index().drop(columns=['user'])
    trainY = train_all_samples.index
    knn = KNeighborsClassifier(n_neighbors)
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
    accs = []
    for train_index, test_index in sss.split(trainX, trainY):
        knn.fit(trainX.loc[train_index], trainY[train_index])
        acc = accuracy_score(knn.predict(trainX.loc[test_index]), trainY[test_index])
        accs.append(acc)
    return float(sum(accs) / len(accs)) if accs else 0.0


def run_keystroke_knn(train_csv: str, test_csv: str, num_bins: int = 10, n_neighbors: int = 1):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    train_user_props, train_all_samples, test_user_props, test_all_samples = _build_hist_feature_tables(train_df, test_df, num_bins=num_bins)

    cv_acc = knn_cross_val_accuracy(train_all_samples, n_neighbors=n_neighbors)

    # Fit final model on all train samples and predict for test (optional)
    trainX = train_all_samples.reset_index().drop(columns=['user'])
    trainY = train_all_samples.index
    testX = test_all_samples.reset_index().drop(columns=['id'])
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(trainX, trainY)
    preds = knn.predict(testX)

    return {
        'cv_accuracy': float(cv_acc),
        'predictions': preds.tolist(),
        'train_users': sorted(list(trainY.unique())),
    }


