from scipy import stats
from mne.stats import permutation_t_test, permutation_cluster_1samp_test
import pandas as pd
import numpy as np

from bcipy.helpers.load import load_experimental_data

def paired_ttest(df, col1, col2):
    return stats.ttest_rel(df[col1], df[col2])

def one_way_anova(df, cols):
    return stats.f_oneway(*[df[col] for col in cols])

def permutation_ttest_mne(data, perm_count=1000, threshold=2.0, tail=0):
    # data = np.random.randn(2, 16, 1000)
    # data[1, :, :] += 1
    return permutation_t_test(data, n_permutations=perm_count, tail=tail)

if __name__ == "__main__":
    
    path = load_experimental_data()
    filename = 'of_cf_flat_all_models.csv'
    df = pd.read_csv(f'{path}/{filename}')
    de_f = len(df) - 1  # degrees of freedom
    alpha = 0.05
    export_stats = {}
    t_thresh = stats.distributions.t.ppf(1 - alpha / 2, df=de_f)
    models = ['LR', 'SVM', 'NN', 'PRK', 'RF', 'LDA']


    of_mcc_cols = [col for col in df.columns if 'OF' in col and 'mcc' in col and 'std' not in col and 'RF' not in col]
    of_mcc_df = df[of_mcc_cols]
    cf_mcc_cols = [col for col in df.columns if 'WD' in col and 'mcc' in col and 'std' not in col and 'RF' not in col]
    cf_mcc_df = df[cf_mcc_cols]

    of_cols = of_mcc_cols + cf_mcc_cols
    all_mcc_df = df[of_cols]
    # result_anova = one_way_anova(df, cols)

    export_stats['MCC'] = {}
    for model in models:
        of_model_mcc_col = [col for col in of_mcc_df.columns if model in col]
        of_model_mcc_df = of_mcc_df[of_model_mcc_col]
        cf_model_mcc_col = [col for col in cf_mcc_df.columns if model in col]
        cf_model_mcc_df = cf_mcc_df[cf_model_mcc_col]
        # print(f'{model} OF vs CF MCC: {paired_ttest(df, of_model_mcc_col, cf_model_mcc_col)}')

        # find the difference between OF and CF MCC for each model
        diff = of_model_mcc_df.to_numpy() - cf_model_mcc_df.to_numpy()
        # print(f'{model} OF vs CF MCC: {permutation_ttest_mne(diff)}')
        # result = permutation_ttest_mne([df[col] for col in cols])
        permutations = 50000
        t_obs, clust, p_value, h_s = permutation_cluster_1samp_test(diff, n_permutations=permutations, threshold=t_thresh, adjacency=None, n_jobs=None, tail=0)
        print(f'{model} OF vs CF MCC: {p_value}')
        export_stats['MCC'][model] = {
            'of_mean': of_model_mcc_df.mean(),
            'of_std': of_model_mcc_df.std(),
            'cf_mean': cf_model_mcc_df.mean(),
            'cf_std': cf_model_mcc_df.std(),
            'p_value': p_value,
        }
    
    of_ba_cols = [col for col in df.columns if 'OF' in col and 'ba' in col and 'std' not in col and 'RF' not in col]
    of_ba_df = df[of_ba_cols]
    cf_ba_cols = [col for col in df.columns if 'WD' in col and 'ba' in col and 'std' not in col and 'RF' not in col]
    cf_ba_df = df[cf_ba_cols]

    of_cols = of_ba_cols + cf_ba_cols
    all_ba_df = df[of_cols]

    export_stats['BA'] = {}
    for model in models:
        of_model_ba_col = [col for col in of_ba_df.columns if model in col]
        of_model_ba_df = of_ba_df[of_model_ba_col]
        cf_model_ba_col = [col for col in cf_ba_df.columns if model in col]
        cf_model_ba_df = cf_ba_df[cf_model_ba_col]
        # print(f'{model} OF vs CF ba: {paired_ttest(df, of_model_ba_col, cf_model_ba_col)}')

        # find the difference between OF and CF ba for each model
        diff = of_model_ba_df.to_numpy() - cf_model_ba_df.to_numpy()
        # print(f'{model} OF vs CF ba: {permutation_ttest_mne(diff)}')
        # result = permutation_ttest_mne([df[col] for col in cols])
        permutations = 50000
        t_obs, clust, p_value, h_s = permutation_cluster_1samp_test(diff, n_permutations=permutations, threshold=t_thresh, adjacency=None, n_jobs=None, tail=0)
        print(f'{model} OF vs CF ba: {p_value}')
        export_stats['BA'][model] = {
            'of_mean': of_model_ba_df.mean(),
            'of_std': of_model_ba_df.std(),
            'cf_mean': cf_model_ba_df.mean(),
            'cf_std': cf_model_ba_df.std(),
            'p_value': p_value,
        }

    # get the average of all models
    of_all_mcc_df = of_mcc_df.mean(axis=1)
    cf_all_mcc_df = cf_mcc_df.mean(axis=1)
    of_all_ba_df = of_ba_df.mean(axis=1)
    cf_all_ba_df = cf_ba_df.mean(axis=1)

    # obvs, p_values, h_stats = permutation_t_test(all_mcc_df, n_permutations=permutations, tail=0)
    breakpoint()

    # find the rows with the largest difference between OF and CF in SVM mcc
    df['diff'] = df['OF_NAR_IIR_SVM_mcc'] - df['WD_NAR_IIR_SVM_mcc']
    df['diff'] = df['OF_NAR_IIR_PRK_mcc'] - df['WD_NAR_IIR_PRK_mcc']