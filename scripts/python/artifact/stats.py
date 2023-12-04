

def paired_ttest(df, col1, col2):
    from scipy import stats
    return stats.ttest_rel(df[col1], df[col2])

def one_way_anova(df, cols):
    from scipy import stats
    return stats.f_oneway(*[df[col] for col in cols])

def permutation_ttest_mne(data, perm_count=1000, threshold=2.0, tail=0):
    from mne.stats import permutation_cluster_1samp_test

    # data = np.random.randn(2, 16, 1000)
    # data[1, :, :] += 1
    return permutation_cluster_1samp_test(data, n_permutations=perm_count, threshold=threshold, tail=tail)

if __name__ == "__main__":
    pass