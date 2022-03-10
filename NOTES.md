Trying different model changes


- "Safest" change: 
    - use `np.exp(log1 - log2)` instead of `np.exp(log1) - np.exp(log2)` and use `np.clip(likelihood_ratios, 1e-2, 1e2)`
    ```shell
    python bcipy/signal/model/offline_analysis.py -d data/IP08/IP08_RSVP_Calibration_Mon_07_Mar_2022_11hr06min08sec_-0800 -p data/IP08/IP08_RSVP_Calibration_Mon_07_Mar_2022_11hr06min08sec_-0800/parameters.json 
    ```
    Note that clipping values differently will not affect AUC (which is computed just from PCA+RDA steps), but will affect the overall updates given.

    Runtime
- 

