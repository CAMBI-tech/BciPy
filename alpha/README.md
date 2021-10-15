# Overall workflow
    raw data:
        7 channel, exclude F7 and Fcz, include Pz, Oz, Po7, Po8
        -1250ms to 1250ms around stim

    pick the 4 channels of interest

    CWT:
        give a parameter for the single wavelet frequency/scale to keep at the end (default ~10)
        consider overlap with SSVEP response (or its harmonics)

    -600ms to -100ms window for "baseline"
        get mean, stdev

    for each point in [300, 800]:
        z-score each sample point using the mean & stdev above

    pca/rda/kde

# TODO
- Hyperparams:
    - when converting from complex, square or not?
    - for z-scoring, per channel, or per trial, etc
    - wavelet and its params
    - Set CWT parameters so that data matches BrainVision - try plotting the full (-1250, 1250) trials, averaged for target and nontarget

- Include the PcaRdaKdeModel and RdaKdeModel
- fine-tune the choice of models

- Can try export data from BrainVision and train model


# Notes Oct 14

RDA/KDE model:
- need same metrics to compare
- use all 7 or only same 4 channels as with others


better tuning:
- 1d search first item, using default for 2nd value
- 1d search second item, using tuned 1st value
