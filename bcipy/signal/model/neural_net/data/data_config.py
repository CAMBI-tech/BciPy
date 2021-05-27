# NOTE - the numbers here include preprocessing steps as performed in VisualERPDatasets

data_config = {
    "gtec": {
        "sample_rate_hz": 128,
        "n_channels": 16,
        "sequences": {
            "length": 397,
            "length_tol": 5,  # data length may be +/- this amount
            "n_classes": 14 + 1,
        },
        "trials": {
            "length": 64,
            "length_tol": 3,  # data length may be +/- this amount
            "n_classes": 2,
        },
    },
    "dsi": {
        "sample_rate_hz": 150,
        "n_channels": 20,
        "sequences": {
            "length": 465,
            "length_tol": 5,  # data length may be +/- this amount
            "n_classes": 14 + 1,
        },
        "trials": {
            "length": 75,
            "length_tol": 3,  # data length may be +/- this amount
            "n_classes": 2,
        },
    },
}
