import subprocess

participant_files = {
    "p01": {
        "1hz": "p01_1hz_RSVP_Calibration_Thu_12_Aug_2021_14hr02min19sec_-0700",
        "4hz": "p01_4hz_RSVP_Calibration_Thu_12_Aug_2021_13hr46min18sec_-0700",
    },
    "p02": {
        "1hz": "p02_1hz_RSVP_Calibration_Wed_04_Aug_2021_14hr44min32sec_-0700",
        "4hz": "p02_4hz_RSVP_Calibration_Wed_04_Aug_2021_15hr13min03sec_-0700",
    },
    "p03": {
        "1hz": "p03_1hz_RSVP_Calibration_Fri_03_Sep_2021_09hr56min33sec_-0700",
        "4hz": "p03_4hz_RSVP_Calibration_Fri_03_Sep_2021_09hr40min46sec_-0700",
    },
    "p04": {
        "1hz": "p04_1hz_RSVP_Calibration_Tue_07_Sep_2021_14hr39min30sec_-0700",
        "4hz": "p04_4hz_RSVP_Calibration_Tue_07_Sep_2021_15hr08min04sec_-0700",
    },
    "p05": {
        "1hz": "p05_1hz_RSVP_Calibration_Fri_20_Aug_2021_08hr20min50sec_-0700",
        "4hz": "p05_4hz_RSVP_Calibration_Fri_20_Aug_2021_08hr52min46sec_-0700",
    },
    "p06": {
        "1hz": "p06_1hz_RSVP_Calibration_Tue_24_Aug_2021_16hr56min47sec_-0700",
        "4hz": "p06_4hz_RSVP_Calibration_Tue_24_Aug_2021_17hr25min03sec_-0700",
    },
    "p07": {
        "1hz": "p07_1hz_RSVP_Calibration_Tue_07_Sep_2021_10hr53min23sec_-0700",
        "4hz": "p07_4hz_RSVP_Calibration_Tue_07_Sep_2021_11hr20min34sec_-0700",
    },
    "p08": {
        "1hz": "p08_1hz_RSVP_Calibration_Wed_18_Aug_2021_11hr19min28sec_-0700",
        "4hz": "p08_4hz_RSVP_Calibration_Wed_18_Aug_2021_11hr45min10sec_-0700",
    },
    "p09": {
        "1hz": "p09_1hz_RSVP_Calibration_Thu_12_Aug_2021_10hr00min36sec_-0700",
        "4hz": "p09_4hz_RSVP_Calibration_Thu_12_Aug_2021_09hr48min11sec_-0700",
    },
    "p10": {
        "1hz": "p10_1hz_RSVP_Calibration_Fri_13_Aug_2021_15hr59min40sec_-0700",
        "4hz": "p10_4hz_RSVP_Calibration_Fri_13_Aug_2021_15hr44min48sec_-0700",
    },
    "p11": {
        "1hz": "p11_1hz_RSVP_Calibration_Wed_25_Aug_2021_17hr58min11sec_-0700",
        "4hz": "p11_4hz_RSVP_Calibration_Wed_25_Aug_2021_17hr42min11sec_-0700",
    },
    "p12": {
        "1hz": "p12_1hz_RSVP_Calibration_Mon_16_Aug_2021_11hr08min42sec_-0700",
        "4hz": "p12_4hz_RSVP_Calibration_Mon_16_Aug_2021_10hr52min29sec_-0700",
    },
}

participant_freqs = {
    "p01": {"1hz": 10, "4hz": 10},
    "p02": {"1hz": 10.5, "4hz": 10.5},
    "p03": {"1hz": 12, "4hz": 12},
    "p04": {"1hz": 10, "4hz": 10},
    "p05": {"1hz": 11, "4hz": 11.5},
    "p06": {"1hz": 11, "4hz": 11},
    "p07": {"1hz": 10, "4hz": 10},
    "p08": {"1hz": 10.5, "4hz": 10.5},
    "p09": {"1hz": 9, "4hz": 9},
    "p10": {"1hz": 10.5, "4hz": 10.5},
    "p11": {"1hz": 9, "4hz": 10},
    "p12": {"1hz": 9, "4hz": 9},
}

for p in participant_files.keys():
    for hz in ["1hz", "4hz"]:
        file = participant_files[p][hz]
        freq = participant_freqs[p][hz]

        cmd = f"xvfb-run --auto-servernum python alpha-experiment.py --input data/bcipy_recordings/{p}/{file} --output results/{p}/{hz}.cwt_freq{freq} --freq {freq}"
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
