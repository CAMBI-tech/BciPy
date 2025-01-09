"""
This script will loop through all users in the experimental data directory and run simulations using the copy phrase data. The primary outcome is to observe how
the task performance changes with different language models across a variety of phrases. This script will output results to a directory defined by OUTPUT_DIR.

The results will be saved in a directory with the following format:

```text

output_dir/
    user1/
        phrase1/
            language_model1/
                user1_phrase1_language_model1_SIM_datetime/
                    run_1/
                        run_1.log
                        session.json
                        session.xlsx
                        triggers.txt
                    run_2/
                        ...
                    summary_data.csv
                    metrics.png
        phrase2/
            language_model1/
                user1_phrase2_language_model1_SIM_datetime/
                    ...
    user2/
        ...

```

The summary data at the simulation run level (summary_data.json) will be the the data used to compare across users and phrases.

TODO: 
- Integrate LLM model and add to the analysis. Waiting for push from DG 01/08/2025. Then I will integrate the branch into BANFF.
"""
from pathlib import Path
from bcipy.io.load import load_json_parameters
from bcipy.simulator.task.task_factory import TaskFactory
from bcipy.simulator.task.task_runner import TaskRunner, init_simulation_dir
import bcipy.simulator.util.metrics as metrics


PHRASES = [
    # EASY PHRASES
    ("I_LOVE_YOU", 0), 
    ("SEE_YOU_LATER", 0),
    # ("I_NEED_SOME_HELP", 0),
    # ("MY_DOG_IS_GOOD", 0),
    # ("HOW_ARE_YOU", 0),
    # ("SHOW_ME_PLEASE", 0),
    # ("THIS_IS_GREAT", 0),
    # ("I_LIKE_BLUE_CAKE", 0),
    # ("GIVE_IT_BACK", 0),
    # ("BE_HOME_SOON", 0),

    # HARD PHRASES
    ("CAN_WE_GO_WED", 0),
    ("WHERE_IN_CANCUN", 0),
    # ("IDC_WHAT_YOU_GET", 0),
    # ("HOW_R_U_TODAY", 0),
    # ("IDK_WHEN", 0),
    # ("GOT_COOL_NEW_TECH", 0),
    # ("HOW_IS_HE_DOC", 0),
    # ("I_LIKE_REO_SPEEDWAGON", 0),
    # ("GONNA_BE_LIT_FAM", 0),
    # ("DOING_HW", 0),

]
LANGUAGE_MODELS = ["UNIFORM", "KENLM", "CAUSAL"] # Add LLM to this list
MODE = "RSVP"
DATA_PATTERN = f"{MODE}_Copy_Phrase"
RUN_COUNT = 5
OUTPUT_DIR = "/Users/scitab/Desktop/sim_output"


def run_simulation(data_dir: Path, user: str, phrase: str, starting_index: int, language_model: str) -> None:
    """Run a simulation for the given user, phrase, and language model"""
    print(f"Running simulation for {user} with phrase {phrase} and language model {language_model}")

    model_path = None
    params_path = None
    mode_calibration_dir = None
    # load the parameters and model for the simulation from the mode calibration directory
    for file in data_dir.iterdir():
        if file.is_dir() and MODE in file.name and "Calibration" in file.name:
            mode_calibration_dir = file
            # search for a pkl file
            for pkl_file in mode_calibration_dir.iterdir():
                if pkl_file.is_file() and pkl_file.suffix == ".pkl":
                    model_path = pkl_file
                    break
    if not model_path:
        raise FileNotFoundError(f"Could not find a model file in {mode_calibration_dir}")
    # load the parameters
    params_path = mode_calibration_dir / "parameters.json"
    parameters = load_json_parameters(params_path, value_cast=True)

    # update the parameters with the new phrase, starting index, and language model
    phrase_length = len(phrase) - starting_index
    print(f"Processing {user}:{phrase}:{language_model} of phrase length: {phrase_length}")
    parameters["task_text"] = phrase
    parameters["spelled_letters_count"] = starting_index
    parameters["lang_model_type"] = language_model

    parameters["max_inq_len"] = phrase_length * 8
    parameters["max_selections"] = phrase_length * 2 # This should be 2 * the length of the phrase to type
    parameters["min_inq_per_series"] = 1
    parameters["max_inq_per_series"] = 8
    parameters["backspace_always_shown"] = True
    parameters["lm_backspace_prob"] = 0.03571
    parameters["max_minutes"] = 120 # This is not used in the simulation. But we set it high to avoid any issues.
    parameters["max_incorrect"] = int(phrase_length / 2) # This should be half the length of the phrase to type

    # get the correct list of source directories from the data directory
    source_dirs = [str(file) for file in data_dir.iterdir() if file.is_dir() and DATA_PATTERN in file.name]
    if not source_dirs:
        raise FileNotFoundError(f"Could not find a data directory for {user}")
    
    print(f"Running simulation for {user} with phrase {phrase} and language model {language_model} using {len(source_dirs)} data directories")

    # create the task factory
    task_factory = TaskFactory(params_path=params_path,
                               source_dirs=source_dirs,
                               signal_model_paths=[str(model_path)],
                               parameters=parameters)
    # create the simulation directory
    # model_auc = model_path.stem.split("_")[-1]
    sim_dir = init_simulation_dir(
        save_location=Path(OUTPUT_DIR),
        prefix=f"{user}/{language_model}/{phrase}/{user}_{phrase}_{language_model}_")
    # create the task runner
    runner = TaskRunner(save_dir=sim_dir,
                        task_factory=task_factory,
                        runs=RUN_COUNT)
    # run the simulation
    runner.run()
    metrics.report(sim_dir)



if __name__ == "__main__":
    from bcipy.io.load import load_experimental_data
    from pathlib import Path
    from tqdm import tqdm
    # load experimental directory. This will have users/run/data
    data_dir = Path(load_experimental_data())

    progress_bar = tqdm(
        Path(data_dir).iterdir(),
        total=len(list(Path(data_dir).iterdir())),
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [est. {remaining}][ela. {elapsed}]\n",
        colour='MAGENTA')
    # loop over all the users
    for user in progress_bar:
        if user.is_dir():
            progress_bar.set_description(f"Processing {user.name}")
            for phrase, starting_index in PHRASES:
                for language_model in LANGUAGE_MODELS:
                    run_simulation(user, user.name, phrase, starting_index, language_model)

    progress_bar.close()