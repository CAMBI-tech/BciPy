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
- fix logging during simulation
- configure output of xlsx (not needed for the analysis / running out of disk space)
- fix metrics visualization when running multiple simulations (currently overwriting the same file?)
- get phrases from LM team, set in the phrase.json or via the PHRASES variable below
- set and verify LM parameters
- write a top level processing / compare script for the outputs. ttest / visualizations
"""
from pathlib import Path
from bcipy.io.load import load_json_parameters, load_signal_model
from bcipy.simulator.task.task_factory import TaskFactory
from bcipy.simulator.task.task_runner import TaskRunner, init_simulation_dir
import bcipy.simulator.util.metrics as metrics


PHRASES = [
    ("WELCOME_HOME", 0), 
    ("HELLO_WORLD", 6),
    ("BCI_IS_COOL", 0),
    ("DANIEL_IS_AWESOME", 0),
    ("SIMULATIONS_ARE_HARD", 12),
]
LANGUAGE_MODELS = ["UNIFORM", "KENLM"]
MODE = "RSVP"
DATA_PATTERN = f"{MODE}_Copy_Phrase"
RUN_COUNT = 25
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
    parameters["task_text"] = phrase
    parameters["spelled_letters_count"] = starting_index
    parameters["lang_model_type"] = language_model

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
    sim_dir = init_simulation_dir(save_location=Path(OUTPUT_DIR), prefix=f"{user}/{phrase}/{language_model}/{user}_{phrase}_{language_model}_")
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
    # load experimental directory. This will have users/run/data
    data_dir = Path(load_experimental_data())

    # loop over all the users
    for user in data_dir.iterdir():
        if user.is_dir():
            for phrase, starting_index in PHRASES:
                for language_model in LANGUAGE_MODELS:
                    run_simulation(user, user.name, phrase, starting_index, language_model)