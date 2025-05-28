"""
This script will loop through all users in the experimental data directory and run simulations using copy phrase data. The primary outcome is to observe how
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

The summary data at the simulation run level (summary_data.json) will be the the data used to compare across users, phrases, language models.

"""
from pathlib import Path
from typing import Optional

import bcipy.simulator.util.metrics as metrics
from bcipy.io.load import load_json_parameters
from bcipy.simulator.task.task_factory import TaskFactory
from bcipy.simulator.task.task_runner import TaskRunner, init_simulation_dir

# Define the phrases, starting indeces, and language models to use for the simulation

# Add phrases and starting indeces to this list
PHRASES = [
    ("I_LOVE_YOU", 0),
    ("SEE_YOU_LATER", 4),
    ("I_NEED_SOME_HELP", 7),
]
# Add registered BciPy language models by name here
LANGUAGE_MODELS = ["UNIFORM"]

# Number of runs for each simulation
RUN_COUNT = 2

# add a custom output path for simulation output
OUTPUT_DIR = "./"

# Filter data by file name that will be loaded and used for the simulation
MODE = "RSVP"
DATA_PATTERN = f"{MODE}_Copy_Phrase"


def run_simulation(
        data_dir: Path,
        user: str,
        phrase: str,
        starting_index: int,
        language_model: str,
        signal_model_path: Optional[str] = None) -> None:
    """Run a simulation for a user, phrase, and language model.

    Optionally, a signal model path can be provided to use a specific signal model for the simulation.
    If not provided, the script will search for a signal model in the mode calibration directory for the user.

    Parameters
    ----------
    data_dir : Path
        The path to the experimental data directory.
    user : str
        The user name.
    phrase : str
        The phrase to type.
    starting_index : int
        The starting index of the phrase.
    language_model : str
        The language model to use.
    signal_model_path : Optional[str], optional
        The path to the signal model to use for the simulation, by default None.
        If not provided, the script will search for a signal model in the mode calibration directory.
        Note*: The output structure in `init_simulation_dir` below should be updated to reflect the
            signal model used if varying signal models.

    Raises
    ------
    FileNotFoundError
        I. If the signal model path is not provided and a signal model cannot be found in the mode calibration directory.
        II. If a data directory cannot be found for the user.

    Outputs
    -------
    The simulation results will be saved in the output directory defined by OUTPUT_DIR.
    """
    print(f"Running simulation for {user} with phrase {phrase} and language model {language_model}")

    model_path = None
    params_path = None
    mode_calibration_dir = None
    # load the parameters and model for the simulation from the mode calibration directory

    if not signal_model_path:

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
    else:
        model_path = signal_model_path

    print(f"Using signal model: {model_path}")

    # load the parameters
    params_path = mode_calibration_dir / "parameters.json"
    parameters = load_json_parameters(params_path, value_cast=True)

    # update the parameters with the new phrase, starting index, and language model
    phrase_length = len(phrase) - starting_index
    print(f"Processing {user}:{phrase}:{language_model} of phrase length: {phrase_length}")
    parameters["task_text"] = phrase
    parameters["spelled_letters_count"] = starting_index
    parameters["lang_model_type"] = language_model

    # Below are task constraints that impact typing speed and letter selection. Here we use criteria based on phrase length
    #  and some sensible defaults.
    parameters["max_inq_len"] = phrase_length * 8
    parameters["max_selections"] = phrase_length * 2  # This should be 2 * the length of the phrase to type
    parameters["min_inq_per_series"] = 1
    parameters["max_inq_per_series"] = 8
    parameters["backspace_always_shown"] = True
    parameters["summarize_session"] = False
    parameters["lm_backspace_prob"] = 0.03571
    # signal model decision threshold for letter selection
    parameters["decision_threshold"] = 0.8
    parameters["max_minutes"] = 120  # This is not used in the simulation. But we set it high to avoid any issues.
    parameters["max_incorrect"] = int(phrase_length / 2)  # This should be half the length of the phrase to type

    # get the correct list of source directories from the data directory
    source_dirs = [str(file) for file in data_dir.iterdir() if file.is_dir() and DATA_PATTERN in file.name]
    if not source_dirs:
        raise FileNotFoundError(f"Could not find a data directory for {user}")

    # Construct the simulation task factory and output directory strucutre
    task_factory = TaskFactory(source_dirs=source_dirs,
                               signal_model_paths=[str(model_path)],
                               parameters=parameters)
    sim_dir = init_simulation_dir(
        save_location=Path(OUTPUT_DIR),
        prefix=f"{user}/{language_model}/{phrase}/{user}_{phrase}_{language_model}_")
    runner = TaskRunner(save_dir=sim_dir,
                        task_factory=task_factory,
                        runs=RUN_COUNT)
    # run the simulation and output the metrics
    try:
        runner.run()
        metrics.report(sim_dir)
    except Exception as e:
        print(f"Error running simulation for {user} with phrase {phrase} and language model {language_model}")
        print(e)


if __name__ == "__main__":
    from pathlib import Path

    from tqdm import tqdm

    from bcipy.io.load import load_experimental_data

    # load experimental directory. This will have users/run/data
    data_dir = Path(load_experimental_data())

    progress_bar = tqdm(
        Path(data_dir).iterdir(),
        total=len(list(Path(data_dir).iterdir())),
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [est. {remaining}][ela. {elapsed}]\n",
        colour='MAGENTA')

    # Loop over all the users, phrases and language models defined.
    # Optionally, a signal model could be provided using the signal_model_path parameter.
    #   This is available in case the trained model is not in the mode calibration directory. It could be used to evaluate different signal models.
    # We recommend changing the init_simulation_dir with some model name
    # indicator to help with organizing the output for analysis.
    for user in progress_bar:
        if user.is_dir():
            progress_bar.set_description(f"Processing {user.name}")
            for phrase, starting_index in PHRASES:
                for language_model in LANGUAGE_MODELS:
                    run_simulation(user, user.name, phrase, starting_index, language_model)

    progress_bar.close()
