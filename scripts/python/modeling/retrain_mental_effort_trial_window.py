from pathlib import Path
from bcipy.io.load import load_experimental_data, load_json_parameters
from bcipy.core.parameters import Parameters
from bcipy.signal.model.offline_analysis import offline_analysis
import json


TRIAL_WINDOWS = [
    "0.0:0.1",
    "0.1:0.2",
    "0.2:0.3",
    "0.3:0.4",
    "0.4:0.5",
    "0.5:0.6",
    "0.6:0.7",
    "0.7:0.8",
]

def train(run_path: Path, params: Parameters):
    
    response = offline_analysis(
        str(run_path),
        parameters=params,
        alert=False
    )
    return response


def main():
    # Load the experimental data
    path_to_data = Path(load_experimental_data())
    print(path_to_data)

    processed_data = {}
    for participant in path_to_data.iterdir():
        if participant.is_dir():
            # Get the participant ID from the folder name
            participant_id = participant.name
            print(f"Processing {participant_id}")

            if participant_id not in processed_data:
                processed_data[participant_id] = {}


            for trial_window in TRIAL_WINDOWS:
                for run in participant.iterdir():
                    if run.is_dir():
                        # Get the run ID from the folder name
                        run_id = run.name
                        if "Calibration" in run_id:
                            if run_id not in processed_data[participant_id]:
                                processed_data[participant_id][run_id] = {}
                            print(f"Processing {run_id} with trial window {trial_window}")
                            params = load_json_parameters(run / "parameters.json", value_cast=True)
                            params["trial_window"] = trial_window

                            response = train(run, params)

                            model = response[0]

                            processed_data[participant_id][run_id][trial_window] = {}
                            processed_data[participant_id][run_id][trial_window]["auc"] = model.auc
                            
                            # Train the model with the specified parameters
                            # train(run, params)
    return processed_data


if __name__ == "__main__":
    response = main()
    print(response)
    breakpoint()

    # Save the processed data to a JSON file
    output_path = Path("ME_trial_window_data.json")
    with open(output_path, "w") as f:
        json.dump(response, f, indent=4)
        print(f"Saved processed data to {output_path}")


# add some standard errors to the auc values - simple statistical analysis
# some variablility in p300 data

# what is contributing to misclassifications - proximity vs simailarity it target (can we drill down this answer)
                        