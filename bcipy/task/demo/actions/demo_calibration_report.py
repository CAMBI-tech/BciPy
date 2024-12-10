import logging
from bcipy.task.actions import BciPyCalibrationReportAction
from bcipy.config import DEFAULT_PARAMETERS_PATH
from bcipy.io.load import load_json_parameters

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate a calibration report from a session data file.')
    # Add the arguments: parameters and protocol
    parser.add_argument(
        '--parameters',
        help=f'Path to a BciPy parameters file. Defaults to {DEFAULT_PARAMETERS_PATH}',
        default=DEFAULT_PARAMETERS_PATH)
    parser.add_argument(
        '--protocol',
        help=('Path to BciPy Protocol. Must contain one or more calibration session directories. '
              'If none provided, a window will prompt for one.'),
        default=None)
    args = parser.parse_args()
    parameters = load_json_parameters(args.parameters, value_cast=True)
    """
    **Note:**
        Save path is not used in this action, the protocol path is most important and where
        the report will be saved. In other tasks, the save path is used to save the output of the tasks
        (raw data, model, etc.). Whereas in this case, the report is saved in the protocol path because it reads
        several of the underlying files in the protocol. In other words, it is a top level report.

        The protocol path is the path to the directory containing the calibration sessions.
    """
    action = BciPyCalibrationReportAction(parameters=parameters, save_path='.', protocol_path=args.protocol)
    print('Generating Report.')
    task_data = action.execute()
