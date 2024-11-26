"""Command line interface for the simulator."""

import json
import logging
from glob import glob
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from rich import print as rprint
from rich.filesize import decimal
from rich.markup import escape
from rich.prompt import Confirm, Prompt, PromptBase
from rich.text import Text
from rich.tree import Tree

from bcipy.gui.file_dialog import ask_directory, ask_filename
from bcipy.helpers.acquisition import active_content_types
from bcipy.helpers.load import choose_model_paths
from bcipy.simulator.data.sampler import (InquirySampler, Sampler,
                                          TargetNontargetSampler)
from bcipy.simulator.task.copy_phrase import SimulatorCopyPhraseTask
from bcipy.simulator.task.task_factory import TaskFactory
from bcipy.simulator.util.artifact import TOP_LEVEL_LOGGER_NAME
from bcipy.task.paradigm.rsvp.copy_phrase import RSVPCopyPhraseTask

logger = logging.getLogger(TOP_LEVEL_LOGGER_NAME)


def do_directories(parent: Path,
                   fn: Callable[[Path], Any],
                   max_depth: int = 3,
                   current_depth: int = 1) -> None:
    """Recursively walk a tree of directories, calling the provided
    function on each path."""

    paths = sorted(
        Path(parent).iterdir(),
        key=lambda path: (path.is_file(), path.name.lower()),
    )
    for path in paths:
        # Remove hidden files
        if path.name.startswith("."):
            continue
        if path.is_dir():
            fn(path)
            if (current_depth + 1) <= max_depth:
                do_directories(path, fn, max_depth, current_depth + 1)


def walk_directory(directory: Path,
                   tree: Tree,
                   show_files: bool = True,
                   skip_dir: Callable[[Path], bool] = lambda p: False) -> None:
    """Recursively build a Tree with directory contents.
    Adapted from rich library examples/tree.py

    Parameters
    ----------
        directory - root directory of the tree
        tree - Tree object that gets recursively updated
        show_files - boolean indicating if files should be displayed
        skip_dir - predicate to determine if a directory should be skipped
    """
    # Sort dirs first then by filename
    paths = sorted(
        Path(directory).iterdir(),
        key=lambda path: (path.is_file(), path.name.lower()),
    )
    for path in paths:
        # Remove hidden files

        if path.name.startswith(".") or (path.is_dir() and skip_dir(path)):
            continue
        if path.is_dir():
            style = "dim" if path.name.startswith("__") else ""
            branch = tree.add(
                f"[blue]:open_file_folder: [link file://{path}]{escape(path.name)}",
                style=style,
                guide_style=style,
            )
            walk_directory(path, branch, show_files, skip_dir)
        else:
            if show_files:
                text_filename = Text(path.name)
                text_filename.stylize(f"link file://{path}")
                file_size = path.stat().st_size
                text_filename.append(f" ({decimal(file_size)})", "blue")
                tree.add(text_filename)
            else:
                continue


def print_directories(parent, paths: List[Path]) -> None:
    """Print the list of directories to terminal as a tree."""
    assert paths, "Paths are required"
    paths = sorted(paths)

    def pred(p):
        return p not in paths

    show_tree(parent, pred)


def excluded(path: Path) -> bool:
    """Predicate to determine if a path should not be presented."""
    return path.is_file() or path.name == 'logs' or 'Action' in path.name


def select_directories(parent: Path) -> List[Path]:
    """Select all directories of interest within the parent path.
    Traverses all directories and prompts for each."""
    accum = []

    def do_prompt(path: Path):
        if not excluded(path) and Confirm.ask(str(path.relative_to(parent))):
            accum.append(path)

    do_directories(parent, do_prompt, max_depth=2)
    return accum


def select_subset(parent, paths: List[Path]) -> List[Path]:
    """Prompt the user to confirm the subset of paths that should be retained."""
    return [
        path for path in paths
        if not excluded(path) and Confirm.ask(str(path.relative_to(parent)))
    ]


def show_tree(directory: Path,
              skip_pred: Callable[[Path], bool] = lambda p: False):
    """Prints a tree representation of a directory.

    Parameters
    ----------
        directory - root path to display
        skip_pred - callable function to determine if a branch in the tree should
            be skipped during rendering.
    """
    tree = Tree(
        f":open_file_folder: [link file://{directory}]{directory}",
        guide_style="bold bright_blue",
    )
    walk_directory(Path(directory), tree, show_files=False, skip_dir=skip_pred)
    rprint(tree)


class PromptPath(PromptBase[Path]):
    """A prompt that returns a Path.

    Example:
        >>> name = PromptPath.ask("Enter a file name")
    """

    response_type = Path

    def process_response(self, value: str) -> Path:
        value = value.strip("'\"")
        return super().process_response(value)


def select_data_folder() -> Path:
    """Select a root data folder for simulation."""
    # PromptPath.ask("Input a data folder")
    return Path(ask_directory("Select a data folder"))


def prompt_filter() -> str:
    """Prompt for an optional glob filter."""
    return Prompt.ask("Enter a glob filter (optional)", default="*")


def select_data(parent_folder: Optional[str] = None) -> List[Path]:
    """Select data to use for the simulation."""
    if parent_folder:
        path = Path(parent_folder)
    else:
        selected = select_data_folder()
        path = Path(selected)

    glob_pattern = prompt_filter()
    paths = sorted([
        Path(p) for p in glob(str(Path(path, glob_pattern)), recursive=True)
        if not excluded(Path(p))
    ])
    print_directories(path, paths)
    if not Confirm.ask("Use all of these?"):
        paths = select_subset(path, paths)
        print_directories(path, paths)
    return paths


def choose_sampling_strategy() -> type[Sampler]:
    """Choose a sampling strategy"""
    classes = [InquirySampler, TargetNontargetSampler]
    options = {klass.__name__: klass for klass in classes}
    selected = Prompt.ask("Choose a sampling strategy",
                          choices=options.keys(),
                          default='TargetNonTargetSampler')
    return options[selected]


def choose_task() -> type[RSVPCopyPhraseTask]:
    """Choose a task to simulate"""
    classes = [SimulatorCopyPhraseTask]
    options = {klass.__name__: klass for klass in classes}
    selected = Prompt.ask("Choose a simulation task",
                          choices=options.keys(),
                          default='SimulatorCopyPhraseTask')
    return options.get(selected, SimulatorCopyPhraseTask)


def select_models(acq_mode: str) -> List[Path]:
    """Select the signal models to use in simulation."""
    signal_models = choose_model_paths(active_content_types(acq_mode))
    return signal_models


def select_parameters_path() -> str:
    """Select the parameters path"""
    return ask_filename(file_types="*.json", prompt="Select Parameters file")


def get_acq_mode(params_path: str):
    """Extract the acquisition mode field from the json file at the given path."""
    with open(params_path, 'r', encoding='utf8') as json_file:
        params = json.load(json_file)
        return params['acq_mode'].get('value', 'EEG')


def command(params: str, models: List[str], source_dirs: List[str]) -> str:
    """Command equivalent to to the result of the interactive selection of
    simulator inputs."""
    model_args = ' '.join([f"-m {path}" for path in models])
    dir_args = ' '.join(f"-d {source}" for source in source_dirs)
    return f"bcipy-sim -p {params} {model_args} {dir_args}"


def main(args: Dict[str, Any]) -> TaskFactory:
    """Main function"""
    params = args.get('parameters', None)
    if not params:
        params = select_parameters_path()

    model_paths = args.get('model_path', None)
    if not model_paths:
        model_paths = select_models(get_acq_mode(params))

    source_dirs = [
        str(path) for path in select_data(args.get('data_folder', None))
    ]

    print(command(params, model_paths, source_dirs))
    return TaskFactory(params_path=params,
                       source_dirs=source_dirs,
                       signal_model_paths=model_paths,
                       sampling_strategy=TargetNontargetSampler,
                       task=SimulatorCopyPhraseTask)


if __name__ == '__main__':
    main({})
