"""Helper functions for the GUI code"""
import inspect
from typing import (Any, Dict, List, NamedTuple, Optional, Tuple, Type, Union,
                    get_args, get_origin)

# pylint: disable=wildcard-import,unused-wildcard-import
# flake8: noqa: F403
from bcipy.simulator.data.sampler import *
from bcipy.simulator.data.sampler import Sampler, TargetNontargetSampler

# This can be expanded as needed (ex. Path support).
SUPPORTED_INPUT_TYPES = ['int', 'str', 'float']


class InputField(NamedTuple):
    """Represents a GUI input parameter."""
    name: str
    input_type: str
    value: Optional[Any] = None
    required: bool = True


def get_param_value(param: inspect.Parameter) -> Any:
    """Default value of the argument."""
    is_empty = param.default is None or param.default == inspect.Parameter.empty
    if is_empty:
        return None
    return param.default


def is_annotated(param: inspect.Parameter) -> bool:
    """Test if an arg is annotated"""
    return param.annotation != inspect.Parameter.empty


def get_input_type(param: inspect.Parameter) -> Tuple[str, bool]:
    """For a given constructor parameter, determine the GUI input type. For
    subscriptable types such as Optional, the first type is returned (except None).

    Returns a tuple of (type str, required)
    """
    input_type = None
    required = True

    # if this is a subscriptable type, such as Optional or Union, get the first sub-type.
    origin = get_origin(param.annotation)
    if origin:
        # only check for Optional types. we don't currently provide GUI inputs for other
        # subscriptable types such as Lists or Dicts.
        if origin == Union:
            sub_types = get_args(param.annotation)
            if type(None) in sub_types:
                required = False
            input_type = next(sub_type for sub_type in sub_types
                              if not isinstance(sub_type, type(None))).__name__
    else:
        input_type = param.annotation.__name__
    return input_type, required


def get_inputs(input_class: Type[Any]) -> List[InputField]:
    """Given the class, determine the input parameters for GUI prompts.
    Only outputs parameters with a type annotation for basic types that can be
    input through a GUI (int, float, str).
    """

    # introspect the model arguments to determine what parameters to pass.
    params = inspect.signature(input_class).parameters

    inputs = []
    for key in params.keys():
        param = params[key]
        if is_annotated(param):
            input_type, required = get_input_type(param)
            # filter on types that we can input through the GUI
            if input_type in SUPPORTED_INPUT_TYPES:
                inputs.append(
                    InputField(name=param.name,
                               input_type=input_type,
                               value=get_param_value(param),
                               required=required))
    return inputs


def all_subclasses(parent_cls: Type[Any]):
    """Get all subclasses of the given class. Note that a class can only see
    a subclass if the module with the sub has been loaded.
    See: https://stackoverflow.com/questions/3862310/how-to-find-all-the-subclasses-of-a-class-given-its-name"""
    return set(parent_cls.__subclasses__()).union([
        sub for cls in parent_cls.__subclasses__()
        for sub in all_subclasses(cls)
    ])


def sampler_options(
    default: Type[Sampler] = TargetNontargetSampler
) -> Dict[str, Type[Sampler]]:
    """Returns available samplers as a name -> class dict.
    Orders with the default item first, then sorted alphabetically."""

    subclasses = all_subclasses(Sampler)
    if default in subclasses:
        subclasses.remove(default)
        subclasses = [
            default, *sorted(subclasses, key=lambda cls: cls.__name__)
        ]
    else:
        subclasses = sorted(subclasses, key=lambda cls: cls.__name__)
    return {cls.__name__: cls for cls in subclasses}


def sampler_inputs(sampler_class: Type[Sampler]) -> List[InputField]:
    """Returns the list of inputs needed to initialize instances of the given
    sampler class."""
    return get_inputs(sampler_class)
