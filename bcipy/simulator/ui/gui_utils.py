"""Helper functions for the GUI code"""
import inspect
from typing import (Any, List, NamedTuple, Optional, Tuple, Type, Union,
                    get_args, get_origin)

from bcipy.simulator.data.sampler.base_sampler import Sampler

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


def get_inputs(sampler_type: Type[Sampler]) -> List[InputField]:
    """Given a sampler, determine the input parameters for GUI prompts.
    Only outputs parameters with a type annotation where that type can be
    input through a GUI (int, float, str).
    """

    # introspect the model arguments to determine what parameters to pass.
    params = inspect.signature(sampler_type).parameters

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
