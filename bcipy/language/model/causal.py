from typing import Optional, List, Tuple

from bcipy.core.symbols import BACKSPACE_CHAR, SPACE_CHAR, DEFAULT_SYMBOL_SET
from bcipy.language.main import LanguageModelAdapter, ResponseType

from aactextpredict.causal import CausalLanguageModel

from bcipy.config import LM_PATH


class CausalLanguageModelAdapter(LanguageModelAdapter):
    """Character language model based on a pre-trained causal model, GPT-2 by default."""

    def __init__(self,
                 response_type: ResponseType,
                 symbol_set: List[str] = DEFAULT_SYMBOL_SET,
                 lang_model_name: Optional[str] = None,
                 lm_path: Optional[str] = None,
                 lm_device: str = "cpu",
                 lm_left_context: str = "",
                 beam_width: int = None,
                 fp16: bool = True,
                 mixed_case_context: bool = True,
                 case_simple: bool = True,
                 max_completed: int = None,
                 ):
        """
        Initialize instance variables and load the language model with given path
        Args:
            response_type - SYMBOL only
            symbol_set         - list of symbol strings
            lang_model_name    - name of the Hugging Face casual language model to load
            lm_path            - load fine-tuned model from specified directory
            lm_device          - device to use for making predictions (cpu, mps, or cuda)
            lm_left_context    - text to condition start of sentence on
            beam_width         - how many hypotheses to keep during the search, None=off
            fp16               - convert model to fp16 to save memory/compute on CUDA
            mixed_case_context - use mixed case for language model left context
            case_simple        - simple fixing of left context case
            max_completed      - stop search once we reach this many completed hypotheses, None=don't prune
        """
        super().__init__(response_type=response_type)

        causal_params = self.parameters['causal']

        self.beam_width = beam_width or int(causal_params['beam_width']['value'])
        self.max_completed = max_completed or int(causal_params['max_completed']['value'])

        # We optionally load the model from a local directory, but if this is not
        # specified, we load a Hugging Face model

        self.model_name = lang_model_name or causal_params['model_name']['value']

        local_model_path = lm_path or causal_params['model_path']['value']
        self.model_dir = f"{LM_PATH}/{local_model_path}" if local_model_path != "" else self.model_name

        # LM doesn't care about backspace, needs literal space
        self.symbol_set = [' ' if ch is SPACE_CHAR else ch for ch in symbol_set]
        self.symbol_set.remove(BACKSPACE_CHAR)

        self.model = CausalLanguageModel(self.model_name, symbol_set=self.symbol_set, lm_path=self.model_dir, 
                                         lm_device=lm_device, lm_left_context=lm_left_context, 
                                         beam_width=self.beam_width, fp16=fp16, mixed_case_context=mixed_case_context,
                                         case_simple=case_simple, max_completed=self.max_completed)


    def supported_response_types(self) -> List[ResponseType]:
        return [ResponseType.SYMBOL]
