from typing import List, Tuple
from pathlib import Path
from bcipy.helpers.task import alphabet
from bcipy.language import LanguageModel
from bcipy.language.base import ResponseType

import huggingface


class TransformerLanguageModel(LanguageModel):

    def predict(self, evidence: List[Tuple]) -> List[tuple]:
        return super().predict(evidence)

    def update(self) -> None:
        return super().update()

    def load(self, path: Path) -> None:
        self.model = huggface.load(path)
        return self.model

    def state_update(self, evidence: List[Tuple]) -> None:
        return super().state_update(evidence)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--data_folder', default=None)
    # args = parser.parse_args()
    # data_folder = args.data_folder
    symbol_set = alphabet()
    response_type = ResponseType.SYMBOL
    lm = TransformerLanguageModel(response_type, symbol_set)
    lm.load()
    # lm.predict(ev)
