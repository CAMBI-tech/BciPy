"""Query module for BCI task control.

This module provides classes for managing stimulus presentation in BCI tasks.
It includes agents that determine which stimuli to present based on different
selection strategies, such as random selection or N-best selection based on
probability distributions.
"""

import random
from abc import ABC, abstractmethod
from typing import List, Optional, Any

import numpy as np

from bcipy.core.stimuli import best_selection


class StimuliAgent(ABC):
    """Abstract base class for stimulus selection agents.

    This class defines the interface for agents that select stimuli to present
    during BCI tasks. Subclasses implement different selection strategies.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset the agent's state."""
        ...

    @abstractmethod
    def return_stimuli(self, list_distribution: np.ndarray,
                       **kwargs: Any) -> List[str]:
        """Update agent with posterior probabilities and select queries.

        Args:
            list_distribution: List of posterior probability distributions.
            **kwargs: Additional keyword arguments.

        Returns:
            List[str]: Selected stimuli for the next query.
        """
        ...

    @abstractmethod
    def do_series(self) -> None:
        """Handle series completion.

        Called when the system decides on a class to let the agent update
        its state accordingly.
        """
        ...


class RandomStimuliAgent(StimuliAgent):
    """Stimuli agent that randomly selects queries.

    This agent chooses random sets of letters for queries instead of using
    probability-based selection.

    Attributes:
        alphabet: List of possible symbols to query from.
        len_query: Number of symbols to include in each query.
    """

    def __init__(self, alphabet: List[str], len_query: int = 4) -> None:
        """Initialize the random stimuli agent.

        Args:
            alphabet: List of possible symbols to query from.
            len_query: Number of symbols to include in each query.
        """
        self.alphabet = alphabet
        self.len_query = len_query

    def reset(self) -> None:
        """Reset the agent's state.

        This querying method is memoryless, so no reset is needed.
        """
        pass

    def return_stimuli(self,
                       list_distribution: np.ndarray,
                       constants: Optional[List[str]] = None) -> List[str]:
        """Return random elements from the alphabet.

        Args:
            list_distribution: List of probability distributions (unused).
            constants: Optional list of symbols to always include in the result.

        Returns:
            List[str]: Randomly selected symbols, with constants if provided.
        """
        tmp = [i for i in self.alphabet]
        query = random.sample(tmp, self.len_query)

        if constants:
            query[-len(constants):] = constants

        return query

    def do_series(self) -> None:
        """Handle series completion.

        This agent is stateless, so no action is needed.
        """
        pass


class NBestStimuliAgent(StimuliAgent):
    """Stimuli agent that selects the N most likely symbols.

    This agent updates its selection based on posterior probabilities,
    choosing the N symbols with highest probability for each query.

    Attributes:
        alphabet: List of possible symbols to query from.
        len_query: Number of symbols to include in each query.
    """

    def __init__(self, alphabet: List[str], len_query: int = 4) -> None:
        """Initialize the N-best stimuli agent.

        Args:
            alphabet: List of possible symbols to query from.
            len_query: Number of symbols to include in each query.
        """
        self.alphabet = alphabet
        self.len_query = len_query

    def reset(self) -> None:
        """Reset the agent's state.

        This agent is stateless, so no reset is needed.
        """
        pass

    def return_stimuli(self,
                       list_distribution: np.ndarray,
                       constants: Optional[List[str]] = None) -> List[str]:
        """Return the N most likely symbols based on probabilities.

        Selects symbols based on their probabilities in the distribution,
        where N is self.len_query. Symbols with equal probabilities are
        ordered randomly.

        Args:
            list_distribution: List of probability distributions. Only the
                last distribution is used.
            constants: Optional list of symbols to always include in the result.

        Returns:
            List[str]: Selected symbols, with constants if provided.
        """
        symbol_probs = list(zip(self.alphabet, list_distribution[-1]))
        randomized = random.sample(symbol_probs, len(symbol_probs))
        symbols, probs = zip(*randomized)
        return best_selection(selection_elements=list(symbols),
                              val=list(probs),
                              len_query=self.len_query,
                              always_included=constants)

    def do_series(self) -> None:
        """Handle series completion.

        This agent is stateless, so no action is needed.
        """
        pass
