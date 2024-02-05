from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import Dict, Any, Optional, List

import numpy as np

from bcipy.helpers.acquisition import max_inquiry_duration
from bcipy.simulator.helpers.state_manager import SimState
from bcipy.simulator.helpers.types import InquiryResult
from bcipy.simulator.simulator_base import Simulator


@dataclass
class SimMetrics1:
    total_series: Optional[int] = None
    total_inquiries: Optional[int] = None
    total_decisions: Optional[int] = None
    total_time_spent: Optional[float] = None

    inquiries_per_selection: Optional[float] = None


class RefereeHandler(ABC):
    @abstractmethod
    def handle(self, sim: Simulator) -> Dict[str, Any]:
        ...


class SimMetrics1Handler(RefereeHandler):

    def handle(self, sim: Simulator) -> Dict[str, Any]:
        state: SimState = getattr(sim, ("state_manager")).get_state()

        # calculating total decisions made
        flattened_inquiries: List[InquiryResult] = reduce(lambda l1, l2: l1 + l2,
                                                          state.series_results, [])
        total_decisions = len(list(filter(lambda inq: bool(inq.decision), flattened_inquiries)))

        # average number of inqs before a decision
        inq_counts = []
        for series in state.series_results:
            if [inq for inq in series if inq.decision]:
                inq_counts.append(len(series))
        inquiries_per_selection = round(np.array(inq_counts).mean(), 3)

        # inq and series counts
        total_inquiries = state.total_inquiry_count()
        total_series = state.series_n

        # total time spent estimate seconds =  (max_inq_time_estimate * total_inquiries)
        parameters = sim.get_parameters()
        max_inq_time = max_inquiry_duration(parameters)
        total_time_spent = total_inquiries * max_inq_time

        ret = SimMetrics1(total_series=total_series, total_inquiries=total_inquiries,
                          total_decisions=total_decisions, total_time_spent=total_time_spent,
                          inquiries_per_selection=inquiries_per_selection)
        return ret.__dict__


class MetricReferee(ABC):

    # May change to SimState that is passed around not whole Simulator
    # Depends on how much data encapsulated by SimState
    @abstractmethod
    def score(self, sim: Simulator):
        """ Generate registered metrics for simulation  """

    @abstractmethod
    def visualize(self, sim: Simulator):
        """ Generate registered visualizations for sim """

    @abstractmethod
    def add_metric_handler(self, name: str, handler: RefereeHandler):
        """ Add a metric calculation that will be executed upon self.score() """

    @abstractmethod
    def add_viz_handler(self, name: str, handler: RefereeHandler):
        """ Register a visualization that will be executed upon self.visualize() """


class RefereeImpl(MetricReferee):

    def __init__(self, metric_handlers=None, viz_handlers=None):
        self.metric_handlers: Dict[str, RefereeHandler] = metric_handlers if metric_handlers else {}
        self.viz_handlers: Dict[str, RefereeHandler] = viz_handlers if viz_handlers else {}

        self.inquiry_time: float = 1  # 1 inq -> 1 second
        # TODO maybe some configurable parameters for visualizations

    def score(self, sim: Simulator) -> SimMetrics1:
        metrics = SimMetrics1()
        metrics_dict = metrics.__dict__
        for handler_name, handler in self.metric_handlers.items():
            handler_metrics = handler.handle(sim)

            for key, val in handler_metrics.items():
                if key in metrics_dict:
                    metrics_dict[key] = val

        return SimMetrics1(**metrics_dict)

    def visualize(self, sim: Simulator):
        for handler_name, viz_handler in self.viz_handlers.items():
            viz_handler.handle(sim)

    def add_metric_handler(self, name: str, handler: RefereeHandler):
        self.metric_handlers[name] = handler

    def add_viz_handler(self, name: str, handler: RefereeHandler):
        self.viz_handlers[name] = handler
