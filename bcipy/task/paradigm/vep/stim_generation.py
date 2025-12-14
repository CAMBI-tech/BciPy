"""VEP stimulus generation module.

This module provides functions for generating visual stimuli used in VEP
(Visual Evoked Potential) tasks. It handles the creation of calibration
inquiries, stimulus box configurations, and inquiry schedules.
"""

import itertools
import math
import random
from typing import Any, List, Optional

from bcipy.core.list import find_index, swapped
from bcipy.core.stimuli import (InquirySchedule, get_fixation,
                                random_target_positions)


def generate_vep_calibration_inquiries(alp: List[str],
                                       timing: Optional[List[float]] = None,
                                       color: Optional[List[str]] = None,
                                       inquiry_count: int = 100,
                                       num_boxes: int = 4,
                                       is_txt: bool = True) -> InquirySchedule:
    """Generate VEP inquiries with target letters in all possible positions.

    In the VEP paradigm, all stimuli in the alphabet are displayed in each
    inquiry. The symbols with the highest likelihoods are displayed alone
    while those with lower likelihoods occur together.

    Args:
        alp: List of stimuli.
        timing: Task specific timing for generator [target, fixation, stimuli].
        color: Task specific color for generator [target, fixation, stimuli].
        inquiry_count: Number of inquiries in a calibration.
        num_boxes: Number of display boxes.
        is_txt: Whether the stimuli type is text (False for image stimuli).

    Returns:
        InquirySchedule: Schedule containing inquiries, timings, and colors.

    Raises:
        AssertionError: If timing list does not contain exactly 3 values.
    """
    if timing is None:
        timing = [0.5, 1, 2]
    default_colors = [
        '#00FF80', '#FFFFB3', '#CB99FF', '#FB8072', '#80B1D3', '#FF8232'
    ]
    if color is None:
        box_colors = list(
            itertools.islice(itertools.cycle(default_colors), num_boxes))
        color = ['green', 'red'] + box_colors
    assert len(
        timing
    ) == 3, "timing must include values for [target, fixation, stimuli]"

    inquiries = generate_vep_inquiries(alp, num_boxes, inquiry_count, is_txt)
    times = [timing for _ in range(inquiry_count)]
    colors = [color for _ in range(inquiry_count)]

    return InquirySchedule(inquiries, times, colors)


def generate_vep_inquiries(symbols: List[str],
                           num_boxes: int = 6,
                           inquiry_count: int = 20,
                           is_txt: bool = True) -> List[List[Any]]:
    """Generate a list of VEP inquiries.

    Args:
        symbols: List of symbols to use in inquiries.
        num_boxes: Number of display boxes.
        inquiry_count: Number of inquiries to generate.
        is_txt: Whether the stimuli type is text.

    Returns:
        List[List[Any]]: List of inquiries, where each inquiry contains:
            - Target symbol
            - Fixation point
            - List of symbols for each box
    """
    fixation = get_fixation(is_txt)
    target_indices = random_target_positions(inquiry_count,
                                             stim_per_inquiry=num_boxes,
                                             percentage_without_target=0)

    # repeat the symbols as necessary to ensure an adequate size for sampling
    # without replacement.
    population = symbols * math.ceil(inquiry_count / len(symbols))
    targets = random.sample(population, inquiry_count)

    return [[target, fixation] + generate_vep_inquiry(alphabet=symbols,
                                                      num_boxes=num_boxes,
                                                      target=target,
                                                      target_pos=target_pos)
            for target, target_pos in zip(targets, target_indices)]


def stim_per_box(num_symbols: int,
                 num_boxes: int = 6,
                 max_empty_boxes: int = 0,
                 max_single_sym_boxes: int = 4) -> List[int]:
    """Determine the number of stimuli per VEP box.

    This function distributes symbols across boxes based on rules derived from
    example sessions. It ensures a balanced distribution while allowing for
    some empty boxes and boxes with single symbols.

    Args:
        num_symbols: Number of symbols to distribute.
        num_boxes: Number of boxes to distribute symbols across.
        max_empty_boxes: Maximum number of boxes that can be empty.
        max_single_sym_boxes: Maximum number of boxes that can have a single symbol.

    Returns:
        List[int]: List where each number represents the number of symbols
            that should be in the box at that position.

    Notes:
        - The sum of the returned list equals num_symbols
        - There will be at most max_empty_boxes with value 0
        - There will be at most max_single_sym_boxes with value 1
        - Distribution is based on example sessions from:
          https://www.youtube.com/watch?v=JNFYSeIIOrw
    """
    if max_empty_boxes + max_single_sym_boxes >= num_boxes:
        max_empty_boxes = 0
        max_single_sym_boxes = num_boxes - 1

    # no more than 1 empty box
    empty_boxes = [0] * random.choice(range(max_empty_boxes + 1))

    # no more than 4 boxes with a single symbol
    single_boxes = [1] * random.choice(
        range((max_single_sym_boxes + 1) - len(empty_boxes)))

    boxes = empty_boxes + single_boxes

    remaining_symbols = num_symbols - sum(boxes)
    remaining_boxes = num_boxes - len(boxes)

    # iterate through the remaining boxes except for the last one
    for _box in range(remaining_boxes - 1):
        # each remaining box must have 2 or more symbols
        box_max = remaining_symbols - (2 * remaining_boxes - 1)

        n = random.choice(range(2, box_max + 1))
        boxes.append(n)
        remaining_boxes -= 1
        remaining_symbols = remaining_symbols - n

    # last box
    boxes.append(remaining_symbols)

    return sorted(boxes)


def generate_vep_inquiry(alphabet: List[str],
                         num_boxes: int = 6,
                         target: Optional[str] = None,
                         target_pos: Optional[int] = None) -> List[List[str]]:
    """Generate a single random VEP inquiry.

    Args:
        alphabet: List of symbols to select from.
        num_boxes: Number of display areas to partition symbols into.
        target: Target symbol for the inquiry.
        target_pos: Box index that should contain the target.

    Returns:
        List[List[str]]: List of lists where each sublist represents a display
            box and contains symbols that should appear in that box.

    Notes:
        - Symbols will not be repeated
        - All symbols will be partitioned into one of the boxes
        - If target is specified, it will be placed in the lowest count box
          greater than 0
    """
    box_counts = stim_per_box(num_symbols=len(alphabet), num_boxes=num_boxes)
    assert len(box_counts) == num_boxes
    syms = [sym for sym in alphabet]
    random.shuffle(syms)

    if target:
        # Move the target to the front so it gets put in the lowest count box
        # greater than 0.
        syms = swapped(syms, index1=0, index2=syms.index(target))

    # Put syms in boxes
    boxes = []
    sym_index = 0
    for count in box_counts:
        box = []
        for _i in range(count):
            box.append(syms[sym_index])
            sym_index += 1
        box.sort()  # sort stim within box
        boxes.append(box)

    random.shuffle(boxes)

    if target and target_pos is not None:
        # Find the box with the target and swap with the box currently at the
        # target_pos.
        current_target_pos = find_index(boxes, lambda box: target in box)
        if current_target_pos is not None:
            boxes = swapped(boxes,
                            index1=current_target_pos,
                            index2=target_pos)

    return boxes
