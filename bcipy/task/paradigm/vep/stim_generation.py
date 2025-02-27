"""Functions related to stimuli generation for VEP tasks"""
import itertools
import math
import random
from typing import Any, List, Optional

from bcipy.helpers.list import find_index, swapped
from bcipy.helpers.stimuli import (InquirySchedule, get_fixation,
                                   random_target_positions)


def generate_vep_calibration_inquiries(alp: List[str],
                                       timing: Optional[List[float]] = None,
                                       color: Optional[List[str]] = None,
                                       inquiry_count: int = 100,
                                       num_boxes: int = 4,
                                       is_txt: bool = True) -> InquirySchedule:
    """
    Generates VEP inquiries with target letters in all possible positions.

    In the VEP paradigm, all stimuli in the alphabet are displayed in each
    inquiry. The symbols with the highest likelihoods are displayed alone
    while those with lower likelihoods occur together.

    Parameters
    ----------
        alp(list[str]): stimuli
        timing(list[float]): Task specific timing for generatoar.
            [target, fixation, stimuli]
        color(list[str]): Task specific color for generator
            [target, fixation, stimuli]
        inquiry_count(int): number of inquiries in a calibration
        num_boxes(int): number of display boxes
        is_txt(bool): whether the stimuli type is text. False would be an image stimuli.

    Return
    ------
        schedule_inq(tuple(
            samples[list[list[str]]]: list of inquiries
            timing(list[list[float]]): list of timings
            color(list(list[str])): list of colors)): scheduled inquiries
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
                           num_boxes: int = 8,
                           inquiry_count: int = 100,
                           is_txt: bool = True) -> List[List[Any]]:
    """Generates inquiries"""
    fixation = get_fixation(is_txt)
    target_usage_count = {symbol: 0 for symbol in symbols}
    max_target_uses = 10

    #counter for each box to ensure <= 12 target placements per box
    box_target_usage_count = {box: 0 for box in range(num_boxes)}
    max_box_target_uses = 10
    max_inquiries_per_box = num_boxes * max_box_target_uses

    # repeat the symbols as necessary to ensure an adequate size for sampling
    # without replacement.
    population = symbols * math.ceil(inquiry_count / len(symbols))
    targets = random.sample(population, inquiry_count)
    # print(f"Targets: {targets}")
    inquiries = []

    for target in targets:
        #stop if every box has already been used max times
        if sum(box_target_usage_count.values()) >= max_inquiries_per_box:
            break
        #check if the target can still be placed in the inquiry
        if target_usage_count[target] < max_target_uses:
            
            # For first box forced target
            # target_pos = 0

            # For random target selection
            #find valid boxes where the target has been used less than max times
            valid_boxes_for_target = [
                idx for idx in range(num_boxes) if box_target_usage_count[idx] < max_box_target_uses
            ]

            #if no valid boxes left then stop
            if not valid_boxes_for_target:
                raise ValueError("No more valid boxes available to place the target.")

            target_pos = random.choice(valid_boxes_for_target)
            
            inquiry = [target, fixation] + generate_vep_inquiry(
                alphabet=symbols,
                num_boxes=num_boxes,
                target=target,
                target_pos=target_pos
            )
            inquiries.append(inquiry)

            #update counter for the target and the box used
            target_usage_count[target] += 1
            box_target_usage_count[target_pos] += 1 
    random.shuffle(inquiries)
    
    return inquiries



def stim_per_box(num_symbols: int,
                 num_boxes: int = 6,
                 max_empty_boxes: int = 0,
                 max_single_sym_boxes: int = 4) -> List[int]:
    """Determine the number of stimuli per vep box.

    Parameters
    ----------
        num_symbols - number of symbols
        num_boxes - number of boxes
        max_empty_boxes - the maximum number of boxes which won't have any
            symbols within them.
        max_single_sym_boxes - maximum number of boxes with a single symbol

    Returns
    -------
        A list of length num_boxes, where each number in the list represents
            the number of symbols that should be in the box at that position.

    Post conditions:
            The sum of the list should equal num_symbols. Further, there should
            be at most max_empty_boxes with value of 0 and max_single_sym_boxes
            with a value of 1.
    """
    # Logic based off of example sessions from:
    # https://www.youtube.com/watch?v=JNFYSeIIOrw
    # [[2, 3, 5, 5, 6, 7],
    # [2, 1, 10, 1, 1, 13],
    # [3, 4, 17, 1, 1, 2],
    # [1, 1, 1, 0, 1, 24],
    # [1, 2, 1, 22, 1, 1],
    # [2, 1, 1, 21, 2, 1],
    # [1, 1, 25, 0, 1, 0]]
    # and
    # [[7, 3, 4, 9, 2, 3],
    # 1, 1, 6, 9, 7, 2],
    # 1, 2, 18, 2, 3, 2],
    # 1, 1, 4, 4, 17, 1],
    # 1, 3, 1, 1, 20, 2],
    # 1, 1, 1, 20, 3, 2],
    # 1, 1, 1, 4, 21, 0]]

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
    """Generates a single random inquiry.

    Parameters
    ----------
        alphabet - list of symbols from which to select.
        num_boxes - number of display areas; symbols will be partitioned into
            these areas.
        target - target symbol for the generated inquiry
        target_pos - box index that should contain the target

    Returns
    -------
        An inquiry represented by a list of lists, where each sublist
        represents a display box and contains symbols that should appear in that box.

    Post-conditions:
        Symbols will not be repeated and all symbols will be partitioned into
        one of the boxes.
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
