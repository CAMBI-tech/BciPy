"""Script to help analyze stoppage criteria and determine an appropriate
cutoff threshold for making a decision."""
import argparse
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import pandas as pd

import bcipy.helpers.session as session
from bcipy.task.data import Inquiry


def parse_folder(name: str) -> Tuple[str, str]:
    """Split the directory name into component parts"""
    parts = name.split("_")
    participant = parts[0]
    zone_offset = parts[-1]
    time = parts[-2]
    year = parts[-3]
    month = parts[-4]
    day = parts[-5]

    fmt = '%d_%b_%Y_%Hhr%Mmin%Ssec_%z'
    stamp = datetime.strptime(f"{day}_{month}_{year}_{time}_{zone_offset}",
                              fmt)
    return participant, stamp.isoformat()


class SeriesSummary(NamedTuple):
    """Summarizes a BciPy Series"""
    participant: str
    mode: str
    task: str
    stamp: str
    series: int
    correct: bool
    likelihood: float
    above_threshold: bool
    inquiry_count: int
    alt_threshold: float
    alt_threshold_inquiry_count: Optional[int] = None
    alt_threshold_is_correct: Optional[bool] = None

    def __str__(self):
        fields = ', '.join([
            f"{name}={self.__getattribute__(name)}"
            for i, name in enumerate(SeriesSummary._fields) if i > 3
        ])
        return f"SeriesSummary({fields})"

    def __repr__(self) -> str:
        return str(self)


def first_inq_index_above(inquiries: List[Inquiry], threshold: float) -> int:
    """Given a list of inquiries, find the first that contains a trial with an
    accumulated likelihood greater than the provided threshold. Returns index."""
    for i, inq in enumerate(inquiries):
        if any(likelhood > threshold for likelhood in inq.likelihood):
            return i
    return None


def max_index(items: List[float]) -> int:
    """Returns the index of the trial with the highest likelihood."""
    assert items, "Items are required"

    maxitem = (None, None)  # index, value
    for i, item in enumerate(items):
        if not maxitem[1] or (item > maxitem[1]):
            maxitem = (i, item)
    return maxitem[0]


def summarize(session_path: Path,
              alt_threshold: float = 0.7) -> List[SeriesSummary]:
    """Summarize the session for the given data directory."""
    assert session_path.parts[
        -1] == "session.json", "Path to session.json required"
    data = session.read_session(session_path)
    rows = []
    for i, inquiries in enumerate(data.series):
        folder = session_path.parts[-2]
        if not inquiries:
            print(f"Skipping series {i} in {folder}\n")
            continue
        participant, stamp = parse_folder(folder)

        last_inq = inquiries[-1]

        likelihood = None
        if last_inq.selection:
            # This can happen if the session timed out before making a selection
            likelihood = last_inq.likelihood[data.symbol_set.index(
                last_inq.selection)]

        # find the first inquiry above the alt_threshold
        alt_threshold_inquiry_count = None
        alt_threshold_is_correct = None
        inq_i = first_inq_index_above(inquiries, alt_threshold)
        if inq_i:
            alt_threshold_inquiry_count = inq_i + 1
            inq = inquiries[inq_i]
            symbol_index = max_index(inq.likelihood)
            selection = data.symbol_set[symbol_index]
            alt_threshold_is_correct = selection == inq.target_letter

        rows.append(
            SeriesSummary(
                participant=participant,
                mode=data.mode,
                task=data.task,
                stamp=stamp,
                series=i + 1,
                correct=last_inq.selection == last_inq.target_letter,
                likelihood=likelihood,
                above_threshold=likelihood
                and likelihood >= data.decision_threshold,
                inquiry_count=len(inquiries),
                alt_threshold=alt_threshold,
                alt_threshold_inquiry_count=alt_threshold_inquiry_count,
                alt_threshold_is_correct=alt_threshold_is_correct))
    return rows


def summarize_all(session_paths: List[Path],
                  alt_threshold: float = 0.7) -> List[SeriesSummary]:
    """Summarize all session.json files. These should be for a single
    participant."""
    rows = []
    for session_path in session_paths:
        rows.extend(summarize(session_path, alt_threshold))
    return rows


def display(df: pd.DataFrame) -> pd.DataFrame:
    """Select a subset of fields for display"""
    fields = [
        'correct', 'likelihood', 'above_threshold', 'inquiry_count',
        'alt_threshold_inquiry_count', 'alt_threshold_is_correct'
    ]
    if 'inq_diff' in df.columns:
        fields.append('inq_diff')
    return df[fields]


def display_paths(paths: List[Path]):
    return '\n\t'.join([p.parent.name for p in paths])


def analyze(session_paths: List[Path],
            alt_threshold: float = 0.7,
            include_stats: bool = False):
    """Outputs a textual summary."""
    print(f"Analyzed:\n\t{display_paths(session_paths)}\n")
    print(f"Changing decision threshold to {alt_threshold}")
    df = pd.DataFrame(summarize_all(session_paths, alt_threshold))

    print(f"Total number of sessions: {len(set(df.stamp))}")
    print(f"Total number of series (letters): {len(df)}")
    print(f"Total number of inquiries: {sum(df.inquiry_count)}")

    ave_inquiry_count = round(df.inquiry_count.mean())
    print(f"\nAverage number of inquiries per selection: {ave_inquiry_count}")

    if include_stats:
        print(f"\n{df.inquiry_count.describe()}")

    print(f"Total correct selections: {len(df[df['correct'] == True])}")
    if include_stats:
        print(f"\n{(df[df['correct'] == True]).inquiry_count.describe()}\n")

    print(f"Total incorrect selections: {len(df[df['correct'] == False])}")

    if include_stats:
        print(f"\n{(df[df['correct'] == False]).inquiry_count.describe()}\n")

    df['inq_diff'] = df.inquiry_count - df.alt_threshold_inquiry_count

    print(
        f"\nNumber of series where a lower threshold resulted in an answer in fewer inquiries: {len(df[df['inq_diff'] > 0])}"
    )

    print(
        f"\tHow many fewer on average? {round(df[df['inq_diff'] > 0].inq_diff.mean())}"
    )

    stayed_same = df[(df['inq_diff'] > 0)
                     & (df['correct'] == df['alt_threshold_is_correct'])]
    print(f"\tNumber in which the answer stayed the same: {len(stayed_same)}")

    went_incorrect = df[(df['inq_diff'] > 0) & (df['correct'] == True) &
                        (df['alt_threshold_is_correct'] == False)]
    print(
        f"\tNumber in which the answer went from correct to incorrect: {len(went_incorrect)}"
    )

    saved_inquiries = int((df[df['inq_diff'] > 0]).inq_diff.sum())
    print(f"Saved inquiries: {saved_inquiries}")

    # TODO: assumes a correct backspace followed by a correct letter.
    # calculation could be refined by accounting for correct selection rate.
    cost = len(went_incorrect) * 2 * ave_inquiry_count
    print(
        f"Estimated cost (number of inquiries) needed for corrections (assumes correct backspace followed by correct selection): {cost}"
    )


def main():
    glob_help = ('glob pattern to select a subset of data folders'
                 ' Ex. "*RSVP_Copy_Phrase*"')
    parser = argparse.ArgumentParser(
        description="Analyzes the effects of changing the decision threshold")
    parser.add_argument('-d',
                        '--dir',
                        help='path to the parent directory (with the data folders)',
                        required=True,
                        default=None)
    parser.add_argument('-p', '--pattern', help=glob_help, default="*")
    parser.add_argument('-t',
                        '--threshold',
                        help='alternate threshold to analyze',
                        default=0.7,
                        type=float)
    args = parser.parse_args()

    parent_dir = Path(args.dir)
    folders = [
        Path(d) for d in glob(str(Path(parent_dir, args.pattern)))
        if Path(d).is_dir()
    ]

    session_files = [Path(folder, "session.json") for folder in folders]
    analyze(session_files, alt_threshold=args.threshold)


if __name__ == "__main__":
    main()
