"""Helper functions for managing and parsing session.json data."""
import csv
import itertools
import json
import os
import sqlite3
import subprocess
from collections import namedtuple

import openpyxl
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import PatternFill
from openpyxl.styles.borders import BORDER_THIN, Border, Side
from openpyxl.styles.colors import BLACK, WHITE, YELLOW

from bcipy.helpers.load import (load_experiment_fields, load_experiments,
                                load_json_parameters)
from bcipy.helpers.system_utils import DEFAULT_ENCODING
from bcipy.helpers.validate import validate_field_data_written
from bcipy.task.data import Inquiry, Session


def session_data(data_dir: str):
    """Returns a dict of session data transformed to map the alphabet letter
    to the likelihood when presenting the evidence. Also removes attributes
    not useful for debugging."""

    parameters = load_json_parameters(os.path.join(data_dir,
                                                   "parameters.json"),
                                      value_cast=True)

    session_path = os.path.join(data_dir, "session.json")
    with open(session_path, 'r', encoding=DEFAULT_ENCODING) as json_file:
        session = Session.from_dict(json.load(json_file))
        data = session.as_dict(evidence_only=True)
        data['target_text'] = parameters['task_text']
        return data


def collect_experiment_field_data(experiment_name,
                                  save_path,
                                  file_name='experiment_data.json') -> None:
    experiment = load_experiments()[experiment_name]
    experiment_fields = load_experiment_fields(experiment)

    if experiment_fields:
        cmd = ('python bcipy/gui/experiments/ExperimentField.py -e'
               f' "{experiment_name}" -p "{save_path}" -f "{file_name}"')
        subprocess.check_call(cmd, shell=True)
        # verify data was written before moving on
        if not validate_field_data_written(save_path, file_name):
            raise Exception('Field data not collected!')


def get_target(task_type, inquiry: Inquiry):
    """Returns the target for the given inquiry. For icon tasks this information
    is in the inquiry, but for Copy Phrase it must be computed."""
    if task_type == 'Copy Phrase':
        return copy_phrase_target(inquiry.target_text, inquiry.current_text)
    return inquiry.target_letter


# Database record
EvidenceRecord = namedtuple('EvidenceRecord', [
    'series', 'inquiry', 'stim', 'lm', 'eeg', 'btn', 'cumulative',
    'inq_position', 'is_target', 'presented', 'above_threshold'
])


def create_session_db(session: Session, db_name: str = 'session.db'):
    """Creates a sqlite database from the given session data.

    Parameters
    ----------
        session - task data (evidence values, stim times, etc.)
        db_name - path  of database to write; defaults to session.db

    Database Schema
    ---------------
    trial:
    - id: int
    - target: str

    evidence:
    - trial integer (0-based)
    - inquiry integer (0-based)
    - letter text (letter or icon)
    - lm real (language model probability for the trial; same for every
        inquiry and only considered in the cumulative value during the
        first inquiry)
    - eeg real (likelihood for the given inquiry; a value of 1.0 indicates
        that the letter was not presented)
    - cumulative real (cumulative likelihood for the trial thus far)
    - inq_position integer (inquiry position; null if not presented)
    - is_target integer (boolean; true(1) if this letter is the target)
    - presented integer (boolean; true if the letter was presented in
        this inquiry)
    - above_threshold (boolean; true if cumulative likelihood was above
        the configured threshold)
    """
    assert session.has_evidence(), "There is no evidence in the provided session"
    assert session.symbol_set, "Session must define a symbol_set"
    # Create database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute('CREATE TABLE trial (id integer, target text)')
    cursor.execute(
        'CREATE TABLE evidence (series integer, inquiry integer, '
        'stim text, lm real, eeg real, btn real, cumulative real, inq_position '
        'integer, is_target integer, presented integer, above_threshold)')
    conn.commit()

    for series_index, inquiry_list in enumerate(session.series):
        for inq_index, inquiry in enumerate(inquiry_list):

            target = get_target(session.task, inquiry)

            if inq_index == 0:
                # create record for the trial
                conn.executemany('INSERT INTO trial VALUES (?,?)',
                                 [(int(series_index + 1), target)])

            evidence = inquiry.stim_evidence(symbol_set=session.symbol_set)

            ev_rows = [
                EvidenceRecord(series=series_index + 1,
                               inquiry=inq_index,
                               stim=stim,
                               lm=evidence['lm_evidence'].get(stim, ''),
                               eeg=evidence['eeg_evidence'].get(stim, ''),
                               btn=evidence.get('btn_evidence',
                                                {}).get(stim, ''),
                               cumulative=evidence['likelihood'][stim],
                               inq_position=inquiry.stimuli.index(stim)
                               if stim in inquiry.stimuli else None,
                               is_target=(target == stim) if target else None,
                               presented=stim in inquiry.stimuli,
                               above_threshold=evidence['likelihood'][stim] >
                               session.decision_threshold)
                for stim, _likelihood_ev in evidence['likelihood'].items()
            ]

            conn.executemany(
                'INSERT INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?,?)', ev_rows)
            conn.commit()
    conn.close()


def session_db(data_dir: str, db_name='session.db'):
    """Writes a relational database (sqlite3) of session data that can
    be used for exploratory analysis.

    Parameters:
    -----------
        data_dir - directory with the session.json data (and parameters.json)
        db_name - name of database to write; defaults to session.db
    """
    with open(os.path.join(data_dir, "session.json"),
              'r',
              encoding=DEFAULT_ENCODING) as json_file:
        session = Session.from_dict(json.load(json_file))
        create_session_db(session, db_name)


def session_csv(db_name='session.db', csv_name='session.csv'):
    """Converts the sqlite3 db generated from session.json to a csv file,
    outputing the evidence table.
    """

    with open(csv_name, "w", encoding=DEFAULT_ENCODING, newline='') as output:
        cursor = sqlite3.connect(db_name).cursor()
        cursor.execute("select * from evidence;")
        columns = [description[0] for description in cursor.description]

        csv_writer = csv.writer(output, delimiter=',')
        csv_writer.writerow(columns)
        for row in cursor:
            csv_writer.writerow(row)


def write_row(excel_sheet, rownum, data, background=None, border=None):
    """Helper method to write a row to an Excel spreadsheet"""
    for col, val in enumerate(data, start=1):
        cell = excel_sheet.cell(row=rownum, column=col)
        cell.value = val
        if background:
            cell.fill = background
        if border:
            cell.border = border


def session_excel(db_name='session.db',
                  excel_name='session.xlsx',
                  include_charts=True):
    """Converts the sqlite3 db generated from session.json to an
    Excel spreadsheet"""

    # Define styles and borders to use within the spreadsheet.
    gray_background = PatternFill(start_color='ededed', fill_type='solid')
    white_background = PatternFill(start_color=WHITE, fill_type=None)
    highlighted_background = PatternFill(start_color=YELLOW, fill_type='solid')

    backgrounds_iter = itertools.cycle([gray_background, white_background])

    thin_gray = Side(border_style=BORDER_THIN, color='d3d3d3')
    default_border = Border(left=thin_gray,
                            top=thin_gray,
                            right=thin_gray,
                            bottom=thin_gray)

    thick_black = Side(border_style=BORDER_THIN, color=BLACK)
    new_series_border = Border(left=thin_gray,
                               top=thick_black,
                               right=thin_gray,
                               bottom=thin_gray)

    # Create the workbook
    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet.title = 'session'

    # Get the data from the generated sqlite3 database.
    cursor = sqlite3.connect(db_name).cursor()
    cursor.execute("select * from evidence;")
    columns = [description[0] for description in cursor.description]

    series_column = columns.index('series')
    inquiry_column = columns.index('inquiry')
    is_target_column = columns.index('is_target')
    series, inquiry, is_target = None, None, None

    # Write header
    write_row(sheet, 1, columns)

    # Maps the chart title to the starting row for the data.
    chart_data = {}

    # Write rows
    inq_background = next(backgrounds_iter)
    alp_len = None
    for i, row in enumerate(cursor):
        rownum = i + 2  # Excel is 1-indexed; also account for column row.
        is_target = int(row[is_target_column]) == 1
        border = default_border

        if row[series_column] != series:
            # Place a thick top border before each new series to make it
            # easier to visually scan the spreadsheet.
            series = row[series_column]
            border = new_series_border

        if row[inquiry_column] != inquiry:
            inquiry = row[inquiry_column]

            # Set the alphabet length used for chart data ranges.
            if not alp_len and i > 0:
                alp_len = i
            chart_data[f"Series {series} inquiry {inquiry}"] = rownum

            # Toggle the background for each inquiry for easier viewing.
            inq_background = next(backgrounds_iter)

        # write to spreadsheet
        write_row(
            sheet,
            rownum,
            row,
            background=highlighted_background if is_target else inq_background,
            border=border)

    # Add chart for each inquiry
    if include_charts:
        stim_col = columns.index('stim') + 1
        lm_col = columns.index('lm') + 1
        likelihood_col = columns.index('cumulative') + 1

        for title, min_row in chart_data.items():
            max_row = min_row + (alp_len - 1)
            chart = BarChart()
            chart.type = "col"
            chart.title = title
            chart.y_axis.title = 'likelihood'
            chart.x_axis.title = 'stimulus'

            data = Reference(sheet,
                             min_col=lm_col,
                             min_row=min_row,
                             max_row=max_row,
                             max_col=likelihood_col)
            categories = Reference(sheet,
                                   min_col=stim_col,
                                   min_row=min_row,
                                   max_row=max_row)
            chart.add_data(data, titles_from_data=False)
            chart.series[0].title = openpyxl.chart.series.SeriesLabel(v="lm")
            chart.series[1].title = openpyxl.chart.series.SeriesLabel(v="eeg")
            chart.series[2].title = openpyxl.chart.series.SeriesLabel(v="btn")
            chart.series[3].title = openpyxl.chart.series.SeriesLabel(
                v="combined")

            chart.set_categories(categories)
            sheet.add_chart(chart, f'M{min_row + 1}')

    # Freeze header row
    sheet.freeze_panes = 'A2'
    wb.save(excel_name)
    print("Wrote output to " + excel_name)


def copy_phrase_target(phrase: str, current_text: str, backspace='<'):
    """Determine the target for the current CopyPhrase inquiry.

    >>> copy_phrase_target("HELLO_WORLD", "")
    'H'
    >>> copy_phrase_target("HELLO_WORLD", "HE")
    'L'
    >>> copy_phrase_target("HELLO_WORLD", "HELLO_WORL")
    'D'
    >>> copy_phrase_target("HELLO_WORLD", "HEA")
    '<'
    >>> copy_phrase_target("HELLO_WORLD", "HEAL")
    '<'
    """
    try:
        # if the current_display is not a substring of phrase, there is a mistake
        # and the backspace should be the next target.
        phrase.index(current_text)
        return phrase[len(current_text)]
    except ValueError:
        return backspace


if __name__ == "__main__":
    import doctest
    doctest.testmod()
