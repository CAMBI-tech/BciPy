"""Helper functions for managing and parsing session.json data."""
import csv
import json
import os
import sqlite3
from collections import Counter

import pandas as pd

from bcipy.helpers.load import load_json_parameters
from bcipy.helpers.task import alphabet


def session_data(data_dir: str, alp=None):
    """Returns a dict of session data transformed to map the alphabet letter
    to the likelihood when presenting the evidence. Also removes attributes
    not useful for debugging."""

    # TODO: Better error handling for missing parameters.
    # Get the alphabet based on the provided parameters (txt or icon).
    parameters = load_json_parameters(os.path.join(data_dir,
                                                   "parameters.json"),
                                      value_cast=True)
    if parameters.get('is_txt_sti', False):
        parameters['is_txt_stim'] = parameters['is_txt_sti']

    if not alp:
        alp = alphabet(parameters=parameters)

    session_path = os.path.join(data_dir, "session.json")
    with open(session_path, 'r') as json_file:
        data = json.load(json_file)
        data['copy_phrase'] = parameters['task_text']
        for epoch in data['epochs'].keys():
            for trial in data['epochs'][epoch].keys():
                likelihood = dict(
                    zip(alp, data['epochs'][epoch][trial]['likelihood']))

                # Remove unused properties
                unused = [
                    'eeg_len', 'timing_sti', 'triggers', 'target_info',
                    'copy_phrase'
                ]
                remove_props(data['epochs'][epoch][trial], unused)

                data['epochs'][epoch][trial]['stimuli'] = data['epochs'][
                    epoch][trial]['stimuli'][0]

                # Associate letters to values
                data['epochs'][epoch][trial]['lm_evidence'] = dict(
                    zip(alp, data['epochs'][epoch][trial]['lm_evidence']))
                data['epochs'][epoch][trial]['eeg_evidence'] = dict(
                    zip(alp, data['epochs'][epoch][trial]['eeg_evidence']))
                data['epochs'][epoch][trial]['likelihood'] = likelihood

                # Display the 5 most likely values.
                data['epochs'][epoch][trial]['most_likely'] = dict(
                    Counter(likelihood).most_common(5))

        return data

def get_stimuli(task_type, sequence):
    """There is some variation in how tasks record session information.
    Returns the list of stimuli for the given trial/sequence"""
    if task_type == 'Copy Phrase':
        return sequence['stimuli'][0]
    return sequence['stimuli']

def get_target(task_type, sequence, above_threshold):
    """Returns the target for the given sequence. For icon tasks this information
    is in the sequence, but for Copy Phrase it must be computed."""
    if task_type == 'Copy Phrase':
        return copy_phrase_target(sequence['copy_phrase'], sequence['current_text'])
    return sequence.get('target_letter', None)

def session_db(data_dir: str, db_name='session.db', alp=None):
    """Writes a relational database (sqlite3) of session data that can
    be used for exploratory analysis.

    Parameters:
    -----------
        data_dir - directory with the session.json data (and parameters.json)
        db_name - name of database to write; defaults to session.db
        alp - optional alphabet to use; may be required if using icons that do
            not exist on the current machine.

    Returns:
    --------
        Creates a sqlite3 database and returns a pandas dataframe of the
        evidence table for use within a repl.

    Schema:
    ------
    trial:
        - id: int
        - target: str

    evidence:
        - trial integer (0-based)
        - sequence integer (0-based)
        - letter text (letter or icon)
        - lm real (language model probability for the trial; same for every
            sequence and only considered in the cumulative value during the
            first sequence)
        - eeg real (likelihood for the given sequence; a value of 1.0 indicates
            that the letter was not presented)
        - cumulative real (cumulative likelihood for the trial thus far)
        - seq_position integer (sequence position; null if not presented)
        - is_target integer (boolean; true(1) if this letter is the target)
        - presented integer (boolean; true if the letter was presented in
            this sequence)
        - above_threshold (boolean; true if cumulative likelihood was above
            the configured threshold)
    """
    # TODO: Better error handling for missing parameters.

    # Get the alphabet based on the provided parameters (txt or icon).
    parameters = load_json_parameters(os.path.join(data_dir,
                                                   "parameters.json"),
                                      value_cast=True)
    if parameters.get('is_txt_sti', False):
        parameters['is_txt_stim'] = parameters['is_txt_sti']
    if not alp:
        alp = alphabet(parameters=parameters)

    session_path = os.path.join(data_dir, "session.json")
    with open(session_path, 'r') as json_file:
        data = json.load(json_file)

        # Create database
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        cursor.execute('CREATE TABLE trial (id integer, target text)')
        cursor.execute(
            'CREATE TABLE evidence (series integer, sequence integer, '
            'stim text, lm real, eeg real, cumulative real, seq_position '
            'integer, is_target integer, presented integer, above_threshold)'
        )
        conn.commit()

        for series in data['epochs'].keys():
            for i, seq_index in enumerate(data['epochs'][series].keys()):
                sequence = data['epochs'][series][seq_index]
                session_type = data['session_type']

                target_letter = get_target(
                    session_type, sequence,
                    max(sequence['likelihood']) >
                    parameters['decision_threshold'])
                stimuli = get_stimuli(session_type, sequence)

                if i == 0:
                    # create record for the trial
                    conn.executemany(
                        'INSERT INTO trial VALUES (?,?)',
                        [(int(series), target_letter)])

                lm_ev = dict(zip(alp, sequence['lm_evidence']))
                cumulative_likelihoods = dict(zip(alp, sequence['likelihood']))

                ev_rows = []
                for letter, prob in zip(alp, sequence['eeg_evidence']):
                    seq_position = None
                    if letter in stimuli:
                        seq_position = stimuli.index(letter)
                    if target_letter:
                        is_target = 1 if target_letter == letter else 0
                    else:
                        is_target = None
                    cumulative = cumulative_likelihoods[letter]
                    above_threshold = cumulative >= parameters[
                        'decision_threshold']
                    ev_row = (int(series), int(seq_index), letter, lm_ev[letter],
                              prob, cumulative, seq_position, is_target,
                              seq_position is not None, above_threshold)
                    ev_rows.append(ev_row)

                conn.executemany(
                    'INSERT INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?)',
                    ev_rows)
                conn.commit()
        dataframe = pd.read_sql_query("SELECT * FROM evidence", conn)
        conn.close()
        return dataframe


def session_csv(db_name='session.db', csv_name='session.csv'):
    """Converts the sqlite3 db generated from session.json to a csv file,
    outputing the evidence table.
    """

    with open(csv_name, "w", encoding='utf-8', newline='') as output:
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
                  chart_sequences=True):
    """Converts the sqlite3 db generated from session.json to an
    Excel spreadsheet"""
    import itertools
    import openpyxl
    from openpyxl.styles import PatternFill
    from openpyxl.styles.colors import YELLOW, WHITE, BLACK
    from openpyxl.styles.borders import Border, Side, BORDER_THIN, BORDER_MEDIUM
    from openpyxl.chart import BarChart, Series, Reference

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

    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet.title = 'session'

    cursor = sqlite3.connect(db_name).cursor()
    cursor.execute("select * from evidence;")
    columns = [description[0] for description in cursor.description]

    series_column = columns.index('series')
    sequence_column = columns.index('sequence')
    is_target_column = columns.index('is_target')
    series, sequence, is_target = None, None, None

    # Write header
    write_row(sheet, 1, columns)

    chart_data = {}
    seq_background = next(backgrounds_iter)
    for i, row in enumerate(cursor):
        rownum = i + 2  # Excel is 1-indexed; also account for column
        is_target = int(row[is_target_column]) == 1
        border = default_border

        if row[series_column] != series:
            series = row[series_column]
            border = new_series_border

        if row[sequence_column] != sequence:
            sequence = row[sequence_column]
            chart_data[f"Series {series} Sequence {sequence}"] = rownum
            seq_background = next(backgrounds_iter)

        # write to spreadsheet
        write_row(
            sheet,
            rownum,
            row,
            background=highlighted_background if is_target else seq_background,
            border=border)

    # Add chart for each sequence
    if chart_sequences:
        for title, min_row in chart_data.items():
            max_row = min_row + 27  # TODO: compute this
            chart = BarChart()
            chart.type = "col"
            chart.title = title
            chart.y_axis.title = 'likelihood'
            chart.x_axis.title = 'stimulus'

            # TODO: parameterize these numbers
            data = Reference(sheet, min_col=4, min_row=min_row, max_row=max_row, max_col=6)
            cats = Reference(sheet, min_col=3, min_row=min_row, max_row=max_row)
            chart.add_data(data)

            chart.set_categories(cats)
            # chart.shape = 4
            sheet.add_chart(chart, f'M{min_row + 1}')

    # Freeze header row
    sheet.freeze_panes = 'A2'
    wb.save(excel_name)


def copy_phrase_target(phrase:str, current_text: str, backspace='<'):
    """Determine the target for the current CopyPhrase sequence. 
    
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


def remove_props(data, proplist):
    """Given a dict, remove the provided keys"""
    for prop in proplist:
        if prop in data:
            data.pop(prop)

if __name__ == "__main__":
    import doctest
    doctest.testmod()