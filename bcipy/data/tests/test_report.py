import os
import unittest
from unittest.mock import patch
import tempfile
import shutil

import matplotlib.pyplot as plt
import numpy as np
from reportlab.platypus import Paragraph, Image
from reportlab.platypus import Flowable, KeepTogether

from bcipy.data.report import Report, SessionReportSection, ReportSection, SignalReportSection


class TestReport(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_init_name(self):
        name = 'report_name.pdf'
        report = Report(self.temp_dir, name=name)
        self.assertEqual(report.name, name)

    def test_init_no_name_default(self):
        report = Report(self.temp_dir)
        self.assertEqual(report.name, Report.DEFAULT_NAME)

    def test_init_sections(self):
        summary = {}
        report_section = SessionReportSection(summary)
        section = [report_section]
        report = Report(self.temp_dir, sections=section)
        self.assertEqual(report.sections, section)

    def test_init_no_sections(self):
        report = Report(self.temp_dir)
        self.assertEqual(report.sections, [])

    def test_init_sections_not_list(self):
        with self.assertRaises(AssertionError):
            Report(self.temp_dir, sections='section')

    def test_init_sections_not_report_section(self):
        with self.assertRaises(AssertionError):
            Report(self.temp_dir, sections=['section'])

    def test_add_section(self):
        report = Report(self.temp_dir)
        summary = {
            'session': 'session_name',
            'date': '11/11/1111',
            'duration': 2,
            'participant': 'T01',
        }
        report_section = SessionReportSection(summary=summary)
        report.add(report_section)
        self.assertEqual(report.sections, [report_section])
        another_report_section = SessionReportSection(summary=summary)
        report.add(another_report_section)
        self.assertEqual(report.sections, [report_section, another_report_section])

    def test_save(self):
        report = Report(self.temp_dir)
        summary = {}
        report_section = SessionReportSection(summary)
        report.add(report_section)
        report.save()
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, report.name)))

    def test_save_no_sections(self):
        report = Report(self.temp_dir)
        report.save()
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, report.name)))

    def test_complile_adds_section_and_header(self):
        report = Report(self.temp_dir)
        summary = {
            'session': 'session_name',
            'date': '11/11/1111',
            'duration': 2,
            'participant': 'T01',
        }
        report_section = SessionReportSection(summary=summary)
        report.add(report_section)
        report.compile()
        self.assertEqual(len(report.elements), 2)

    def test_create_header_is_called_once_compile(self):
        with patch('bcipy.data.report.Report._construct_report_header') as mock_construct_header:
            report = Report(self.temp_dir)
            summary = {
                'session': 'session_name',
                'date': '11/11/1111',
                'duration': 2,
                'participant': 'T01',
            }
            report_section = SessionReportSection(summary=summary)
            report.add(report_section)
            report.compile()
            mock_construct_header.assert_called_once()


class TestSessionReportSection(unittest.TestCase):

    def setUp(self) -> None:
        self.session_data = {
            'session': 'session_name',
            'date': '11/11/1111',
            'duration': 2,
            'participant': 'T01',
        }

    def test_init(self):
        summary = {}
        report_section = SessionReportSection(summary)
        self.assertIsInstance(report_section, ReportSection)
        self.assertIsNotNone(report_section.style)

    def test_create_summary_text(self):
        report_section = SessionReportSection(summary=self.session_data)

        table = report_section._create_summary_flowable()
        self.assertIsInstance(table, Flowable)

    def test_create_header(self):
        summary = {}
        report_section = SessionReportSection(summary)
        header = report_section._create_header()
        self.assertIsInstance(header, Paragraph)

    def test_compile(self):
        report_section = SessionReportSection(self.session_data)
        compiled = report_section.compile()
        self.assertIsInstance(compiled, KeepTogether)


class TestSignalReportSectionSection(unittest.TestCase):

    def setUp(self) -> None:
        # create matplotlib figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(np.random.rand(10))
        self.figures = [self.fig]

    def test_init(self):
        report_section = SignalReportSection(self.figures)
        self.assertIsInstance(report_section, ReportSection)
        self.assertIsNotNone(report_section.style)
        self.assertEqual(report_section.figures, self.figures)

    def test_create_header(self):
        report_section = SignalReportSection(self.figures)
        header = report_section._create_header()
        self.assertIsInstance(header, Paragraph)

    def test_create_epochs_section(self):
        report_section = SignalReportSection(self.figures)
        epochs = report_section._create_epochs_section()
        self.assertIsInstance(epochs, list)
        self.assertIsInstance(epochs[0], Image)

    def test_convert_figure_to_image(self):
        report_section = SignalReportSection(self.figures)
        image = report_section.convert_figure_to_image(self.fig)
        self.assertIsInstance(image, Image)

    def test_compile(self):
        report_section = SignalReportSection(self.figures)
        compiled = report_section.compile()
        self.assertIsInstance(compiled, KeepTogether)


if __name__ == '__main__':
    unittest.main()
