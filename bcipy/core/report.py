# mypy: disable-error-code="union-attr"
import io
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Flowable, Image, KeepTogether, Paragraph,
                                SimpleDocTemplate)

from bcipy.config import BCIPY_FULL_LOGO_PATH
from bcipy.signal.evaluate.artifact import ArtifactDetection


class ReportSection(ABC):
    """Abstract base class for report sections in BciPy.

    This class defines the interface for creating sections in a BciPy Report.
    All report sections must implement the `compile` method to generate their content.
    """

    def compile(self) -> Flowable:
        """Compile the section into a flowable for the report.

        This method must be implemented by child classes. It is intended to be called
        during final report build, not immediately after class initialization.

        Returns:
            Flowable: A reportlab flowable object containing the section's content.
        """
        ...

    def _create_header(self) -> Flowable:
        """Create the header for the section.

        Returns:
            Flowable: A reportlab flowable object containing the section header.
        """
        ...


class SignalReportSection(ReportSection):
    """Section for displaying signal analysis results in a BciPy Report.

    This section can include signal figures and artifact detection results.

    Attributes:
        figures (List[Figure]): List of matplotlib figures to include in the report.
        report_flowables (List[Flowable]): List of reportlab flowables for the section.
        artifact (Optional[ArtifactDetection]): Optional artifact detection results.
        style: Reportlab style sheet for formatting.
    """

    def __init__(
            self,
            figures: List[Figure],
            artifact: Optional[ArtifactDetection] = None) -> None:
        """Initialize SignalReportSection.

        Args:
            figures (List[Figure]): List of matplotlib figures to include.
            artifact (Optional[ArtifactDetection], optional): Artifact detection results.
                Defaults to None.

        Raises:
            AssertionError: If artifact is provided but analysis is not complete.
        """
        self.figures = figures
        self.report_flowables: List[Flowable] = []
        self.artifact = artifact

        if self.artifact:
            assert self.artifact.analysis_done is not False, (
                "If providing artifact for this report, an analysis must be complete to run this report.")
        self.style = getSampleStyleSheet()

    def compile(self) -> Flowable:
        """Compile the signal report section into a flowable.

        Returns:
            Flowable: A reportlab flowable containing the compiled section.
        """
        self.report_flowables.append(self._create_header())
        if self.artifact:
            self.report_flowables.append(self._create_artifact_section())
        self.report_flowables.extend(self._create_epochs_section())

        return KeepTogether(self.report_flowables)

    def _create_artifact_section(self) -> Flowable:
        """Create a section displaying artifact detection results.

        Returns:
            Flowable: A reportlab flowable containing artifact information and visualizations.
        """
        artifact_report = []
        artifacts_detected = self.artifact.dropped
        artifact_text = '<b>Artifact:</b>'
        artifact_section = Paragraph(artifact_text, self.style['BodyText'])
        artifact_overview = f'<b>Artifacts Detected:</b> {artifacts_detected}'
        artifact_section = Paragraph(artifact_overview, self.style['BodyText'])
        artifact_report.append(artifact_section)

        if self.artifact.eog_annotations:
            eog_artifacts = f'<b>EOG Artifacts:</b> {len(self.artifact.eog_annotations)}'
            eog_section = Paragraph(eog_artifacts, self.style['BodyText'])
            artifact_report.append(eog_section)
            heatmap = self._create_heatmap(
                self.artifact.eog_annotations.onset,
                (0, self.artifact.total_time),
                'EOG')
            artifact_report.append(heatmap)

        if self.artifact.voltage_annotations:
            voltage_artifacts = f'<b>Voltage Artifacts:</b> {len(self.artifact.voltage_annotations)}'
            voltage_section = Paragraph(
                voltage_artifacts, self.style['BodyText'])
            artifact_report.append(voltage_section)

            # create a heatmap with the onset values of the voltage artifacts
            onsets = self.artifact.voltage_annotations.onset
            heatmap = self._create_heatmap(
                onsets,
                (0, self.artifact.total_time),
                'Voltage')
            artifact_report.append(heatmap)
        return KeepTogether(artifact_report)

    def _create_heatmap(self, onsets: List[float], range: Tuple[float, float], type: str) -> Image:
        """Create a heatmap visualization of artifact onsets.

        Args:
            onsets (List[float]): List of artifact onset times.
            range (Tuple[float, float]): Time range for the heatmap.
            type (str): Type of artifact being visualized.

        Returns:
            Image: A reportlab Image containing the heatmap.
        """
        # create a heatmap with the onset values
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 2)
        ax.hist(onsets, bins=100, range=range, color='red', alpha=0.7)
        ax.set_title(f'{type} Artifact Onsets')
        ax.set_xlabel('Time (s)')
        # make the label text smaller
        ax.set_ylabel('Frequency')
        heatmap = self.convert_figure_to_image(fig)
        return heatmap

    def _create_epochs_section(self) -> List[Image]:
        """Create a section containing all signal figures.

        Returns:
            List[Image]: List of reportlab Images containing the signal figures.
        """
        # create a flowable for each figure
        flowables = [self.convert_figure_to_image(fig) for fig in self.figures]
        return flowables

    def convert_figure_to_image(self, fig: Figure) -> Image:
        """Convert a matplotlib figure to a reportlab Image.

        Args:
            fig (Figure): Matplotlib figure to convert.

        Returns:
            Image: A reportlab Image containing the figure.
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        x, y = fig.get_size_inches()
        return Image(buf, x * inch, y * inch)

    def _create_header(self) -> Paragraph:
        """Create the header for the signal report section.

        Returns:
            Paragraph: A reportlab Paragraph containing the section header.
        """
        header = Paragraph('<u>Signal Report</u>', self.style['Heading3'])
        return header


class SessionReportSection(ReportSection):
    """Section for displaying session summary information in a BciPy Report.

    Attributes:
        summary (dict): Dictionary containing session summary information.
        session_name (str): Name of the session or default name if not specified.
        style: Reportlab style sheet for formatting.
        summary_table (Optional[Flowable]): The compiled summary table.
    """

    def __init__(self, summary: Dict[str, Any]) -> None:
        """Initialize SessionReportSection.

        Args:
            summary (Dict[str, Any]): Dictionary containing session summary information.
        """
        self.summary = summary
        if 'task' in self.summary:
            self.session_name = self.summary['task']
        else:
            self.session_name = 'Session Summary'
        self.style = getSampleStyleSheet()
        self.summary_table: Optional[Flowable] = None

    def compile(self) -> Flowable:
        """Compile the session report section into a flowable.

        Returns:
            Flowable: A reportlab flowable containing the compiled section.
        """
        summary_table = self._create_summary_flowable()
        self.summary_table = summary_table
        return summary_table

    def _create_summary_flowable(self) -> Flowable:
        """Create a flowable containing the session summary.

        Returns:
            Flowable: A reportlab flowable containing the formatted summary.
        """
        if self.summary:
            # split the summary keys and values into a list
            values = list(self.summary.values())
            keys = sorted(self.summary.keys())
            # sort the keys by alphabetical order and get the values in the same order
            values = [self.summary[key] for key in keys]
            keys = [str(key).replace('_', ' ').capitalize() for key in keys]
            # create a seperate table with the keys and values
            summary_list = self._create_summary_text(keys, values)
            return KeepTogether(summary_list)

    def _create_summary_text(self, keys: List[str], values: List[Any]) -> List[Paragraph]:
        """Create formatted text for the summary.

        Args:
            keys (List[str]): List of summary keys.
            values (List[Any]): List of summary values.

        Returns:
            List[Paragraph]: List of reportlab Paragraphs containing the formatted summary.
        """
        # create a table with the keys and values, adding a header
        table = [self._create_header()]
        for key, value in zip(keys, values):
            text = Paragraph(f'<b>{key}</b>: {value}', self.style['BodyText'])
            table.append(text)
        return table

    def _create_header(self) -> Paragraph:
        """Create the header for the session report section.

        Returns:
            Paragraph: A reportlab Paragraph containing the section header.
        """
        header = Paragraph(
            f'<u>{self.session_name}</u>', self.style['Heading3'])
        return header


class Report:
    """Class for compiling and saving BciPy Reports.

    This class handles the creation of PDF reports containing multiple sections
    of signal analysis and session information.

    Attributes:
        DEFAULT_NAME (str): Default name for the report file.
        sections (List[ReportSection]): List of report sections to include.
        elements (List[Flowable]): List of reportlab flowables for the report.
        name (str): Name of the report file.
        path (str): Full path where the report will be saved.
        document (SimpleDocTemplate): Reportlab document template.
        styles: Reportlab style sheet for formatting.
        header (Optional[Flowable]): Report header containing logo and title.
    """

    DEFAULT_NAME: str = 'BciPyReport.pdf'

    def __init__(self,
                 save_path: str,
                 name: Optional[str] = None,
                 sections: Optional[List[ReportSection]] = None,
                 autocompile: bool = False):
        """Initialize Report.

        Args:
            save_path (str): Directory where the report will be saved.
            name (Optional[str], optional): Name of the report file. Defaults to None.
            sections (Optional[List[ReportSection]], optional): List of report sections.
                Defaults to None.
            autocompile (bool, optional): Whether to compile the report immediately.
                Defaults to False.

        Raises:
            AssertionError: If sections is not a list or contains invalid section types.
            AssertionError: If name does not end with .pdf.
        """
        if sections:
            assert isinstance(sections, list), "Sections should be a list."
            assert all(isinstance(section, ReportSection)
                       for section in sections), "All sections should be of type ReportSection."
        self.sections = sections or []
        self.elements: List[Flowable] = []
        if not name:
            name = self.DEFAULT_NAME
        assert name.endswith('.pdf'), "The report name should end with .pdf"

        self.name = name
        self.path = f'{save_path}/{name}'
        self.document = SimpleDocTemplate(self.path, pagesize=letter)

        self.styles = getSampleStyleSheet()
        self.header: Optional[Flowable] = None

        if sections is not None and autocompile:
            self.compile()

    def add(self, section: ReportSection) -> None:
        """Add a section to the report.

        Args:
            section (ReportSection): The section to add to the report.
        """
        self.sections.append(section)

    def compile(self) -> None:
        """Compile the report by adding the header and all sections."""
        if self.header is None:
            self._construct_report_header()
            header_group = KeepTogether(self.header)
            self.elements.append(header_group)
        for section in self.sections:
            self.elements.append(section.compile())

    def save(self) -> None:
        """Save the report as a PDF file."""
        self.document.build(self.elements)

    def _construct_report_header(self) -> None:
        """Construct the report header with logo and title.

        Raises:
            AssertionError: If called after other elements have been added.
        """
        assert len(
            self.elements) < 1, "The report header should be constructed before other elements"
        report_title = Paragraph('BciPy Report', self.styles['Title'])
        logo = Image(BCIPY_FULL_LOGO_PATH, hAlign='LEFT', width=170, height=50)
        report_title.hAlign = 'CENTER'
        report_title.height = 100
        header = [logo, report_title]
        self.header = header


if __name__ == '__main__':
    # use the demo_visualization.py to generate figure handles
    # sr = SignalReportSection(figure_handles)
    report = Report('.')
    session = {'session': 1, 'date': '2021-10-01'}
    session_text = SessionReportSection(session)
    report.add(session_text)
    # report.add(sr)
    report.compile()
    report.save()
