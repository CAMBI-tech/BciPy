import io
from abc import ABC
from typing import List, Optional, Tuple

from matplotlib import pyplot as plt

from matplotlib.figure import Figure
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Flowable, KeepTogether
from reportlab.lib.units import inch

from bcipy.config import BCIPY_FULL_LOGO_PATH
from bcipy.signal.evaluate.artifact import ArtifactDetection


class ReportSection(ABC):
    """Report Section.

    An abstract class to handle the creation of a section in a BciPy Report.
    """

    def compile(self) -> Flowable:
        """Compile.

        This method must be implemented by the child class.
        It is intented to be called on final Report build,
            as opposed to immediatley after class initiatlization,
            to compile the section into a usuable flowable for a Report.
        """
        ...

    def _create_header(self) -> Flowable:
        ...


class SignalReportSection(ReportSection):
    """Signal Report Section.

    A class to handle the creation of a Signal Report section in a BciPy Report.
    """

    def __init__(
            self,
            figures: List[Figure],
            artifact: Optional[ArtifactDetection] = None) -> None:
        self.figures = figures
        self.report_flowables: List[Flowable] = []
        self.artifact = artifact

        if self.artifact:
            assert self.artifact.analysis_done is not False, (
                "If providing artifact for this report, an analysis must be complete to run this report.")
        self.style = getSampleStyleSheet()

    def compile(self) -> Flowable:
        """Compile.

        Compiles the Signal Report sections into a flowable that can be used to generate a Report.
        """
        self.report_flowables.append(self._create_header())
        if self.artifact:
            self.report_flowables.append(self._create_artifact_section())
        self.report_flowables.extend(self._create_epochs_section())

        return KeepTogether(self.report_flowables)

    def _create_artifact_section(self) -> Flowable:
        """Create Artifact Section.

        Creates a paragraph with the artifact information. This is only included if an artifact detection is provided.
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
            voltage_section = Paragraph(voltage_artifacts, self.style['BodyText'])
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
        """Create Heatmap.

        Creates a heatmap image with the onset values of the voltage artifacts.
        """
        # create a heatmap with the onset values
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 3)
        ax.hist(onsets, bins=100, range=range, color='red', alpha=0.7)
        ax.set_title(f'{type} Artifact Onsets')
        ax.set_xlabel('Time (s)')
        # make the label text smaller
        ax.set_ylabel('Frequency')
        heatmap = self.convert_figure_to_image(fig)
        return heatmap

    def _create_epochs_section(self) -> List[Image]:
        """Create Epochs Section.

        Creates a flowable image for each figure in the Signal Report.
        """
        # create a flowable for each figure
        flowables = [self.convert_figure_to_image(fig) for fig in self.figures]
        return flowables

    def convert_figure_to_image(self, fig: Figure) -> Image:
        """Convert Figure to Image.

        Converts a matplotlib figure to a reportlab Image.
            retrieved from: https://nicd.org.uk/knowledge-hub/creating-pdf-reports-with-reportlab-and-pandas
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        x, y = fig.get_size_inches()
        return Image(buf, x * inch, y * inch)

    def _create_header(self) -> Paragraph:
        """Create Header.

        Creates a header for the Signal Report section.
        """
        header = Paragraph('<u>Signal Report</u>', self.style['Heading3'])
        return header


class SessionReportSection(ReportSection):
    """Session Report Section.

    A class to handle the creation of a Session Report section in a BciPy Report using a summary dictionary.
    """

    def __init__(self, summary: Optional[dict] = None) -> None:
        self.summary = summary
        self.style = getSampleStyleSheet()
        self.summary_table = None

    def compile(self) -> Flowable:
        """Compile.

        Compiles the Session Report sections into a flowable that can be used to generate a Report.
        """
        summary_table = self._create_summary_flowable()
        self.summary_table = summary_table
        return summary_table

    def _create_summary_flowable(self) -> Flowable:
        """Create Summary Flowable.

        Creates a flowable table with the summary dictionary.
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

    def _create_summary_text(self, keys: list, values: list) -> List[Paragraph]:
        """Create Summary Text.

        Creates a list of paragraphs with the keys and values from the provided summary.
        """
        # create a table with the keys and values, adding a header
        table = [self._create_header()]
        for key, value in zip(keys, values):
            text = Paragraph(f'<b>{key}</b>: {value}', self.style['BodyText'])
            table.append(text)
        return table

    def _create_header(self) -> Paragraph:
        """Create Header.

        Creates a header for the Session Report section.
        """
        header = Paragraph('<u>Session Summary</u>', self.style['Heading3'])
        return header


class Report:
    """Report.

    A class to handle compiling and saving a BciPy Report after at least one session.
    """

    DEFAULT_NAME: str = 'BciPyReport.pdf'

    def __init__(self,
                 save_path: str,
                 name: Optional[str] = None,
                 sections: Optional[List[ReportSection]] = None,
                 autocompile: bool = False):
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
        """Add.

        Adds a ReportSection to the Report.
        """
        self.sections.append(section)

    def compile(self) -> None:
        """Compile.

        Compiles the Report by adding the header and all sections to the elements list.
        """
        if self.header is None:
            self._construct_report_header()
            header_group = KeepTogether(self.header)
            self.elements.append(header_group)
        for section in self.sections:
            self.elements.append(section.compile())

    def save(self) -> None:
        """Save.

        Exports the Report to a PDF file.
        """
        self.document.build(self.elements)

    def _construct_report_header(self) -> None:
        """Construct Report Header.

        Constructs the header for the Report. This should be called before adding any other elements.
        The header should consist of the CAMBI logo and a report title.
        """
        assert len(self.elements) < 1, "The report header should be constructed before other elements"
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
