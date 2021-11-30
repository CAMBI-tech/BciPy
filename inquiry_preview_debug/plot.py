import matplotlib.pyplot as plt
import umap
import seaborn as sns
import umap.plot
from pathlib import Path

from inquiry_preview_debug.data import load_data

if __name__ == "__main__":
    input_path = (
        Path(__file__).parent.parent.resolve()
        / "data"
        / "bcipy_recordings"
        / "p01"
        / "p01_1hz_RSVP_Calibration_Thu_12_Aug_2021_14hr02min19sec_-0700"
    )
    output_path = Path(__file__).parent.resolve() / "result.png"

    data, labels = load_data(input_path)
    mapper = umap.UMAP().fit(data.reshape(data.shape[1], -1))
    fig, ax = plt.subplots(constrained_layout=True, figsize=(16, 9))
    ax = umap.plot.points(mapper, labels=labels)
    # sns.scatterplot(
    #     x=mapper.embedding_[:, 0],
    #     y=mapper.embedding_[:, 1],
    #     hue=labels,
    #     legend=None,
    #     ax=ax,
    #     s=5,
    #     alpha=0.5,
    #     palette="husl",
    # )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
