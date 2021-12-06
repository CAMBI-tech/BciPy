import matplotlib.pyplot as plt
import umap
import seaborn as sns
import umap.plot
from pathlib import Path

from inquiry_preview_debug.data import load_data, load_session_csv
import seaborn as sns

# # Copy/paste the "EEG" column from spreadsheet to scrap.txt

# # cat scrap.txt | sort | uniq > scrap2.txt

# # Plot it:
# import seaborn as sns
# import matplotlib.pyplot as plt

# with open("scrap2.txt", "r") as fh:
#     contents = fh.read().splitlines()

# nums = [float(x) for x in contents]
# sns.violinplot(nums)
# plt.savefig("violin.png")


# if __name__ == "__main__":
#     input_path = (
#         Path(__file__).parent.parent.resolve() / "data" / "ip05_RSVP_Copy_Phrase_Thu_04_Nov_2021_11hr58min49sec_-0700"
#     )
#     output_path = Path(__file__).parent.resolve() / "result.png"

#     data, labels = load_data(input_path)
#     mapper = umap.UMAP().fit(data.reshape(data.shape[1], -1))
#     fig, ax = plt.subplots(constrained_layout=True, figsize=(16, 9))
#     ax = umap.plot.points(mapper, labels=labels)
#     # sns.scatterplot(
#     #     x=mapper.embedding_[:, 0],
#     #     y=mapper.embedding_[:, 1],
#     #     hue=labels,
#     #     legend=None,
#     #     ax=ax,
#     #     s=5,
#     #     alpha=0.5,
#     #     palette="husl",
#     # )
#     plt.savefig(output_path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    input_path = (
        Path(__file__).parent.parent.resolve()
        / "data"
        / "ip05_RSVP_Copy_Phrase_Thu_04_Nov_2021_11hr58min49sec_-0700"
        / "session.csv"
    )
    output_path = Path(__file__).parent.resolve() / "result.png"

    df = load_session_csv(input_path)
    targets = df[(df["presented"] == 1) & (df["is_target"] == 1)]
    nontargets = df[(df["presented"] == 1) & (df["is_target"] == 0)]

    subset = df[(df["presented"] == 1)]
    fig, ax = plt.subplots(constrained_layout=True, figsize=(16, 9))
    sns.stripplot(x="is_target", y="eeg", data=subset, ax=ax)
    sns.boxplot(
        x="is_target",
        y="eeg",
        data=subset,
        ax=ax,
        boxprops={"facecolor": "none", "edgecolor": "black"},
        medianprops={"color": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        showfliers=False,
    )
    sns.boxplot(
        x="is_target",
        y="eeg",
        data=subset,
        ax=ax,
        showmeans=True,
        meanline=True,
        showbox=False,
        showcaps=False,
        showfliers=False,
        zorder=10,
    )
    ax.axhline(1.0, color="red", linestyle="--", linewidth=2)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    print("targets", targets["eeg"].describe())
    print("nontargets", nontargets["eeg"].describe())
