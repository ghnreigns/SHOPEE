from metric import row_wise_f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F


def find_threshold(df, features, lower_count_thresh, upper_count_thresh, search_space):
    """
    Compute the optimal threshold for the given count threshold.
    """
    score_by_threshold = []
    best_score = 0
    best_threshold = -1
    features = F.normalize(features)
    for i in tqdm(search_space):
        sim_thresh = i / 100
        selection = (
            ((features @ features.T) > sim_thresh).cpu().numpy()
        )  # TODO: understand features and features.T more.
        matches = []
        oof = []
        for row in selection:
            oof.append(df.iloc[row].posting_id.tolist())
            matches.append(" ".join(df.iloc[row].posting_id.tolist()))
        tmp = df.groupby("label_group").posting_id.agg("unique").to_dict()
        df["target"] = df.label_group.map(tmp)
        scores, score = row_wise_f1_score(df.target, oof)
        df["score"] = scores
        df["oof"] = oof

        selected_score = df.query(
            f"count > {lower_count_thresh} and count < {upper_count_thresh}"
        ).score.mean()
        score_by_threshold.append(selected_score)
        if selected_score > best_score:
            best_score = selected_score
            best_threshold = i

    # plt.title(
    #     f"Threshold Finder for count in [{lower_count_thresh},{upper_count_thresh}]."
    # )
    # plt.plot(score_by_threshold)
    # plt.axis("off")
    # plt.show()
    print(f"Best score is {best_score} and best threshold is {best_threshold/100}")
    return best_score, best_threshold
