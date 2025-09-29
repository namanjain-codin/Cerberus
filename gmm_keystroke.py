import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


class GMMKeystrokeModel:
    def __init__(self, M=3, delta=1.0, s_thresh=0.3, train_ratio=0.7, valid_ratio=0.3):
        self.M = M
        self.delta = delta
        self.s_thresh = s_thresh
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.users = []
        self.user_digraphs_train = {}
        self.user_digraphs_valid = {}
        self.user_gmm_params = {}

    def load_csv(self, csv_path: str) -> None:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["H", "UD", "DD"])  # ensure critical columns

        self.users = sorted(df["subject"].unique().tolist())

        for user_id in self.users:
            df_user = df[df["subject"] == user_id].copy()
            df_user = df_user.sample(frac=1.0, random_state=42).reset_index(drop=True)

            n_total = len(df_user)
            n_train = int(self.train_ratio * n_total)
            n_valid = int(self.valid_ratio * n_total)

            df_train = df_user.iloc[:n_train]
            df_valid = df_user.iloc[n_train : n_train + n_valid]

            self.user_digraphs_train[user_id] = self._subset_to_digraphs(df_train)
            self.user_digraphs_valid[user_id] = self._subset_to_digraphs(df_valid)

    def _subset_to_digraphs(self, df_subset: pd.DataFrame):
        digraph_dict = {}
        for _, row in df_subset.iterrows():
            digraph_str = self._make_digraph(row)
            hold_time = row["H"]
            if digraph_str not in digraph_dict:
                digraph_dict[digraph_str] = []
            digraph_dict[digraph_str].append(float(hold_time))
        for dg in list(digraph_dict.keys()):
            digraph_dict[dg] = np.array(digraph_dict[dg], dtype=float)
        return digraph_dict

    def _make_digraph(self, row) -> str:
        return str(row["key"])  # treat single key as digraph placeholder

    def fit(self) -> None:
        self.user_gmm_params = {}
        for user_id in self.users:
            self.user_gmm_params[user_id] = {}
            train_dict = self.user_digraphs_train.get(user_id, {})
            for digraph_str, times_array in train_dict.items():
                if len(times_array) < self.M or len(set(times_array)) < self.M:
                    continue
                X = times_array.reshape(-1, 1)
                gmm = GaussianMixture(n_components=self.M, covariance_type="full", random_state=42)
                gmm.fit(X)
                self.user_gmm_params[user_id][digraph_str] = (
                    gmm.means_, gmm.covariances_, gmm.weights_
                )

    def calculate_scores(self, split: str = "valid"):
        genuine_scores = []
        imposter_scores = []
        for query_user_id in self.users:
            for claimed_user_id in self.users:
                S = self._compute_similarity(query_user_id, claimed_user_id, split=split)
                if query_user_id == claimed_user_id:
                    genuine_scores.append(S)
                else:
                    imposter_scores.append(S)
        return genuine_scores, imposter_scores

    def _compute_similarity(self, query_user_id, claimed_user_id, split: str = "valid") -> float:
        if split == "valid":
            query_dict = self.user_digraphs_valid.get(query_user_id, {})
        else:
            query_dict = self.user_digraphs_valid.get(query_user_id, {})

        claimed_gmms = self.user_gmm_params.get(claimed_user_id, {})
        total_count = 0
        similarity_sum = 0.0

        for digraph_str, times_array in query_dict.items():
            if digraph_str not in claimed_gmms:
                continue
            means, covars, weights = claimed_gmms[digraph_str]
            means = means.flatten()
            covars = covars.flatten()
            weights = weights.flatten()

            for t in times_array:
                total_count += 1
                for i in range(min(self.M, len(means))):
                    mean_i = float(means[i])
                    covar_i = float(covars[i])
                    weight_i = float(weights[i])
                    stdev_i = float(np.sqrt(max(covar_i, 1e-12)))
                    lower = mean_i - self.delta * stdev_i
                    upper = mean_i + self.delta * stdev_i
                    if lower <= float(t) <= upper:
                        similarity_sum += weight_i

        if total_count == 0:
            return 0.0
        return similarity_sum / float(total_count)

    @staticmethod
    def compute_FAR(imposter_scores, threshold: float) -> float:
        if len(imposter_scores) == 0:
            return 0.0
        fa_count = sum(score >= threshold for score in imposter_scores)
        return fa_count / float(len(imposter_scores))

    @staticmethod
    def compute_FRR(genuine_scores, threshold: float) -> float:
        if len(genuine_scores) == 0:
            return 0.0
        fr_count = sum(score < threshold for score in genuine_scores)
        return fr_count / float(len(genuine_scores))


def train_gmm_model(csv_path: str, M: int = 3, delta: float = 1.0, train_ratio: float = 0.7, valid_ratio: float = 0.3):
    model = GMMKeystrokeModel(M=M, delta=delta, s_thresh=0.3, train_ratio=train_ratio, valid_ratio=valid_ratio)
    model.load_csv(csv_path=csv_path)
    model.fit()
    genuine_scores, imposter_scores = model.calculate_scores(split="valid")

    thresholds = np.arange(0.0, 1.01, 0.01)
    fpr_list = []
    tpr_list = []
    for thr in thresholds:
        FAR = GMMKeystrokeModel.compute_FAR(imposter_scores, float(thr))
        FRR = GMMKeystrokeModel.compute_FRR(genuine_scores, float(thr))
        fpr_list.append(FAR)
        tpr_list.append(1.0 - FRR)

    return np.array(fpr_list), np.array(tpr_list), thresholds


