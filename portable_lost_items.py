from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF, TruncatedSVD


POSITIVE_EVENT_TYPES = (1, 2)
DEFAULT_SEEDS = (41, 42, 43)
OFFICIAL_INCIDENT_START = "2025-10-01 00:00:00"
OFFICIAL_INCIDENT_END = "2025-11-01 00:00:00"
OFFICIAL_INCIDENT_START_TS = pd.Timestamp(OFFICIAL_INCIDENT_START)
OFFICIAL_INCIDENT_END_TS = pd.Timestamp(OFFICIAL_INCIDENT_END)


@dataclass
class BlendWeights:
    svd: float = 0.22
    als: float = 0.00
    cooc: float = 0.24
    author: float = 0.14
    genre: float = 0.04
    publisher: float = 0.01
    language: float = 0.005
    incident_pop: float = 0.03
    recent_pop: float = 0.02
    pop: float = 0.02
    book: float = 0.005
    quality: float = 0.03
    item_svd: float = 0.08
    user_knn: float = 0.12
    nmf: float = 0.0
    full_score: float = 0.0

    @classmethod
    def from_path(cls, path: Path | None) -> "BlendWeights":
        if path is None:
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Weights JSON must be an object.")
        if "best_weights" in payload:
            payload = payload["best_weights"]
        elif "weights" in payload:
            payload = payload["weights"]
        allowed_keys = set(cls().__dict__.keys())
        filtered = {key: value for key, value in payload.items() if key in allowed_keys}
        if not filtered:
            raise ValueError(
                "Could not find blend weights in JSON. Expected top-level weights or best_weights."
            )
        payload = filtered
        return cls(**payload)

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass
class SourceLimits:
    svd_topk: int = 600
    als_topk: int = 280
    cooc_topk: int = 500
    book_topk: int = 300
    author_topk: int = 400
    genre_topk: int = 350
    publisher_topk: int = 150
    language_topk: int = 120
    incident_pop_topk: int = 250
    recent_pop_topk: int = 200
    pop_topk: int = 160
    item_svd_topk: int = 500
    user_knn_topk: int = 500
    user_knn_neighbors: int = 120
    candidate_cap: int = 2200
    user_history_cap: int = 70
    entity_pref_cap: int = 25
    entity_items_cap: int = 100


@dataclass
class SvdConfig:
    components: int = 192
    recent_decay_days: float = 100.0
    read_weight: float = 2.8
    wishlist_weight: float = 1.6
    read_rating_boost: float = 0.06
    default_read_rating: float = 8.0
    incident_window_bonus: float = 1.25
    post_window_multiplier: float = 0.75


@dataclass
class AlsConfig:
    enabled: bool = False
    factors: int = 128
    regularization: float = 0.02
    iterations: int = 22
    alpha: float = 32.0
    use_gpu: bool = False


@dataclass
class ValidationConfig:
    incident_start: str = OFFICIAL_INCIDENT_START
    incident_end: str = OFFICIAL_INCIDENT_END
    mask_ratio: float = 0.2
    user_limit: int | None = None
    user_scope: str = "targets"
    mask_style: str = "heterogeneous"
    time_bias_power: float = 1.35
    user_heterogeneity: float = 0.75


@dataclass
class RerankerConfig:
    enabled: bool = False
    mode: str = "ranker"
    iterations: int = 450
    depth: int = 8
    learning_rate: float = 0.05
    l2_leaf_reg: float = 6.0
    train_topk_per_user: int = 140
    blend_alpha: float = 0.82
    task_type: str = "CPU"
    prediction_batch_size: int = 250_000


@dataclass
class LoadedData:
    interactions: pd.DataFrame
    targets: pd.DataFrame
    editions: pd.DataFrame
    book_genres: pd.DataFrame
    users: pd.DataFrame


@dataclass
class PreparedPseudoFrame:
    seed: int
    frame: pd.DataFrame
    builder: "CandidateBuilder"
    relevant_by_user: dict[int, set[int]]
    target_users: list[int]
    masked_size: int


@dataclass
class FittedReranker:
    model: object
    feature_cols: list[str]
    cat_features: list[str]
    prediction_kind: str


@dataclass
class SelfTrainConfig:
    enabled: bool = False
    topk_per_user: int = 1
    pair_weight_scale: float = 0.55


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portable NTO lost-items solution.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_flags(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--data-dir", type=Path, default=Path("data"))
        subparser.add_argument("--weights-json", type=Path, default=None)
        subparser.add_argument("--use-implicit-als", action="store_true")
        subparser.add_argument("--als-use-gpu", action="store_true")
        subparser.add_argument("--svd-components", type=int, default=192)
        subparser.add_argument("--candidate-cap", type=int, default=2200)
        subparser.add_argument("--user-history-cap", type=int, default=70)
        subparser.add_argument(
            "--recovery-bonus-scale",
            type=float,
            default=0.0,
            help="Optional tiny recall bonus on top of the old blend. 0.0 keeps exact old behavior.",
        )
        subparser.add_argument("--quiet", action="store_true")

    def add_validation_flags(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--incident-start",
            type=str,
            default=OFFICIAL_INCIDENT_START,
            help="Incident window start, inclusive. Default matches official October 2025 window.",
        )
        subparser.add_argument(
            "--incident-end",
            type=str,
            default=OFFICIAL_INCIDENT_END,
            help="Incident window end, exclusive. Default matches official October 2025 window.",
        )
        subparser.add_argument("--mask-ratio", type=float, default=0.2)
        subparser.add_argument("--user-limit", type=int, default=None)
        subparser.add_argument(
            "--validation-user-scope",
            type=str,
            choices=["all", "targets"],
            default="targets",
            help="Use all incident users or only targets.csv users for pseudo-validation.",
        )
        subparser.add_argument(
            "--validation-mask-style",
            type=str,
            choices=["uniform", "heterogeneous"],
            default="heterogeneous",
            help="Pseudo-mask sampling style. Heterogeneous better reflects uneven losses.",
        )
        subparser.add_argument(
            "--validation-time-bias",
            type=float,
            default=1.35,
            help="Bias masked pairs toward later incident timestamps when using heterogeneous masks.",
        )
        subparser.add_argument(
            "--validation-user-heterogeneity",
            type=float,
            default=0.75,
            help="Log-normal dispersion for per-user mask propensities in heterogeneous masks.",
        )

    def add_reranker_flags(
        subparser: argparse.ArgumentParser,
        include_training_window: bool = False,
    ) -> None:
        subparser.add_argument("--use-catboost-reranker", action="store_true")
        subparser.add_argument(
            "--catboost-mode",
            type=str,
            choices=["ranker", "classifier"],
            default="ranker",
        )
        subparser.add_argument("--catboost-iterations", type=int, default=450)
        subparser.add_argument("--catboost-depth", type=int, default=8)
        subparser.add_argument("--catboost-learning-rate", type=float, default=0.05)
        subparser.add_argument("--catboost-l2-leaf-reg", type=float, default=6.0)
        subparser.add_argument("--catboost-train-topk", type=int, default=140)
        subparser.add_argument("--catboost-blend-alpha", type=float, default=0.82)
        subparser.add_argument(
            "--catboost-task-type",
            type=str,
            choices=["CPU", "GPU"],
            default="CPU",
        )
        if include_training_window:
            subparser.add_argument(
                "--reranker-seeds",
                nargs="*",
                type=int,
                default=list(DEFAULT_SEEDS),
            )
            subparser.add_argument("--reranker-mask-ratio", type=float, default=0.2)
            subparser.add_argument("--reranker-train-user-limit", type=int, default=None)
            subparser.add_argument(
                "--reranker-incident-start",
                type=str,
                default=OFFICIAL_INCIDENT_START,
            )
            subparser.add_argument(
                "--reranker-incident-end",
                type=str,
                default=OFFICIAL_INCIDENT_END,
            )

    def add_selftrain_flags(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--use-pseudo-refit", action="store_true")
        subparser.add_argument("--pseudo-refit-topk", type=int, default=1)
        subparser.add_argument("--pseudo-refit-weight", type=float, default=0.55)

    validate_parser = subparsers.add_parser("validate", help="Run pseudo-incident validation.")
    add_shared_flags(validate_parser)
    validate_parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    add_validation_flags(validate_parser)
    add_reranker_flags(validate_parser)
    add_selftrain_flags(validate_parser)

    tune_parser = subparsers.add_parser("tune", help="Search better blend weights.")
    add_shared_flags(tune_parser)
    tune_parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    add_validation_flags(tune_parser)
    tune_parser.add_argument("--trials", type=int, default=120)
    tune_parser.add_argument("--out-json", type=Path, default=Path("blend_weights.json"))

    submit_parser = subparsers.add_parser("submit", help="Generate submission.csv.")
    add_shared_flags(submit_parser)
    add_reranker_flags(submit_parser, include_training_window=True)
    add_selftrain_flags(submit_parser)
    submit_parser.add_argument("--submission-path", type=Path, default=Path("submission.csv"))

    blend_parser = subparsers.add_parser("blend", help="Blend multiple submission CSVs.")
    blend_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    blend_parser.add_argument(
        "--input-submissions",
        nargs="+",
        type=Path,
        required=True,
        help="Submission CSV paths to blend.",
    )
    blend_parser.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=None,
        help="Optional weights aligned with --input-submissions.",
    )
    blend_parser.add_argument(
        "--method",
        type=str,
        choices=["rrf", "inverse_rank", "borda"],
        default="rrf",
    )
    blend_parser.add_argument(
        "--rrf-k",
        type=float,
        default=20.0,
        help="RRF denominator offset for method=rrf.",
    )
    blend_parser.add_argument("--submission-path", type=Path, default=Path("submission_blend.csv"))

    return parser.parse_args()


def log(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(message, flush=True)


def load_data(data_dir: Path) -> LoadedData:
    interactions = pd.read_csv(
        data_dir / "interactions.csv",
        parse_dates=["event_ts"],
        usecols=["user_id", "edition_id", "event_type", "rating", "event_ts"],
    )
    targets = pd.read_csv(data_dir / "targets.csv", usecols=["user_id"])
    editions = pd.read_csv(
        data_dir / "editions.csv",
        usecols=[
            "edition_id",
            "book_id",
            "author_id",
            "publication_year",
            "age_restriction",
            "language_id",
            "publisher_id",
        ],
    )
    book_genres = pd.read_csv(data_dir / "book_genres.csv", usecols=["book_id", "genre_id"])
    users = pd.read_csv(data_dir / "users.csv", usecols=["user_id", "gender", "age"])

    interactions["user_id"] = interactions["user_id"].astype("int64")
    interactions["edition_id"] = interactions["edition_id"].astype("int64")
    interactions["event_type"] = interactions["event_type"].astype("int8")
    targets["user_id"] = targets["user_id"].astype("int64")
    editions = editions.astype(
        {
            "edition_id": "int64",
            "book_id": "int64",
            "author_id": "int64",
            "publication_year": "int32",
            "age_restriction": "int32",
            "language_id": "int64",
            "publisher_id": "int64",
        }
    )
    book_genres = book_genres.astype({"book_id": "int64", "genre_id": "int64"})
    users["user_id"] = users["user_id"].astype("int64")

    return LoadedData(
        interactions=interactions,
        targets=targets,
        editions=editions,
        book_genres=book_genres,
        users=users,
    )


def unique_positive_events(interactions: pd.DataFrame) -> pd.DataFrame:
    return interactions[interactions["event_type"].isin(POSITIVE_EVENT_TYPES)].copy()


def build_weighted_pairs(positives: pd.DataFrame, max_ts: pd.Timestamp, cfg: SvdConfig) -> pd.DataFrame:
    pairs = positives.copy()
    anchor_ts = min(pd.Timestamp(max_ts), OFFICIAL_INCIDENT_END_TS)
    effective_ts = pairs["event_ts"].where(pairs["event_ts"] <= anchor_ts, anchor_ts)
    age_days = (anchor_ts - effective_ts).dt.total_seconds() / 86400.0
    is_read = pairs["event_type"].eq(2)
    base = np.where(is_read, cfg.read_weight, cfg.wishlist_weight)
    rating_factor = np.where(
        is_read,
        0.80 + cfg.read_rating_boost * pairs["rating"].fillna(cfg.default_read_rating).clip(1, 10),
        1.0,
    )
    incident_bonus = np.where(
        (pairs["event_ts"] >= OFFICIAL_INCIDENT_START_TS)
        & (pairs["event_ts"] < OFFICIAL_INCIDENT_END_TS),
        cfg.incident_window_bonus,
        1.0,
    )
    post_multiplier = np.where(
        pairs["event_ts"] >= OFFICIAL_INCIDENT_END_TS,
        cfg.post_window_multiplier,
        1.0,
    )
    pairs["pair_weight"] = (
        base
        * rating_factor
        * incident_bonus
        * post_multiplier
        * np.exp(-age_days / cfg.recent_decay_days)
    )
    pairs = (
        pairs.groupby(["user_id", "edition_id"], as_index=False)
        .agg(
            pair_weight=("pair_weight", "max"),
            last_ts=("event_ts", "max"),
            last_event_type=("event_type", "max"),
            max_rating=("rating", "max"),
        )
        .reset_index(drop=True)
    )
    pairs["days_since_last"] = (max_ts - pairs["last_ts"]).dt.total_seconds() / 86400.0
    return pairs


def parse_incident_window(incident_start: str, incident_end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.to_datetime(incident_start, errors="coerce")
    end_ts = pd.to_datetime(incident_end, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        raise ValueError("incident_start and incident_end must be valid datetimes")
    if start_ts >= end_ts:
        raise ValueError("incident_start must be earlier than incident_end")
    return pd.Timestamp(start_ts), pd.Timestamp(end_ts)


def build_incident_pair_frame(
    positives: pd.DataFrame,
    incident_start: str,
    incident_end: str,
 ) -> pd.DataFrame:
    incident_start_ts, incident_end_ts = parse_incident_window(incident_start, incident_end)
    incident_pairs = (
        positives[
            (positives["event_ts"] >= incident_start_ts)
            & (positives["event_ts"] < incident_end_ts)
        ]
        .groupby(["user_id", "edition_id"], as_index=False)
        .agg(
            last_ts=("event_ts", "max"),
            last_event_type=("event_type", "max"),
            max_rating=("rating", "max"),
        )
        .reset_index(drop=True)
    )
    if incident_pairs.empty:
        raise ValueError(
            "Incident window does not contain positive pairs. "
            f"window=[{incident_start_ts}, {incident_end_ts})"
        )
    return incident_pairs


def make_pseudo_mask(
    positives: pd.DataFrame,
    incident_start: str,
    incident_end: str,
    mask_ratio: float,
    seed: int,
    allowed_users: Iterable[int] | None = None,
    user_limit: int | None = None,
    mask_style: str = "uniform",
    time_bias_power: float = 1.0,
    user_heterogeneity: float = 0.0,
) -> pd.DataFrame:
    if not 0.0 < mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be in the interval (0, 1]")
    if user_limit is not None and user_limit <= 0:
        raise ValueError("user_limit must be positive when provided")
    if mask_style not in {"uniform", "heterogeneous"}:
        raise ValueError("mask_style must be one of {'uniform', 'heterogeneous'}")
    if time_bias_power <= 0.0:
        raise ValueError("time_bias_power must be positive")
    if user_heterogeneity < 0.0:
        raise ValueError("user_heterogeneity must be non-negative")

    incident_start_ts, incident_end_ts = parse_incident_window(incident_start, incident_end)
    incident_pairs = build_incident_pair_frame(positives, incident_start, incident_end)
    if allowed_users is not None:
        allowed_user_ids = set(int(user_id) for user_id in allowed_users)
        incident_pairs = incident_pairs[incident_pairs["user_id"].isin(allowed_user_ids)].reset_index(
            drop=True
        )
    if incident_pairs.empty:
        raise ValueError("Pseudo-mask candidate pool is empty after applying user filters.")

    if user_limit is not None and user_limit < incident_pairs["user_id"].nunique():
        rng = np.random.default_rng(seed)
        sampled_users = rng.choice(
            np.sort(incident_pairs["user_id"].unique()),
            size=user_limit,
            replace=False,
        )
        incident_pairs = incident_pairs[incident_pairs["user_id"].isin(sampled_users)].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    mask_size = max(1, int(len(incident_pairs) * mask_ratio))
    if mask_style == "uniform":
        probabilities = None
    else:
        user_ids = np.sort(incident_pairs["user_id"].unique())
        if user_heterogeneity > 0.0:
            user_noise = pd.Series(
                rng.lognormal(mean=0.0, sigma=user_heterogeneity, size=len(user_ids)),
                index=user_ids,
                dtype="float64",
            )
        else:
            user_noise = pd.Series(1.0, index=user_ids, dtype="float64")
        duration_seconds = max((incident_end_ts - incident_start_ts).total_seconds(), 1.0)
        time_progress = (
            (incident_pairs["last_ts"] - incident_start_ts).dt.total_seconds() / duration_seconds
        ).clip(lower=0.0, upper=1.0)
        time_weights = 0.7 + 0.6 * np.power(time_progress.to_numpy(dtype="float64"), time_bias_power)
        type_weights = np.where(incident_pairs["last_event_type"].eq(2), 1.05, 0.95)
        rating_weights = 0.9 + 0.02 * incident_pairs["max_rating"].fillna(0.0).clip(0.0, 10.0)
        probabilities = (
            incident_pairs["user_id"].map(user_noise).to_numpy(dtype="float64")
            * time_weights
            * type_weights
            * rating_weights.to_numpy(dtype="float64")
        )
        probabilities = np.maximum(probabilities, 1e-12)
        probabilities /= probabilities.sum()
    mask_index = rng.choice(len(incident_pairs), size=mask_size, replace=False, p=probabilities)
    masked = incident_pairs.iloc[mask_index].copy().reset_index(drop=True)
    masked = masked[["user_id", "edition_id"]].copy()
    masked["is_lost"] = 1
    return masked


def make_uniform_pseudo_mask(
    positives: pd.DataFrame,
    incident_start: str,
    incident_end: str,
    mask_ratio: float,
    seed: int,
    user_limit: int | None = None,
) -> pd.DataFrame:
    return make_pseudo_mask(
        positives=positives,
        incident_start=incident_start,
        incident_end=incident_end,
        mask_ratio=mask_ratio,
        seed=seed,
        user_limit=user_limit,
        mask_style="uniform",
    )


def drop_masked_pairs(positives: pd.DataFrame, masked_pairs: pd.DataFrame) -> pd.DataFrame:
    observed = positives.merge(
        masked_pairs[["user_id", "edition_id"]].assign(_masked=1),
        on=["user_id", "edition_id"],
        how="left",
    )
    return observed[observed["_masked"].isna()].drop(columns=["_masked"]).reset_index(drop=True)


def ndcg_at_k(predicted: list[int], relevant: set[int], k: int = 20) -> float:
    dcg = 0.0
    for rank, edition_id in enumerate(predicted[:k], start=1):
        if edition_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def validate_submission_frame(submission: pd.DataFrame, targets: pd.DataFrame, top_k: int = 20) -> None:
    required = {"user_id", "edition_id", "rank"}
    if not required.issubset(submission.columns):
        raise ValueError(f"Submission must contain columns {sorted(required)}")

    expected_users = set(targets["user_id"].astype("int64").tolist())
    got_users = set(submission["user_id"].astype("int64").tolist())
    missing = expected_users - got_users
    extra = got_users - expected_users
    if missing:
        raise ValueError(f"Missing target users: {len(missing)}")
    if extra:
        raise ValueError(f"Unexpected users in submission: {len(extra)}")

    for user_id, group in submission.groupby("user_id"):
        if len(group) != top_k:
            raise ValueError(f"user_id={user_id}: expected {top_k} rows, got {len(group)}")
        ranks = group["rank"].astype(int).tolist()
        editions = group["edition_id"].astype("int64").tolist()
        if sorted(ranks) != list(range(1, top_k + 1)):
            raise ValueError(f"user_id={user_id}: ranks must be 1..{top_k} without gaps")
        if len(set(editions)) != top_k:
            raise ValueError(f"user_id={user_id}: edition_id values must be unique")


def make_ranked_pref_map(
    frame: pd.DataFrame,
    entity_column: str,
    score_column: str,
) -> dict[int, list[int]]:
    if frame.empty or entity_column not in frame.columns:
        return {}
    ranked = (
        frame.dropna(subset=[entity_column])
        .sort_values(["user_id", score_column, entity_column], ascending=[True, False, True])
        .groupby("user_id")[entity_column]
        .agg(list)
        .to_dict()
    )
    return {
        int(user_id): [int(entity_id) for entity_id in values if pd.notna(entity_id)]
        for user_id, values in ranked.items()
    }


def blend_preference_maps(
    primary: dict[int, list[int]],
    secondary: dict[int, list[int]],
) -> dict[int, list[int]]:
    blended: dict[int, list[int]] = {}
    for user_id in set(primary) | set(secondary):
        ordered = list(dict.fromkeys(primary.get(user_id, []) + secondary.get(user_id, [])))
        blended[int(user_id)] = ordered
    return blended


def merge_preference_maps(*ordered_maps: dict[int, list[int]]) -> dict[int, list[int]]:
    merged: dict[int, list[int]] = {}
    user_ids: set[int] = set()
    for pref_map in ordered_maps:
        user_ids.update(int(user_id) for user_id in pref_map)
    for user_id in user_ids:
        ordered: list[int] = []
        for pref_map in ordered_maps:
            ordered.extend(pref_map.get(user_id, []))
        merged[int(user_id)] = list(dict.fromkeys(ordered))
    return merged


class CandidateBuilder:
    def __init__(
        self,
        pairs: pd.DataFrame,
        editions: pd.DataFrame,
        book_genres: pd.DataFrame,
        users: pd.DataFrame,
        max_ts: pd.Timestamp,
        limits: SourceLimits,
        svd_cfg: SvdConfig,
        als_cfg: AlsConfig,
        quiet: bool,
    ) -> None:
        self.pairs = pairs.copy()
        self.editions = editions.copy()
        self.book_genres = book_genres.copy()
        self.users = users.copy()
        self.max_ts = max_ts
        self.limits = limits
        self.svd_cfg = svd_cfg
        self.als_cfg = als_cfg
        self.quiet = quiet

        self.user_ids = np.sort(self.pairs["user_id"].unique())
        self.item_ids = np.sort(self.pairs["edition_id"].unique())
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.item_to_idx = {edition_id: idx for idx, edition_id in enumerate(self.item_ids)}

        self._build_tables()

    def _build_tables(self) -> None:
        log("Preparing item/user statistics...", self.quiet)
        pre_window_start = OFFICIAL_INCIDENT_START_TS - pd.Timedelta(days=30)
        pre_pairs = self.pairs[
            (self.pairs["last_ts"] >= pre_window_start)
            & (self.pairs["last_ts"] < OFFICIAL_INCIDENT_START_TS)
        ].copy()
        incident_pairs = self.pairs[
            (self.pairs["last_ts"] >= OFFICIAL_INCIDENT_START_TS)
            & (self.pairs["last_ts"] < OFFICIAL_INCIDENT_END_TS)
        ].copy()
        post_pairs = self.pairs[self.pairs["last_ts"] >= OFFICIAL_INCIDENT_END_TS].copy()

        self.user_stats = (
            self.pairs.groupby("user_id", as_index=False)
            .agg(
                user_seen=("edition_id", "count"),
                user_weight_sum=("pair_weight", "sum"),
                user_recent_cnt=("days_since_last", lambda x: int((x <= 30).sum())),
                user_avg_gap=("days_since_last", "mean"),
                user_read_seen=("last_event_type", lambda x: int((x == 2).sum())),
                user_wishlist_seen=("last_event_type", lambda x: int((x == 1).sum())),
                user_mean_rating=("max_rating", "mean"),
                user_high_rating_cnt=("max_rating", lambda x: int((x.fillna(0) >= 8).sum())),
            )
            .reset_index(drop=True)
        )
        self.user_pre_stats = (
            pre_pairs.groupby("user_id", as_index=False)
            .agg(
                user_pre_seen=("edition_id", "count"),
                user_pre_weight_sum=("pair_weight", "sum"),
                user_pre_read_seen=("last_event_type", lambda x: int((x == 2).sum())),
            )
            .reset_index(drop=True)
        )
        self.user_incident_stats = (
            incident_pairs.groupby("user_id", as_index=False)
            .agg(
                user_incident_seen=("edition_id", "count"),
                user_incident_weight_sum=("pair_weight", "sum"),
                user_incident_read_seen=("last_event_type", lambda x: int((x == 2).sum())),
            )
            .reset_index(drop=True)
        )
        self.user_post_stats = (
            post_pairs.groupby("user_id", as_index=False)
            .agg(
                user_post_seen=("edition_id", "count"),
                user_post_weight_sum=("pair_weight", "sum"),
                user_post_read_seen=("last_event_type", lambda x: int((x == 2).sum())),
            )
            .reset_index(drop=True)
        )

        self.item_stats = (
            self.pairs.groupby("edition_id", as_index=False)
            .agg(
                item_pop=("user_id", "count"),
                item_weight_sum=("pair_weight", "sum"),
                item_read_pop=("last_event_type", lambda x: int((x == 2).sum())),
                item_wishlist_pop=("last_event_type", lambda x: int((x == 1).sum())),
                item_mean_rating=("max_rating", "mean"),
                item_high_rating_cnt=("max_rating", lambda x: int((x.fillna(0) >= 8).sum())),
            )
            .reset_index(drop=True)
        )
        self.recent_item_stats = (
            self.pairs[self.pairs["days_since_last"] <= 30]
            .groupby("edition_id", as_index=False)
            .agg(
                item_recent_pop=("user_id", "count"),
                item_recent_read_pop=("last_event_type", lambda x: int((x == 2).sum())),
                item_recent_wishlist_pop=("last_event_type", lambda x: int((x == 1).sum())),
            )
            .reset_index(drop=True)
        )
        self.pre_item_stats = (
            pre_pairs.groupby("edition_id", as_index=False)
            .agg(
                item_pre_pop=("user_id", "count"),
                item_pre_weight_sum=("pair_weight", "sum"),
                item_pre_read_pop=("last_event_type", lambda x: int((x == 2).sum())),
                item_pre_wishlist_pop=("last_event_type", lambda x: int((x == 1).sum())),
            )
            .reset_index(drop=True)
        )
        self.incident_item_stats = (
            incident_pairs.groupby("edition_id", as_index=False)
            .agg(
                item_incident_pop=("user_id", "count"),
                item_incident_weight_sum=("pair_weight", "sum"),
                item_incident_read_pop=("last_event_type", lambda x: int((x == 2).sum())),
                item_incident_wishlist_pop=("last_event_type", lambda x: int((x == 1).sum())),
            )
            .reset_index(drop=True)
        )
        self.post_item_stats = (
            post_pairs.groupby("edition_id", as_index=False)
            .agg(
                item_post_pop=("user_id", "count"),
                item_post_weight_sum=("pair_weight", "sum"),
                item_post_read_pop=("last_event_type", lambda x: int((x == 2).sum())),
                item_post_wishlist_pop=("last_event_type", lambda x: int((x == 1).sum())),
            )
            .reset_index(drop=True)
        )

        self.seen_by_user = self.pairs.groupby("user_id")["edition_id"].agg(set).to_dict()
        self.history_by_user = (
            self.pairs.sort_values(["user_id", "pair_weight"], ascending=[True, False])
            .groupby("user_id")[["edition_id", "pair_weight"]]
            .apply(lambda group: list(zip(group["edition_id"], group["pair_weight"])))
            .to_dict()
        )

        self.user_book = (
            self.pairs[["user_id", "edition_id", "pair_weight", "days_since_last"]]
            .merge(self.editions[["edition_id", "book_id"]], on="edition_id", how="left")
            .groupby(["user_id", "book_id"], as_index=False)
            .agg(
                book_aff=("pair_weight", "sum"),
                book_last_gap=("days_since_last", "min"),
            )
        )
        self.user_book_pre = (
            pre_pairs[["user_id", "edition_id", "pair_weight", "days_since_last"]]
            .merge(self.editions[["edition_id", "book_id"]], on="edition_id", how="left")
            .groupby(["user_id", "book_id"], as_index=False)
            .agg(
                book_pre_aff=("pair_weight", "sum"),
                book_pre_last_gap=("days_since_last", "min"),
            )
        )
        self.user_book_incident = (
            incident_pairs[["user_id", "edition_id", "pair_weight", "days_since_last"]]
            .merge(self.editions[["edition_id", "book_id"]], on="edition_id", how="left")
            .groupby(["user_id", "book_id"], as_index=False)
            .agg(
                book_incident_aff=("pair_weight", "sum"),
                book_incident_last_gap=("days_since_last", "min"),
            )
        )
        self.user_book_post = (
            post_pairs[["user_id", "edition_id", "pair_weight"]]
            .merge(self.editions[["edition_id", "book_id"]], on="edition_id", how="left")
            .groupby(["user_id", "book_id"], as_index=False)
            .agg(book_post_aff=("pair_weight", "sum"))
        )

        pairs_meta = self.pairs.merge(self.editions, on="edition_id", how="left")
        pairs_meta = pairs_meta.merge(self.book_genres, on="book_id", how="left")
        pre_pairs_meta = pre_pairs.merge(self.editions, on="edition_id", how="left")
        pre_pairs_meta = pre_pairs_meta.merge(self.book_genres, on="book_id", how="left")
        incident_pairs_meta = incident_pairs.merge(self.editions, on="edition_id", how="left")
        incident_pairs_meta = incident_pairs_meta.merge(self.book_genres, on="book_id", how="left")
        post_pairs_meta = post_pairs.merge(self.editions, on="edition_id", how="left")
        post_pairs_meta = post_pairs_meta.merge(self.book_genres, on="book_id", how="left")

        self.user_author = (
            pairs_meta.groupby(["user_id", "author_id"], as_index=False)
            .agg(
                author_aff=("pair_weight", "sum"),
                author_last_gap=("days_since_last", "min"),
            )
            .reset_index(drop=True)
        )
        self.user_author_pre = (
            pre_pairs_meta.groupby(["user_id", "author_id"], as_index=False)
            .agg(author_pre_aff=("pair_weight", "sum"))
            .reset_index(drop=True)
        )
        self.user_author_incident = (
            incident_pairs_meta.groupby(["user_id", "author_id"], as_index=False)
            .agg(author_incident_aff=("pair_weight", "sum"))
            .reset_index(drop=True)
        )
        self.user_author_post = (
            post_pairs_meta.groupby(["user_id", "author_id"], as_index=False)
            .agg(author_post_aff=("pair_weight", "sum"))
            .reset_index(drop=True)
        )
        self.user_genre = (
            pairs_meta.groupby(["user_id", "genre_id"], as_index=False)
            .agg(
                genre_aff=("pair_weight", "sum"),
                genre_last_gap=("days_since_last", "min"),
            )
            .reset_index(drop=True)
        )
        self.user_genre_pre = (
            pre_pairs_meta.groupby(["user_id", "genre_id"], as_index=False)
            .agg(genre_pre_aff=("pair_weight", "sum"))
            .reset_index(drop=True)
        )
        self.user_genre_incident = (
            incident_pairs_meta.groupby(["user_id", "genre_id"], as_index=False)
            .agg(genre_incident_aff=("pair_weight", "sum"))
            .reset_index(drop=True)
        )
        self.user_genre_post = (
            post_pairs_meta.groupby(["user_id", "genre_id"], as_index=False)
            .agg(genre_post_aff=("pair_weight", "sum"))
            .reset_index(drop=True)
        )
        self.user_publisher = (
            pairs_meta.groupby(["user_id", "publisher_id"], as_index=False)
            .agg(pub_aff=("pair_weight", "sum"))
            .reset_index(drop=True)
        )
        self.user_publisher_pre = (
            pre_pairs_meta.groupby(["user_id", "publisher_id"], as_index=False)
            .agg(pub_pre_aff=("pair_weight", "sum"))
            .reset_index(drop=True)
        )
        self.user_publisher_incident = (
            incident_pairs_meta.groupby(["user_id", "publisher_id"], as_index=False)
            .agg(pub_incident_aff=("pair_weight", "sum"))
            .reset_index(drop=True)
        )
        self.user_publisher_post = (
            post_pairs_meta.groupby(["user_id", "publisher_id"], as_index=False)
            .agg(pub_post_aff=("pair_weight", "sum"))
            .reset_index(drop=True)
        )
        self.user_language = (
            pairs_meta.groupby(["user_id", "language_id"], as_index=False)
            .agg(lang_aff=("pair_weight", "sum"))
            .reset_index(drop=True)
        )
        self.user_language_pre = (
            pre_pairs_meta.groupby(["user_id", "language_id"], as_index=False)
            .agg(lang_pre_aff=("pair_weight", "sum"))
            .reset_index(drop=True)
        )
        self.user_language_incident = (
            incident_pairs_meta.groupby(["user_id", "language_id"], as_index=False)
            .agg(lang_incident_aff=("pair_weight", "sum"))
            .reset_index(drop=True)
        )
        self.user_language_post = (
            post_pairs_meta.groupby(["user_id", "language_id"], as_index=False)
            .agg(lang_post_aff=("pair_weight", "sum"))
            .reset_index(drop=True)
        )

        self.user_book_pref = merge_preference_maps(
            make_ranked_pref_map(self.user_book_incident, "book_id", "book_incident_aff"),
            make_ranked_pref_map(self.user_book_pre, "book_id", "book_pre_aff"),
            make_ranked_pref_map(self.user_book, "book_id", "book_aff"),
            make_ranked_pref_map(self.user_book_post, "book_id", "book_post_aff"),
        )
        self.user_author_pref = merge_preference_maps(
            make_ranked_pref_map(self.user_author_incident, "author_id", "author_incident_aff"),
            make_ranked_pref_map(self.user_author_pre, "author_id", "author_pre_aff"),
            make_ranked_pref_map(self.user_author, "author_id", "author_aff"),
            make_ranked_pref_map(self.user_author_post, "author_id", "author_post_aff"),
        )
        self.user_genre_pref = merge_preference_maps(
            make_ranked_pref_map(self.user_genre_incident, "genre_id", "genre_incident_aff"),
            make_ranked_pref_map(self.user_genre_pre, "genre_id", "genre_pre_aff"),
            make_ranked_pref_map(self.user_genre, "genre_id", "genre_aff"),
            make_ranked_pref_map(self.user_genre_post, "genre_id", "genre_post_aff"),
        )
        self.user_publisher_pref = merge_preference_maps(
            make_ranked_pref_map(
                self.user_publisher_incident,
                "publisher_id",
                "pub_incident_aff",
            ),
            make_ranked_pref_map(self.user_publisher_pre, "publisher_id", "pub_pre_aff"),
            make_ranked_pref_map(self.user_publisher, "publisher_id", "pub_aff"),
            make_ranked_pref_map(self.user_publisher_post, "publisher_id", "pub_post_aff"),
        )
        self.user_language_pref = merge_preference_maps(
            make_ranked_pref_map(
                self.user_language_incident,
                "language_id",
                "lang_incident_aff",
            ),
            make_ranked_pref_map(self.user_language_pre, "language_id", "lang_pre_aff"),
            make_ranked_pref_map(self.user_language, "language_id", "lang_aff"),
            make_ranked_pref_map(self.user_language_post, "language_id", "lang_post_aff"),
        )

        item_rank_frame = (
            self.item_stats.merge(self.recent_item_stats, on="edition_id", how="left")
            .merge(self.pre_item_stats, on="edition_id", how="left")
            .merge(self.incident_item_stats, on="edition_id", how="left")
            .merge(self.post_item_stats, on="edition_id", how="left")
            .fillna(
                {
                    "item_recent_pop": 0.0,
                    "item_recent_read_pop": 0.0,
                    "item_recent_wishlist_pop": 0.0,
                    "item_pre_pop": 0.0,
                    "item_pre_weight_sum": 0.0,
                    "item_pre_read_pop": 0.0,
                    "item_pre_wishlist_pop": 0.0,
                    "item_incident_pop": 0.0,
                    "item_incident_weight_sum": 0.0,
                    "item_incident_read_pop": 0.0,
                    "item_incident_wishlist_pop": 0.0,
                    "item_post_pop": 0.0,
                    "item_post_weight_sum": 0.0,
                    "item_post_read_pop": 0.0,
                    "item_post_wishlist_pop": 0.0,
                }
            )
        )
        self.popular_incident = (
            item_rank_frame.sort_values(
                [
                    "item_incident_read_pop",
                    "item_incident_pop",
                    "item_pre_read_pop",
                    "item_read_pop",
                    "edition_id",
                ],
                ascending=[False, False, False, False, True],
            )["edition_id"]
            .astype("int64")
            .tolist()
        )
        self.popular_all = (
            item_rank_frame.sort_values(
                [
                    "item_read_pop",
                    "item_incident_read_pop",
                    "item_pre_read_pop",
                    "item_weight_sum",
                    "item_pop",
                    "edition_id",
                ],
                ascending=[False, False, False, False, False, True],
            )["edition_id"]
            .astype("int64")
            .tolist()
        )
        self.popular_recent = (
            item_rank_frame.sort_values(
                [
                    "item_incident_read_pop",
                    "item_incident_pop",
                    "item_pre_read_pop",
                    "item_recent_read_pop",
                    "item_recent_pop",
                    "item_post_read_pop",
                    "edition_id",
                ],
                ascending=[False, False, False, False, False, False, True],
            )["edition_id"]
            .astype("int64")
            .tolist()
        )

        self.author_items = self._build_entity_items(entity_column="author_id", topk=self.limits.entity_items_cap)
        self.genre_items = self._build_entity_items(entity_column="genre_id", topk=self.limits.entity_items_cap)
        self.publisher_items = self._build_entity_items(
            entity_column="publisher_id",
            topk=self.limits.entity_items_cap,
        )
        self.language_items = self._build_entity_items(
            entity_column="language_id",
            topk=self.limits.entity_items_cap,
        )
        self.book_items = self._build_entity_items(entity_column="book_id", topk=self.limits.entity_items_cap)

        self._fit_svd()
        self._fit_nmf()
        self._fit_als()
        self._fit_cooccurrence()
        self._build_item_svd_index()
        self._build_user_knn_index()

    def _build_entity_items(self, entity_column: str, topk: int) -> dict[int, list[int]]:
        if entity_column == "genre_id":
            frame = self.editions[["edition_id", "book_id"]].merge(
                self.book_genres,
                on="book_id",
                how="inner",
            )
        else:
            frame = self.editions[["edition_id", entity_column]].copy()

        frame = (
            frame.merge(self.item_stats, on="edition_id", how="left")
            .merge(self.pre_item_stats, on="edition_id", how="left")
            .merge(self.incident_item_stats, on="edition_id", how="left")
            .fillna(
                {
                    "item_pop": 0.0,
                    "item_weight_sum": 0.0,
                    "item_read_pop": 0.0,
                    "item_pre_pop": 0.0,
                    "item_pre_read_pop": 0.0,
                    "item_incident_pop": 0.0,
                    "item_incident_read_pop": 0.0,
                }
            )
        )
        frame = frame.sort_values(
            [
                entity_column,
                "item_incident_read_pop",
                "item_incident_pop",
                "item_pre_read_pop",
                "item_read_pop",
                "item_weight_sum",
                "item_pop",
                "edition_id",
            ],
            ascending=[True, False, False, False, False, False, False, True],
        )
        mapping: dict[int, list[int]] = {}
        for entity_id, group in frame.groupby(entity_column):
            mapping[int(entity_id)] = group["edition_id"].astype("int64").head(topk).tolist()
        return mapping

    def _fit_svd(self) -> None:
        log("Fitting TruncatedSVD retrieval...", self.quiet)
        n_components = min(
            self.svd_cfg.components,
            max(8, min(len(self.user_ids) - 1, len(self.item_ids) - 1)),
        )
        matrix = sparse.csr_matrix(
            (
                self.pairs["pair_weight"].astype("float32").to_numpy(),
                (
                    self.pairs["user_id"].map(self.user_to_idx).to_numpy(),
                    self.pairs["edition_id"].map(self.item_to_idx).to_numpy(),
                ),
            ),
            shape=(len(self.user_ids), len(self.item_ids)),
            dtype=np.float32,
        )
        self.user_item_matrix = matrix
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd_user_factors = svd.fit_transform(matrix).astype("float32")
        self.svd_item_factors = svd.components_.T.astype("float32")
        self.svd_item_ids = self.item_ids.astype("int64")

    def _fit_nmf(self) -> None:
        log("Fitting NMF retrieval...", self.quiet)
        n_components = min(64, max(8, min(len(self.user_ids) - 1, len(self.item_ids) - 1)))
        try:
            nmf = NMF(n_components=n_components, random_state=42, max_iter=200, init="nndsvda")
            self.nmf_user_factors = nmf.fit_transform(self.user_item_matrix).astype("float32")
            self.nmf_item_factors = nmf.components_.T.astype("float32")
            norms = np.linalg.norm(self.nmf_item_factors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            self.nmf_item_factors_normed = (self.nmf_item_factors / norms).astype("float32")
        except Exception as exc:
            log(f"NMF fitting failed: {exc}", self.quiet)
            self.nmf_user_factors = None
            self.nmf_item_factors = None
            self.nmf_item_factors_normed = None

    def _fit_als(self) -> None:
        self.als_model = None
        if not self.als_cfg.enabled:
            return

        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError:
            log("implicit is not installed, skipping ALS source.", self.quiet)
            return

        log("Fitting implicit ALS retrieval...", self.quiet)
        user_item = (self.user_item_matrix * self.als_cfg.alpha).tocsr()
        model = AlternatingLeastSquares(
            factors=self.als_cfg.factors,
            regularization=self.als_cfg.regularization,
            iterations=self.als_cfg.iterations,
            random_state=42,
            use_gpu=self.als_cfg.use_gpu,
        )
        model.fit(user_item)
        self.als_model = model

    def _fit_cooccurrence(self) -> None:
        log("Building item-item co-occurrence source...", self.quiet)
        item_popularity = self.pairs.groupby("edition_id").size().to_dict()
        co_counts: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))

        for history in self.history_by_user.values():
            clipped = history[: self.limits.user_history_cap]
            for idx_a in range(len(clipped)):
                item_a, weight_a = clipped[idx_a]
                for idx_b in range(idx_a + 1, len(clipped)):
                    item_b, weight_b = clipped[idx_b]
                    increment = min(float(weight_a), float(weight_b))
                    co_counts[int(item_a)][int(item_b)] += increment
                    co_counts[int(item_b)][int(item_a)] += increment

        self.cooc_sims: dict[int, list[tuple[int, float]]] = {}
        for item_id, neighbours in co_counts.items():
            pop_a = max(1.0, float(item_popularity.get(item_id, 1)))
            scored = [
                (
                    other_id,
                    value / ((pop_a * max(1.0, float(item_popularity.get(other_id, 1)))) ** 0.35),
                )
                for other_id, value in neighbours.items()
            ]
            scored.sort(key=lambda pair: (-pair[1], pair[0]))
            self.cooc_sims[item_id] = scored[: self.limits.cooc_topk]

    def _build_item_svd_index(self) -> None:
        log("Building item-item SVD similarity index...", self.quiet)
        norms = np.linalg.norm(self.svd_item_factors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.svd_item_factors_normed = (self.svd_item_factors / norms).astype("float32")
        self.user_item_matrix_t = self.user_item_matrix.T.tocsr()
        item_pop = np.asarray(self.user_item_matrix.getnnz(axis=0)).flatten().astype("float32")
        self.item_pop_sqrt = np.sqrt(np.maximum(item_pop, 1.0))

    def _build_user_knn_index(self) -> None:
        log("Building user-user KNN index...", self.quiet)
        norms = np.linalg.norm(self.svd_user_factors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.svd_user_factors_normed = (self.svd_user_factors / norms).astype("float32")
        incident_pairs = self.pairs[
            (self.pairs["last_ts"] >= OFFICIAL_INCIDENT_START_TS)
            & (self.pairs["last_ts"] < OFFICIAL_INCIDENT_END_TS)
        ]
        pre_window = self.pairs[
            (self.pairs["last_ts"] >= OFFICIAL_INCIDENT_START_TS - pd.Timedelta(days=30))
            & (self.pairs["last_ts"] < OFFICIAL_INCIDENT_START_TS)
        ]
        self.user_incident_items: dict[int, list[tuple[int, float]]] = {}
        for user_id, group in incident_pairs.groupby("user_id"):
            items = list(zip(
                group["edition_id"].astype("int64").tolist(),
                group["pair_weight"].astype("float64").tolist(),
            ))
            items.sort(key=lambda x: -x[1])
            self.user_incident_items[int(user_id)] = items[:self.limits.user_history_cap]
        self.user_pre_items: dict[int, list[tuple[int, float]]] = {}
        for user_id, group in pre_window.groupby("user_id"):
            items = list(zip(
                group["edition_id"].astype("int64").tolist(),
                (group["pair_weight"].astype("float64") * 0.5).tolist(),
            ))
            items.sort(key=lambda x: -x[1])
            self.user_pre_items[int(user_id)] = items[:self.limits.user_history_cap]

    def _item_svd_source(self, user_id: int, seen_u: set[int]) -> list[tuple[int, float, float]]:
        """Find items similar to user's history items using SVD item embeddings."""
        history = self.history_by_user.get(user_id, [])[:self.limits.user_history_cap]
        if not history:
            return []
        indices = []
        weights = []
        for item_id, weight in history:
            if item_id in self.item_to_idx:
                indices.append(self.item_to_idx[item_id])
                weights.append(float(weight))
        if not indices:
            return []
        weight_arr = np.array(weights, dtype="float32")
        weight_arr /= weight_arr.sum()
        user_profile = (self.svd_item_factors_normed[indices].T @ weight_arr).astype("float32")
        norm = np.linalg.norm(user_profile)
        if norm < 1e-8:
            return []
        user_profile /= norm
        scores = self.svd_item_factors_normed @ user_profile
        seen_indices = [self.item_to_idx[item] for item in seen_u if item in self.item_to_idx]
        if seen_indices:
            scores[seen_indices] = -1e9
        topk = min(self.limits.item_svd_topk, len(scores))
        top_idx = np.argpartition(-scores, topk - 1)[:topk]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [
            (int(self.svd_item_ids[idx]), float(scores[idx]), 1.0 / rank)
            for rank, idx in enumerate(top_idx, start=1)
            if scores[idx] > 0
        ]

    def _user_knn_source(self, user_id: int, seen_u: set[int]) -> list[tuple[int, float, float]]:
        """Find similar users and recommend their incident + pre-incident items."""
        if user_id not in self.user_to_idx:
            return []
        user_idx = self.user_to_idx[user_id]
        user_vec = self.svd_user_factors_normed[user_idx]
        sims = self.svd_user_factors_normed @ user_vec
        sims[user_idx] = -1e9
        n_neighbors = min(self.limits.user_knn_neighbors, len(sims) - 1)
        if n_neighbors <= 0:
            return []
        top_idx = np.argpartition(-sims, n_neighbors - 1)[:n_neighbors]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        item_scores: dict[int, float] = {}
        for neighbor_idx in top_idx:
            sim = float(sims[neighbor_idx])
            if sim <= 0:
                continue
            neighbor_id = int(self.user_ids[neighbor_idx])
            for item_id, weight in self.user_incident_items.get(neighbor_id, []):
                if item_id in seen_u:
                    continue
                item_scores[item_id] = item_scores.get(item_id, 0.0) + sim * weight
            for item_id, weight in self.user_pre_items.get(neighbor_id, []):
                if item_id in seen_u:
                    continue
                item_scores[item_id] = item_scores.get(item_id, 0.0) + sim * weight
        if not item_scores:
            return []
        ranked = sorted(item_scores.items(), key=lambda x: (-x[1], x[0]))
        return [
            (item_id, score, 1.0 / rank)
            for rank, (item_id, score) in enumerate(ranked[:self.limits.user_knn_topk], start=1)
        ]

    def _svd_source(self, user_id: int, seen_u: set[int]) -> list[tuple[int, float, float]]:
        if user_id not in self.user_to_idx:
            return []
        user_idx = self.user_to_idx[user_id]
        scores = self.svd_user_factors[user_idx] @ self.svd_item_factors.T
        scores = scores.copy()
        seen_indices = [self.item_to_idx[item] for item in seen_u if item in self.item_to_idx]
        if seen_indices:
            scores[seen_indices] = -1e9
        topk = min(self.limits.svd_topk, len(scores))
        top_idx = np.argpartition(-scores, topk - 1)[:topk]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [
            (
                int(self.svd_item_ids[item_idx]),
                float(scores[item_idx]),
                1.0 / rank,
            )
            for rank, item_idx in enumerate(top_idx, start=1)
        ]

    def _nmf_source(self, user_id: int, seen_u: set[int]) -> list[tuple[int, float, float]]:
        """Find items using NMF user-item factorization (complementary to SVD)."""
        if self.nmf_user_factors is None or user_id not in self.user_to_idx:
            return []
        user_idx = self.user_to_idx[user_id]
        user_vec = self.nmf_user_factors[user_idx]
        scores = self.nmf_item_factors @ user_vec
        scores = scores.copy()
        seen_indices = [self.item_to_idx[item] for item in seen_u if item in self.item_to_idx]
        if seen_indices:
            scores[seen_indices] = -1e9
        topk = min(self.limits.svd_topk, len(scores))
        top_idx = np.argpartition(-scores, topk - 1)[:topk]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [
            (int(self.svd_item_ids[item_idx]), float(scores[item_idx]), 1.0 / rank)
            for rank, item_idx in enumerate(top_idx, start=1)
            if scores[item_idx] > 0
        ]

    def _als_source(self, user_id: int) -> list[tuple[int, float, float]]:
        if self.als_model is None or user_id not in self.user_to_idx:
            return []
        user_idx = self.user_to_idx[user_id]
        try:
            items, scores = self.als_model.recommend(
                userid=user_idx,
                user_items=self.user_item_matrix[user_idx],
                N=self.limits.als_topk,
                filter_already_liked_items=True,
                recalculate_user=False,
            )
        except ValueError as exc:
            log(f"ALS recommend failed for user_id={user_id}, skipping ALS source: {exc}", self.quiet)
            return []
        return [
            (
                int(self.item_ids[item_idx]),
                float(score),
                1.0 / rank,
            )
            for rank, (item_idx, score) in enumerate(zip(items.tolist(), scores.tolist()), start=1)
        ]

    def _cooc_source(self, user_id: int, seen_u: set[int]) -> list[tuple[int, float, float]]:
        score_by_item: dict[int, float] = {}
        history = self.history_by_user.get(user_id, [])[: self.limits.user_history_cap]
        for source_item, pair_weight in history:
            for candidate_item, similarity in self.cooc_sims.get(int(source_item), []):
                if candidate_item in seen_u:
                    continue
                score_by_item[candidate_item] = score_by_item.get(candidate_item, 0.0) + (
                    float(pair_weight) * similarity
                )
        ranked = sorted(score_by_item.items(), key=lambda pair: (-pair[1], pair[0]))
        return [
            (candidate_item, score, 1.0 / rank)
            for rank, (candidate_item, score) in enumerate(
                ranked[: self.limits.cooc_topk],
                start=1,
            )
        ]

    def _entity_source(
        self,
        user_pref: dict[int, list[int]],
        entity_items: dict[int, list[int]],
        user_id: int,
        seen_u: set[int],
        source_topk: int,
    ) -> list[tuple[int, float]]:
        pref_ids = user_pref.get(user_id, [])[: self.limits.entity_pref_cap]
        candidates: list[int] = []
        for entity_id in pref_ids:
            candidates.extend(entity_items.get(int(entity_id), [])[: self.limits.entity_items_cap])
        unique_candidates = [item for item in dict.fromkeys(candidates) if item not in seen_u]
        return [
            (candidate_item, 1.0 / rank)
            for rank, candidate_item in enumerate(unique_candidates[:source_topk], start=1)
        ]

    def _popularity_source(
        self,
        seen_u: set[int],
        ranked_items: list[int],
        topk: int,
    ) -> list[tuple[int, float]]:
        filtered = [item for item in ranked_items if item not in seen_u][:topk]
        return [(candidate_item, 1.0 / rank) for rank, candidate_item in enumerate(filtered, start=1)]

    def _score_snapshot(self, features: dict[str, float]) -> float:
        return (
            0.22 * features.get("svd_rank_inv", 0.0)
            + 0.16 * features.get("cooc_rank_inv", 0.0)
            + 0.10 * features.get("als_rank_inv", 0.0)
            + 0.12 * features.get("item_svd_rank_inv", 0.0)
            + 0.10 * features.get("user_knn_rank_inv", 0.0)
            + 0.06 * features.get("book_rank_inv", 0.0)
            + 0.06 * features.get("author_rank_inv", 0.0)
            + 0.04 * features.get("genre_rank_inv", 0.0)
            + 0.02 * features.get("publisher_rank_inv", 0.0)
            + 0.01 * features.get("language_rank_inv", 0.0)
            + 0.06 * features.get("incident_pop_rank_inv", 0.0)
            + 0.03 * features.get("recent_pop_rank_inv", 0.0)
            + 0.02 * features.get("pop_rank_inv", 0.0)
        )

    def _full_score_user(self, user_id: int, seen_u: set[int]) -> np.ndarray:
        """Score ALL items using SVD + sparse 2-hop CF + item-SVD + user-KNN."""
        n_items = len(self.item_ids)
        scores = np.zeros(n_items, dtype="float32")

        if user_id not in self.user_to_idx:
            return scores

        user_idx = self.user_to_idx[user_id]

        svd_scores = self.svd_user_factors[user_idx] @ self.svd_item_factors.T
        scores += svd_scores * 0.30

        user_row = self.user_item_matrix[user_idx]
        user_sims = user_row @ self.user_item_matrix_t
        user_sims[0, user_idx] = 0.0
        sparse_cf = user_sims @ self.user_item_matrix
        sparse_cf_dense = np.asarray(sparse_cf.todense()).flatten().astype("float32")
        sparse_cf_dense /= np.maximum(self.item_pop_sqrt, 1.0)
        cf_max = sparse_cf_dense.max()
        if cf_max > 0:
            sparse_cf_dense /= cf_max
        scores += sparse_cf_dense * 0.25

        history = self.history_by_user.get(user_id, [])[:self.limits.user_history_cap]
        if history:
            indices = []
            weights = []
            for item_id, weight in history:
                if item_id in self.item_to_idx:
                    indices.append(self.item_to_idx[item_id])
                    weights.append(float(weight))
            if indices:
                weight_arr = np.array(weights, dtype="float32")
                weight_arr /= weight_arr.sum()
                user_profile = (self.svd_item_factors_normed[indices].T @ weight_arr).astype("float32")
                norm = np.linalg.norm(user_profile)
                if norm > 1e-8:
                    user_profile /= norm
                    item_svd_scores = self.svd_item_factors_normed @ user_profile
                    scores += item_svd_scores * 0.25

        user_vec = self.svd_user_factors_normed[user_idx]
        sims = self.svd_user_factors_normed @ user_vec
        sims[user_idx] = -1e9
        n_neighbors = min(self.limits.user_knn_neighbors, len(sims) - 1)
        if n_neighbors > 0:
            top_idx = np.argpartition(-sims, n_neighbors - 1)[:n_neighbors]
            for neighbor_idx in top_idx:
                sim = float(sims[neighbor_idx])
                if sim <= 0:
                    continue
                neighbor_id = int(self.user_ids[neighbor_idx])
                for item_id, weight in self.user_incident_items.get(neighbor_id, []):
                    if item_id in self.item_to_idx:
                        scores[self.item_to_idx[item_id]] += sim * weight * 0.20

        seen_indices = [self.item_to_idx[item] for item in seen_u if item in self.item_to_idx]
        if seen_indices:
            scores[seen_indices] = -1e9

        return scores

    def generate_candidate_frame(self, target_users: Iterable[int]) -> pd.DataFrame:
        log("Generating candidates...", self.quiet)
        records: list[dict[str, float | int]] = []
        target_users = list(dict.fromkeys(int(user_id) for user_id in target_users))

        for row_idx, user_id in enumerate(target_users, start=1):
            seen_u = self.seen_by_user.get(user_id, set())
            feature_by_item: dict[int, dict[str, float]] = defaultdict(dict)

            full_scores = self._full_score_user(user_id, seen_u)
            full_topk = min(self.limits.candidate_cap, int(np.sum(full_scores > -1e8)))
            if full_topk > 0:
                top_idx = np.argpartition(-full_scores, full_topk - 1)[:full_topk]
                top_idx = top_idx[np.argsort(-full_scores[top_idx])]
                for rank, idx in enumerate(top_idx, start=1):
                    item_id = int(self.svd_item_ids[idx])
                    feature_by_item[item_id]["full_score"] = float(full_scores[idx])
                    feature_by_item[item_id]["full_rank_inv"] = 1.0 / rank

            for item_id, raw_score, rank_inv in self._svd_source(user_id, seen_u):
                feature_by_item[item_id]["svd_score"] = raw_score
                feature_by_item[item_id]["svd_rank_inv"] = rank_inv

            for item_id, raw_score, rank_inv in self._als_source(user_id):
                feature_by_item[item_id]["als_score"] = raw_score
                feature_by_item[item_id]["als_rank_inv"] = rank_inv

            for item_id, raw_score, rank_inv in self._item_svd_source(user_id, seen_u):
                feature_by_item[item_id]["item_svd_score"] = raw_score
                feature_by_item[item_id]["item_svd_rank_inv"] = rank_inv

            for item_id, raw_score, rank_inv in self._user_knn_source(user_id, seen_u):
                feature_by_item[item_id]["user_knn_score"] = raw_score
                feature_by_item[item_id]["user_knn_rank_inv"] = rank_inv

            for item_id, raw_score, rank_inv in self._cooc_source(user_id, seen_u):
                feature_by_item[item_id]["cooc_score"] = raw_score
                feature_by_item[item_id]["cooc_rank_inv"] = rank_inv

            for item_id, rank_inv in self._entity_source(
                self.user_book_pref,
                self.book_items,
                user_id,
                seen_u,
                self.limits.book_topk,
            ):
                feature_by_item[item_id]["book_rank_inv"] = rank_inv

            for item_id, rank_inv in self._entity_source(
                self.user_author_pref,
                self.author_items,
                user_id,
                seen_u,
                self.limits.author_topk,
            ):
                feature_by_item[item_id]["author_rank_inv"] = rank_inv

            for item_id, rank_inv in self._entity_source(
                self.user_genre_pref,
                self.genre_items,
                user_id,
                seen_u,
                self.limits.genre_topk,
            ):
                feature_by_item[item_id]["genre_rank_inv"] = rank_inv

            for item_id, rank_inv in self._entity_source(
                self.user_publisher_pref,
                self.publisher_items,
                user_id,
                seen_u,
                self.limits.publisher_topk,
            ):
                feature_by_item[item_id]["publisher_rank_inv"] = rank_inv

            for item_id, rank_inv in self._entity_source(
                self.user_language_pref,
                self.language_items,
                user_id,
                seen_u,
                self.limits.language_topk,
            ):
                feature_by_item[item_id]["language_rank_inv"] = rank_inv

            for item_id, rank_inv in self._popularity_source(
                seen_u=seen_u,
                ranked_items=self.popular_incident,
                topk=self.limits.incident_pop_topk,
            ):
                feature_by_item[item_id]["incident_pop_rank_inv"] = rank_inv

            for item_id, rank_inv in self._popularity_source(
                seen_u=seen_u,
                ranked_items=self.popular_recent,
                topk=self.limits.recent_pop_topk,
            ):
                feature_by_item[item_id]["recent_pop_rank_inv"] = rank_inv

            for item_id, rank_inv in self._popularity_source(
                seen_u=seen_u,
                ranked_items=self.popular_all,
                topk=self.limits.pop_topk,
            ):
                feature_by_item[item_id]["pop_rank_inv"] = rank_inv

            ranked_items = sorted(
                feature_by_item.items(),
                key=lambda pair: (-self._score_snapshot(pair[1]), pair[0]),
            )[: self.limits.candidate_cap]

            for item_id, features in ranked_items:
                row: dict[str, float | int] = {"user_id": user_id, "edition_id": int(item_id)}
                row.update(features)
                records.append(row)

            if row_idx % 500 == 0:
                log(f"  processed users: {row_idx}/{len(target_users)}", self.quiet)

        frame = pd.DataFrame(records)
        if frame.empty:
            return frame

        numeric_cols = [col for col in frame.columns if col not in {"user_id", "edition_id"}]
        frame[numeric_cols] = frame[numeric_cols].fillna(0.0)
        frame = frame.groupby(["user_id", "edition_id"], as_index=False)[numeric_cols].max()
        return self._attach_features(frame)

    def _attach_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.merge(self.editions, on="edition_id", how="left")
        frame = frame.merge(self.user_stats, on="user_id", how="left")
        frame = frame.merge(self.user_pre_stats, on="user_id", how="left")
        frame = frame.merge(self.user_incident_stats, on="user_id", how="left")
        frame = frame.merge(self.user_post_stats, on="user_id", how="left")
        frame = frame.merge(self.item_stats, on="edition_id", how="left")
        frame = frame.merge(self.recent_item_stats, on="edition_id", how="left")
        frame = frame.merge(self.pre_item_stats, on="edition_id", how="left")
        frame = frame.merge(self.incident_item_stats, on="edition_id", how="left")
        frame = frame.merge(self.post_item_stats, on="edition_id", how="left")
        frame = frame.merge(self.users, on="user_id", how="left")
        frame = frame.merge(self.user_author, on=["user_id", "author_id"], how="left")
        frame = frame.merge(self.user_author_pre, on=["user_id", "author_id"], how="left")
        frame = frame.merge(self.user_author_incident, on=["user_id", "author_id"], how="left")
        frame = frame.merge(self.user_author_post, on=["user_id", "author_id"], how="left")
        frame = frame.merge(self.user_publisher, on=["user_id", "publisher_id"], how="left")
        frame = frame.merge(self.user_publisher_pre, on=["user_id", "publisher_id"], how="left")
        frame = frame.merge(self.user_publisher_incident, on=["user_id", "publisher_id"], how="left")
        frame = frame.merge(self.user_publisher_post, on=["user_id", "publisher_id"], how="left")
        frame = frame.merge(self.user_language, on=["user_id", "language_id"], how="left")
        frame = frame.merge(self.user_language_pre, on=["user_id", "language_id"], how="left")
        frame = frame.merge(self.user_language_incident, on=["user_id", "language_id"], how="left")
        frame = frame.merge(self.user_language_post, on=["user_id", "language_id"], how="left")
        frame = frame.merge(self.user_book, on=["user_id", "book_id"], how="left")
        frame = frame.merge(self.user_book_pre, on=["user_id", "book_id"], how="left")
        frame = frame.merge(self.user_book_incident, on=["user_id", "book_id"], how="left")
        frame = frame.merge(self.user_book_post, on=["user_id", "book_id"], how="left")

        genre_features = (
            frame[["user_id", "edition_id", "book_id"]]
            .drop_duplicates()
            .merge(self.book_genres, on="book_id", how="left")
            .merge(self.user_genre, on=["user_id", "genre_id"], how="left")
            .groupby(["user_id", "edition_id"], as_index=False)
            .agg(
                genre_aff=("genre_aff", "sum"),
                genre_last_gap=("genre_last_gap", "min"),
                genre_match_count=("genre_id", "nunique"),
            )
        )
        frame = frame.merge(genre_features, on=["user_id", "edition_id"], how="left")
        pre_genre_features = (
            frame[["user_id", "edition_id", "book_id"]]
            .drop_duplicates()
            .merge(self.book_genres, on="book_id", how="left")
            .merge(self.user_genre_pre, on=["user_id", "genre_id"], how="left")
            .groupby(["user_id", "edition_id"], as_index=False)
            .agg(
                genre_pre_aff=("genre_pre_aff", "sum"),
            )
        )
        frame = frame.merge(pre_genre_features, on=["user_id", "edition_id"], how="left")
        incident_genre_features = (
            frame[["user_id", "edition_id", "book_id"]]
            .drop_duplicates()
            .merge(self.book_genres, on="book_id", how="left")
            .merge(self.user_genre_incident, on=["user_id", "genre_id"], how="left")
            .groupby(["user_id", "edition_id"], as_index=False)
            .agg(
                genre_incident_aff=("genre_incident_aff", "sum"),
            )
        )
        frame = frame.merge(incident_genre_features, on=["user_id", "edition_id"], how="left")
        post_genre_features = (
            frame[["user_id", "edition_id", "book_id"]]
            .drop_duplicates()
            .merge(self.book_genres, on="book_id", how="left")
            .merge(self.user_genre_post, on=["user_id", "genre_id"], how="left")
            .groupby(["user_id", "edition_id"], as_index=False)
            .agg(
                genre_post_aff=("genre_post_aff", "sum"),
            )
        )
        frame = frame.merge(post_genre_features, on=["user_id", "edition_id"], how="left")

        fill_zero = [
            "user_pre_seen",
            "user_pre_weight_sum",
            "user_pre_read_seen",
            "user_incident_seen",
            "user_incident_weight_sum",
            "user_incident_read_seen",
            "user_post_seen",
            "user_post_weight_sum",
            "user_post_read_seen",
            "user_read_seen",
            "user_wishlist_seen",
            "user_mean_rating",
            "user_high_rating_cnt",
            "item_recent_pop",
            "item_recent_read_pop",
            "item_recent_wishlist_pop",
            "item_pre_pop",
            "item_pre_weight_sum",
            "item_pre_read_pop",
            "item_pre_wishlist_pop",
            "item_incident_pop",
            "item_incident_weight_sum",
            "item_incident_read_pop",
            "item_incident_wishlist_pop",
            "item_post_pop",
            "item_post_weight_sum",
            "item_post_read_pop",
            "item_post_wishlist_pop",
            "item_read_pop",
            "item_wishlist_pop",
            "item_mean_rating",
            "item_high_rating_cnt",
            "author_aff",
            "author_pre_aff",
            "author_incident_aff",
            "author_post_aff",
            "author_last_gap",
            "pub_aff",
            "pub_pre_aff",
            "pub_incident_aff",
            "pub_post_aff",
            "lang_aff",
            "lang_pre_aff",
            "lang_incident_aff",
            "lang_post_aff",
            "book_aff",
            "book_pre_aff",
            "book_incident_aff",
            "book_post_aff",
            "book_last_gap",
            "genre_aff",
            "genre_pre_aff",
            "genre_incident_aff",
            "genre_post_aff",
            "genre_last_gap",
            "genre_match_count",
            "age",
            "gender",
        ]
        for column in fill_zero:
            if column in frame.columns:
                frame[column] = frame[column].fillna(0.0)
        if "user_mean_rating" in frame.columns:
            frame["user_mean_rating"] = frame["user_mean_rating"].replace(0.0, np.nan).fillna(8.0)
        if "item_mean_rating" in frame.columns:
            frame["item_mean_rating"] = frame["item_mean_rating"].replace(0.0, np.nan).fillna(8.0)

        if "publication_year" in frame.columns:
            frame["publication_year"] = frame["publication_year"].fillna(0).astype("int32")
        if "age_restriction" in frame.columns:
            frame["age_restriction"] = frame["age_restriction"].fillna(0).astype("int32")
        expected_user_incident = 0.5 * (frame["user_pre_seen"] + frame["user_post_seen"])
        expected_item_incident = 0.5 * (frame["item_pre_pop"] + frame["item_post_pop"])
        frame = frame.copy()
        derived = {
            "author_recent_flag": (frame["author_last_gap"] <= 30).astype("float32"),
            "genre_recent_flag": (frame["genre_last_gap"] <= 30).astype("float32"),
            "book_recent_flag": (frame["book_last_gap"] <= 30).astype("float32"),
            "author_pre_flag": frame["author_pre_aff"].gt(0).astype("float32"),
            "author_incident_flag": frame["author_incident_aff"].gt(0).astype("float32"),
            "author_post_flag": frame["author_post_aff"].gt(0).astype("float32"),
            "genre_pre_flag": frame["genre_pre_aff"].gt(0).astype("float32"),
            "genre_incident_flag": frame["genre_incident_aff"].gt(0).astype("float32"),
            "genre_post_flag": frame["genre_post_aff"].gt(0).astype("float32"),
            "book_pre_flag": frame["book_pre_aff"].gt(0).astype("float32"),
            "book_incident_flag": frame["book_incident_aff"].gt(0).astype("float32"),
            "book_post_flag": frame["book_post_aff"].gt(0).astype("float32"),
            "publisher_pre_flag": frame["pub_pre_aff"].gt(0).astype("float32"),
            "publisher_incident_flag": frame["pub_incident_aff"].gt(0).astype("float32"),
            "publisher_post_flag": frame["pub_post_aff"].gt(0).astype("float32"),
            "language_pre_flag": frame["lang_pre_aff"].gt(0).astype("float32"),
            "language_incident_flag": frame["lang_incident_aff"].gt(0).astype("float32"),
            "language_post_flag": frame["lang_post_aff"].gt(0).astype("float32"),
            "user_read_share": frame["user_read_seen"] / np.maximum(frame["user_seen"], 1.0),
            "user_pre_read_share": frame["user_pre_read_seen"] / np.maximum(frame["user_pre_seen"], 1.0),
            "user_incident_read_share": frame["user_incident_read_seen"]
            / np.maximum(frame["user_incident_seen"], 1.0),
            "item_read_share": frame["item_read_pop"] / np.maximum(frame["item_pop"], 1.0),
            "item_pre_read_share": frame["item_pre_read_pop"] / np.maximum(frame["item_pre_pop"], 1.0),
            "item_incident_read_share": frame["item_incident_read_pop"]
            / np.maximum(frame["item_incident_pop"], 1.0),
            "item_pre_ratio": frame["item_pre_pop"] / (frame["item_pop"] + 1.0),
            "item_incident_ratio": frame["item_incident_pop"] / (frame["item_pop"] + 1.0),
            "item_recent_ratio": frame["item_recent_pop"] / (frame["item_pop"] + 1.0),
            "item_post_ratio": frame["item_post_pop"] / (frame["item_pop"] + 1.0),
            "item_recent_read_share": frame["item_recent_read_pop"] / np.maximum(
                frame["item_recent_pop"], 1.0
            ),
            "item_post_read_share": frame["item_post_read_pop"] / np.maximum(frame["item_post_pop"], 1.0),
            "user_high_rating_share": frame["user_high_rating_cnt"] / np.maximum(frame["user_read_seen"], 1.0),
            "item_high_rating_share": frame["item_high_rating_cnt"] / np.maximum(frame["item_read_pop"], 1.0),
            "user_incident_gap": np.maximum(expected_user_incident - frame["user_incident_seen"], 0.0)
            / (expected_user_incident + 1.0),
            "item_incident_gap": np.maximum(expected_item_incident - frame["item_incident_pop"], 0.0)
            / (expected_item_incident + 1.0),
            "item_bridge_pop": np.minimum(frame["item_pre_pop"], frame["item_post_pop"]),
            "user_rating_norm": frame["user_mean_rating"].clip(0.0, 10.0) / 10.0,
            "item_rating_norm": frame["item_mean_rating"].clip(0.0, 10.0) / 10.0,
            "user_item_pop_ratio": frame["item_pop"] / (frame["user_seen"] + 1.0),
            "age_year_gap": np.abs(frame["publication_year"] - (2025 - frame["age"].fillna(0))),
        }
        derived["item_bridge_ratio"] = derived["item_bridge_pop"] / (frame["item_pop"] + 1.0)
        derived["entity_bridge_flags"] = (
            derived["author_pre_flag"] * derived["author_post_flag"]
            + derived["genre_pre_flag"] * derived["genre_post_flag"]
            + derived["book_pre_flag"] * derived["book_post_flag"]
        ).astype("float32")
        derived["type_alignment"] = 1.0 - np.abs(derived["user_read_share"] - derived["item_read_share"])
        derived["incident_type_alignment"] = 1.0 - np.abs(
            derived["user_incident_read_share"] - derived["item_incident_read_share"]
        )
        return frame.assign(**derived)


def normalize_log1p(series: pd.Series) -> pd.Series:
    values = np.log1p(series.astype("float64"))
    max_value = float(values.max()) if len(values) else 0.0
    if max_value <= 0.0:
        return pd.Series(np.zeros(len(series), dtype="float32"), index=series.index)
    return (values / max_value).astype("float32")


def percentile_by_user(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.zeros(len(df), dtype="float32"), index=df.index)
    values = df[column].fillna(0.0)
    present = values.gt(0)
    if not present.any():
        return pd.Series(np.zeros(len(df), dtype="float32"), index=df.index)
    ranked = values.where(present, 0.0).groupby(df["user_id"]).rank(pct=True, method="average")
    return ranked.where(present, 0.0).astype("float32")


def add_scoring_terms(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    for column in [
        "full_score",
        "full_rank_inv",
        "svd_score",
        "als_score",
        "cooc_score",
        "item_svd_score",
        "user_knn_score",
        "nmf_score",
        "svd_rank_inv",
        "als_rank_inv",
        "cooc_rank_inv",
        "item_svd_rank_inv",
        "user_knn_rank_inv",
        "nmf_rank_inv",
        "author_rank_inv",
        "genre_rank_inv",
        "publisher_rank_inv",
        "language_rank_inv",
        "book_rank_inv",
        "incident_pop_rank_inv",
        "recent_pop_rank_inv",
        "pop_rank_inv",
        "author_aff",
        "author_pre_aff",
        "author_incident_aff",
        "genre_aff",
        "genre_pre_aff",
        "genre_incident_aff",
        "pub_aff",
        "pub_pre_aff",
        "pub_incident_aff",
        "lang_aff",
        "lang_pre_aff",
        "lang_incident_aff",
        "book_aff",
        "book_pre_aff",
        "book_incident_aff",
        "item_pop",
        "item_pre_pop",
        "item_incident_pop",
        "item_recent_pop",
        "item_pre_read_pop",
        "item_incident_read_pop",
        "item_recent_read_pop",
        "item_post_pop",
        "item_post_read_pop",
        "item_read_share",
        "item_pre_read_share",
        "item_incident_read_share",
        "item_recent_read_share",
        "item_post_read_share",
        "item_pre_ratio",
        "item_incident_ratio",
        "item_recent_ratio",
        "item_post_ratio",
        "user_read_share",
        "user_incident_gap",
        "item_incident_gap",
        "item_bridge_ratio",
        "entity_bridge_flags",
        "user_high_rating_share",
        "item_high_rating_share",
        "user_rating_norm",
        "item_rating_norm",
        "type_alignment",
        "incident_type_alignment",
        "author_pre_flag",
        "author_incident_flag",
        "author_post_aff",
        "genre_post_aff",
        "book_post_aff",
        "genre_pre_flag",
        "genre_incident_flag",
        "book_pre_flag",
        "book_incident_flag",
        "author_post_flag",
        "genre_post_flag",
        "book_post_flag",
        "publisher_pre_flag",
        "publisher_incident_flag",
        "publisher_post_flag",
        "language_pre_flag",
        "language_incident_flag",
        "language_post_flag",
        "pub_post_aff",
        "lang_post_aff",
    ]:
        if column not in scored.columns:
            scored[column] = 0.0

    scored["svd_term"] = (
        0.55 * percentile_by_user(scored, "svd_score")
        + 0.45 * scored["svd_rank_inv"].astype("float32")
    )
    scored["als_term"] = (
        0.55 * percentile_by_user(scored, "als_score")
        + 0.45 * scored["als_rank_inv"].astype("float32")
    )
    scored["cooc_term"] = (
        0.55 * percentile_by_user(scored, "cooc_score")
        + 0.45 * scored["cooc_rank_inv"].astype("float32")
    )
    scored["item_svd_term"] = (
        0.55 * percentile_by_user(scored, "item_svd_score")
        + 0.45 * scored["item_svd_rank_inv"].astype("float32")
    )
    scored["user_knn_term"] = (
        0.55 * percentile_by_user(scored, "user_knn_score")
        + 0.45 * scored["user_knn_rank_inv"].astype("float32")
    )
    scored["nmf_term"] = (
        0.55 * percentile_by_user(scored, "nmf_score")
        + 0.45 * scored["nmf_rank_inv"].astype("float32")
    )
    scored["full_score_term"] = (
        0.55 * percentile_by_user(scored, "full_score")
        + 0.45 * scored["full_rank_inv"].astype("float32")
    )
    scored["author_term"] = (
        scored["author_rank_inv"].astype("float32")
        * (
            0.45 * np.sqrt(0.25 + normalize_log1p(scored["author_aff"]))
            + 0.35 * normalize_log1p(scored["author_incident_aff"])
            + 0.20 * normalize_log1p(scored["author_pre_aff"])
        )
        * (
            1.0
            + 0.12 * scored["author_recent_flag"].astype("float32")
            + 0.14 * scored["author_incident_flag"].astype("float32")
            + 0.06 * scored["author_pre_flag"].astype("float32")
            + 0.03 * scored["author_post_flag"].astype("float32")
        )
    )
    scored["genre_term"] = (
        scored["genre_rank_inv"].astype("float32")
        * (
            0.45 * np.sqrt(0.25 + normalize_log1p(scored["genre_aff"]))
            + 0.35 * normalize_log1p(scored["genre_incident_aff"])
            + 0.20 * normalize_log1p(scored["genre_pre_aff"])
        )
        * (
            1.0
            + 0.10 * scored["genre_recent_flag"].astype("float32")
            + 0.13 * scored["genre_incident_flag"].astype("float32")
            + 0.06 * scored["genre_pre_flag"].astype("float32")
            + 0.03 * scored["genre_post_flag"].astype("float32")
        )
    )
    scored["publisher_term"] = (
        scored["publisher_rank_inv"].astype("float32")
        * (
            0.65 * np.sqrt(0.25 + normalize_log1p(scored["pub_aff"]))
            + 0.20 * normalize_log1p(scored["pub_incident_aff"])
            + 0.10 * normalize_log1p(scored["pub_pre_aff"])
            + 0.05 * normalize_log1p(scored["pub_post_aff"])
        )
    )
    scored["language_term"] = (
        scored["language_rank_inv"].astype("float32")
        * (
            0.65 * np.sqrt(0.25 + normalize_log1p(scored["lang_aff"]))
            + 0.20 * normalize_log1p(scored["lang_incident_aff"])
            + 0.10 * normalize_log1p(scored["lang_pre_aff"])
            + 0.05 * normalize_log1p(scored["lang_post_aff"])
        )
    )
    scored["incident_pop_term"] = (
        0.55 * scored["incident_pop_rank_inv"].astype("float32")
        + 0.20 * normalize_log1p(scored["item_incident_pop"])
        + 0.10 * normalize_log1p(scored["item_pre_pop"])
        + 0.10 * scored["item_incident_read_share"].astype("float32")
        + 0.05 * scored["item_incident_gap"].astype("float32")
    )
    scored["recent_pop_term"] = (
        0.35 * scored["recent_pop_rank_inv"].astype("float32")
        + 0.25 * normalize_log1p(scored["item_incident_pop"])
        + 0.15 * normalize_log1p(scored["item_pre_pop"])
        + 0.10 * normalize_log1p(scored["item_recent_pop"])
        + 0.10 * scored["item_incident_read_share"].astype("float32")
        + 0.05 * scored["item_recent_read_share"].astype("float32")
    )
    scored["pop_term"] = (
        0.40 * scored["pop_rank_inv"].astype("float32")
        + 0.20 * normalize_log1p(scored["item_pop"])
        + 0.15 * scored["item_read_share"].astype("float32")
        + 0.10 * scored["item_rating_norm"].astype("float32")
        + 0.05 * scored["item_high_rating_share"].astype("float32")
        + 0.05 * scored["item_bridge_ratio"].astype("float32")
        + 0.05 * scored["item_incident_gap"].astype("float32")
    )
    scored["book_term"] = (
        0.30 * scored["book_rank_inv"].astype("float32")
        + 0.30 * np.minimum(scored["book_aff"].astype("float32"), 3.0) / 3.0
        + 0.20 * normalize_log1p(scored["book_incident_aff"])
        + 0.10 * normalize_log1p(scored["book_pre_aff"])
        + 0.10 * scored["book_recent_flag"].astype("float32")
    ) * (
        1.0
        + 0.10 * scored["book_incident_flag"].astype("float32")
        + 0.05 * scored["book_pre_flag"].astype("float32")
        + 0.03 * scored["book_post_flag"].astype("float32")
    )
    scored["quality_term"] = (
        0.18 * scored["type_alignment"].astype("float32")
        + 0.18 * scored["incident_type_alignment"].astype("float32")
        + 0.15 * scored["user_rating_norm"].astype("float32") * scored["item_rating_norm"].astype("float32")
        + 0.10 * scored["item_rating_norm"].astype("float32")
        + 0.10 * scored["user_high_rating_share"].astype("float32") * scored["item_high_rating_share"].astype("float32")
        + 0.10 * scored["item_incident_ratio"].astype("float32")
        + 0.07 * scored["item_pre_ratio"].astype("float32")
        + 0.07 * scored["item_bridge_ratio"].astype("float32")
        + 0.03 * (scored["entity_bridge_flags"].astype("float32") / 3.0)
        + 0.02 * (
            scored["user_incident_gap"].astype("float32") * scored["item_incident_gap"].astype("float32")
        )
    )
    return scored


def apply_blend(
    frame: pd.DataFrame,
    weights: BlendWeights,
    recovery_bonus_scale: float = 0.0,
) -> pd.DataFrame:
    scored = add_scoring_terms(frame)
    scored["final_score"] = (
        weights.svd * scored["svd_term"]
        + weights.als * scored["als_term"]
        + weights.cooc * scored["cooc_term"]
        + weights.item_svd * scored["item_svd_term"]
        + weights.user_knn * scored["user_knn_term"]
        + weights.nmf * scored["nmf_term"]
        + weights.full_score * scored["full_score_term"]
        + weights.author * scored["author_term"]
        + weights.genre * scored["genre_term"]
        + weights.publisher * scored["publisher_term"]
        + weights.language * scored["language_term"]
        + weights.incident_pop * scored["incident_pop_term"]
        + weights.recent_pop * scored["recent_pop_term"]
        + weights.pop * scored["pop_term"]
        + weights.book * scored["book_term"]
        + weights.quality * scored["quality_term"]
    )
    if recovery_bonus_scale > 0.0:
        scored["final_score"] += float(recovery_bonus_scale) * (
            0.010 * scored["incident_pop_term"]
            + 0.004 * scored["book_term"]
            + 0.003 * scored["full_score_term"]
        )
    return scored


def finalize_topk(
    scored: pd.DataFrame,
    target_users: Iterable[int],
    seen_by_user: dict[int, set[int]],
    popular_incident: list[int] | None,
    popular_recent: list[int],
    popular_all: list[int],
    top_k: int = 20,
) -> pd.DataFrame:
    chosen = (
        scored.sort_values(["user_id", "final_score", "edition_id"], ascending=[True, False, True])
        .groupby("user_id", group_keys=False)
        .head(top_k)
        .copy()
    )
    chosen["rank"] = chosen.groupby("user_id").cumcount() + 1
    chosen = chosen[["user_id", "edition_id", "rank", "final_score"]].copy()

    chosen_pairs = set(map(tuple, chosen[["user_id", "edition_id"]].itertuples(index=False, name=None)))
    missing_rows: list[dict[str, int | float]] = []

    for user_id in target_users:
        user_id = int(user_id)
        current = chosen[chosen["user_id"] == user_id]
        next_rank = int(len(current)) + 1
        if next_rank > top_k:
            continue
        seen = seen_by_user.get(user_id, set())
        fallback_items = (popular_incident or []) + popular_recent + popular_all
        for edition_id in fallback_items:
            pair = (user_id, int(edition_id))
            if pair in chosen_pairs or int(edition_id) in seen:
                continue
            missing_rows.append(
                {
                    "user_id": user_id,
                    "edition_id": int(edition_id),
                    "rank": next_rank,
                    "final_score": 0.0,
                }
            )
            chosen_pairs.add(pair)
            next_rank += 1
            if next_rank > top_k:
                break

    if missing_rows:
        chosen = pd.concat([chosen, pd.DataFrame(missing_rows)], ignore_index=True)

    chosen = chosen.sort_values(["user_id", "rank"]).reset_index(drop=True)
    return chosen


def evaluate_predictions(predictions: pd.DataFrame, relevant_by_user: dict[int, set[int]], users: list[int]) -> float:
    score_sum = 0.0
    for user_id in users:
        rel = relevant_by_user.get(int(user_id), set())
        user_pred = predictions[predictions["user_id"] == int(user_id)].sort_values("rank")
        ranked = user_pred["edition_id"].astype("int64").tolist()
        score_sum += ndcg_at_k(ranked, rel, 20)
    return score_sum / max(1, len(users))


def build_evaluation_frame(
    data: LoadedData,
    masked_pairs: pd.DataFrame,
    limits: SourceLimits,
    svd_cfg: SvdConfig,
    als_cfg: AlsConfig,
    weights: BlendWeights | None,
    recovery_bonus_scale: float,
    self_train_cfg: SelfTrainConfig | None,
    quiet: bool,
) -> tuple[pd.DataFrame, CandidateBuilder]:
    positives = unique_positive_events(data.interactions)
    observed = drop_masked_pairs(positives, masked_pairs)
    max_ts = positives["event_ts"].max()
    pairs = build_weighted_pairs(observed, max_ts=max_ts, cfg=svd_cfg)
    builder = CandidateBuilder(
        pairs=pairs,
        editions=data.editions,
        book_genres=data.book_genres,
        users=data.users,
        max_ts=max_ts,
        limits=limits,
        svd_cfg=svd_cfg,
        als_cfg=als_cfg,
        quiet=quiet,
    )
    target_users = masked_pairs["user_id"].astype("int64").drop_duplicates().tolist()
    frame = builder.generate_candidate_frame(target_users)
    if weights is not None and self_train_cfg is not None and self_train_cfg.enabled and not frame.empty:
        scored = score_frame(frame, weights, recovery_bonus_scale=recovery_bonus_scale)
        pairs = augment_pairs_with_pseudo_labels(pairs, scored, max_ts=max_ts, cfg=self_train_cfg)
        builder = CandidateBuilder(
            pairs=pairs,
            editions=data.editions,
            book_genres=data.book_genres,
            users=data.users,
            max_ts=max_ts,
            limits=limits,
            svd_cfg=svd_cfg,
            als_cfg=als_cfg,
            quiet=quiet,
        )
        frame = builder.generate_candidate_frame(target_users)
    frame["label"] = 0
    if not frame.empty:
        lost_set = set(map(tuple, masked_pairs[["user_id", "edition_id"]].itertuples(index=False, name=None)))
        frame["label"] = [
            1 if (int(user_id), int(edition_id)) in lost_set else 0
            for user_id, edition_id in frame[["user_id", "edition_id"]].itertuples(index=False, name=None)
        ]
    return frame, builder


def score_frame(
    frame: pd.DataFrame,
    weights: BlendWeights,
    recovery_bonus_scale: float = 0.0,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    return apply_blend(frame, weights, recovery_bonus_scale=recovery_bonus_scale)


def augment_pairs_with_pseudo_labels(
    base_pairs: pd.DataFrame,
    scored: pd.DataFrame,
    max_ts: pd.Timestamp,
    cfg: SelfTrainConfig | None,
) -> pd.DataFrame:
    if cfg is None or not cfg.enabled or cfg.topk_per_user <= 0 or scored.empty:
        return base_pairs

    selected = (
        scored.sort_values(["user_id", "final_score", "edition_id"], ascending=[True, False, True])
        .groupby("user_id", group_keys=False)
        .head(int(cfg.topk_per_user))
        .copy()
    )
    if selected.empty:
        return base_pairs

    base_weight = float(base_pairs["pair_weight"].median()) if not base_pairs.empty else 1.0
    base_weight = min(2.5, max(0.8, base_weight))
    score_pct = percentile_by_user(selected, "final_score").clip(0.0, 1.0)
    pseudo_weight = (
        float(cfg.pair_weight_scale)
        * base_weight
        * (0.55 + 0.90 * score_pct.astype("float32"))
    )
    synthetic_ts = OFFICIAL_INCIDENT_END_TS - pd.Timedelta(minutes=1)
    read_share = selected.get("item_read_share", pd.Series(1.0, index=selected.index)).fillna(1.0)
    rating_norm = selected.get("item_rating_norm", pd.Series(0.5, index=selected.index)).fillna(0.5)
    pseudo_pairs = pd.DataFrame(
        {
            "user_id": selected["user_id"].astype("int64").to_numpy(),
            "edition_id": selected["edition_id"].astype("int64").to_numpy(),
            "pair_weight": pseudo_weight.astype("float32").to_numpy(),
            "last_ts": synthetic_ts,
            "last_event_type": np.where(read_share >= 0.5, 2, 1).astype("int8"),
            "max_rating": (6.5 + 3.0 * rating_norm).clip(1.0, 10.0).astype("float32"),
        }
    )

    augmented = pd.concat(
        [
            base_pairs[["user_id", "edition_id", "pair_weight", "last_ts", "last_event_type", "max_rating"]],
            pseudo_pairs,
        ],
        ignore_index=True,
    )
    augmented = (
        augmented.groupby(["user_id", "edition_id"], as_index=False)
        .agg(
            pair_weight=("pair_weight", "max"),
            last_ts=("last_ts", "max"),
            last_event_type=("last_event_type", "max"),
            max_rating=("max_rating", "max"),
        )
        .reset_index(drop=True)
    )
    augmented["days_since_last"] = (max_ts - augmented["last_ts"]).dt.total_seconds() / 86400.0
    return augmented


def prepare_pseudo_frames(
    data: LoadedData,
    limits: SourceLimits,
    svd_cfg: SvdConfig,
    als_cfg: AlsConfig,
    validation_cfg: ValidationConfig,
    weights: BlendWeights | None,
    recovery_bonus_scale: float,
    self_train_cfg: SelfTrainConfig | None,
    seeds: list[int],
    quiet: bool,
) -> list[PreparedPseudoFrame]:
    positives = unique_positive_events(data.interactions)
    allowed_users: list[int] | None = None
    if validation_cfg.user_scope == "targets":
        allowed_users = data.targets["user_id"].astype("int64").tolist()
    prepared: list[PreparedPseudoFrame] = []
    for seed in seeds:
        masked = make_pseudo_mask(
            positives=positives,
            incident_start=validation_cfg.incident_start,
            incident_end=validation_cfg.incident_end,
            mask_ratio=validation_cfg.mask_ratio,
            seed=seed,
            allowed_users=allowed_users,
            user_limit=validation_cfg.user_limit,
            mask_style=validation_cfg.mask_style,
            time_bias_power=validation_cfg.time_bias_power,
            user_heterogeneity=validation_cfg.user_heterogeneity,
        )
        frame, builder = build_evaluation_frame(
            data=data,
            masked_pairs=masked,
            limits=limits,
            svd_cfg=svd_cfg,
            als_cfg=als_cfg,
            weights=weights,
            recovery_bonus_scale=recovery_bonus_scale,
            self_train_cfg=self_train_cfg,
            quiet=quiet,
        )
        prepared.append(
            PreparedPseudoFrame(
                seed=int(seed),
                frame=frame,
                builder=builder,
                relevant_by_user=masked.groupby("user_id")["edition_id"].agg(set).to_dict(),
                target_users=masked["user_id"].astype("int64").drop_duplicates().tolist(),
                masked_size=int(len(masked)),
            )
        )
    return prepared


def add_reranker_features(scored: pd.DataFrame) -> pd.DataFrame:
    reranked = scored.copy()
    if reranked.empty:
        return reranked

    rank_columns = [
        "svd_rank_inv",
        "als_rank_inv",
        "cooc_rank_inv",
        "item_svd_rank_inv",
        "user_knn_rank_inv",
        "nmf_rank_inv",
        "book_rank_inv",
        "author_rank_inv",
        "genre_rank_inv",
        "publisher_rank_inv",
        "language_rank_inv",
        "recent_pop_rank_inv",
        "pop_rank_inv",
    ]
    required_columns = rank_columns + [
        "svd_term",
        "als_term",
        "cooc_term",
        "item_svd_term",
        "user_knn_term",
        "nmf_term",
        "full_score_term",
        "book_term",
        "author_term",
        "genre_term",
        "publisher_term",
        "language_term",
        "recent_pop_term",
        "pop_term",
        "author_recent_flag",
        "genre_recent_flag",
        "book_recent_flag",
        "author_post_flag",
        "genre_post_flag",
        "book_post_flag",
        "item_pop",
        "item_recent_pop",
        "author_aff",
        "author_post_aff",
        "genre_aff",
        "genre_post_aff",
        "pub_aff",
        "lang_aff",
        "book_aff",
        "book_post_aff",
        "final_score",
    ]
    for column in required_columns:
        if column not in reranked.columns:
            reranked[column] = 0.0

    reranked["source_count"] = reranked[rank_columns].gt(0).sum(axis=1).astype("float32")
    reranked["collab_term_sum"] = (
        reranked["svd_term"] + reranked["als_term"] + reranked["cooc_term"]
        + reranked["nmf_term"] + reranked["full_score_term"]
    ).astype("float32")
    reranked["collab_term_max"] = (
        reranked[["svd_term", "als_term", "cooc_term", "nmf_term", "full_score_term"]].max(axis=1).astype("float32")
    )
    reranked["meta_term_sum"] = (
        reranked["book_term"]
        + reranked["author_term"]
        + reranked["genre_term"]
        + reranked["publisher_term"]
        + reranked["language_term"]
    ).astype("float32")
    reranked["meta_term_max"] = (
        reranked[
            ["book_term", "author_term", "genre_term", "publisher_term", "language_term"]
        ]
        .max(axis=1)
        .astype("float32")
    )
    reranked["item_pop_user_pct"] = percentile_by_user(reranked, "item_pop")
    reranked["item_recent_pop_user_pct"] = percentile_by_user(reranked, "item_recent_pop")
    reranked["heuristic_user_pct"] = percentile_by_user(reranked, "final_score")
    reranked["recent_minus_pop"] = (
        reranked["recent_pop_term"] - reranked["pop_term"]
    ).astype("float32")
    reranked["book_author_synergy"] = (
        reranked["book_term"] * reranked["author_term"]
    ).astype("float32")
    reranked["book_genre_synergy"] = (
        reranked["book_term"] * reranked["genre_term"]
    ).astype("float32")
    reranked["svd_cooc_synergy"] = (
        reranked["svd_term"] * reranked["cooc_term"]
    ).astype("float32")
    reranked["affinity_mass"] = (
        normalize_log1p(reranked["author_aff"])
        + normalize_log1p(reranked["genre_aff"])
        + normalize_log1p(reranked["pub_aff"])
        + normalize_log1p(reranked["lang_aff"])
        + normalize_log1p(reranked["book_aff"])
    ).astype("float32")
    reranked["recent_affinity_flags"] = (
        reranked["author_recent_flag"]
        + reranked["genre_recent_flag"]
        + reranked["book_recent_flag"]
    ).astype("float32")
    reranked["post_window_flags"] = (
        reranked["author_post_flag"]
        + reranked["genre_post_flag"]
        + reranked["book_post_flag"]
    ).astype("float32")
    return reranked


def select_reranker_training_rows(
    scored: pd.DataFrame,
    topk_per_user: int,
) -> pd.DataFrame:
    if scored.empty or topk_per_user <= 0:
        return scored.copy()
    ordered = scored.sort_values(
        ["user_id", "final_score", "edition_id"],
        ascending=[True, False, True],
    )
    top_rows = ordered.groupby("user_id", group_keys=False).head(topk_per_user)
    if "label" not in scored.columns:
        return top_rows.reset_index(drop=True)
    positive_rows = scored[scored["label"].gt(0)]
    keep_index = pd.Index(top_rows.index).union(positive_rows.index)
    return scored.loc[keep_index].copy().reset_index(drop=True)


def reranker_feature_spec(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded = {"user_id", "edition_id", "label"}
    feature_cols = [column for column in frame.columns if column not in excluded]
    cat_features = [
        column
        for column in [
            "book_id",
            "author_id",
            "language_id",
            "publisher_id",
            "gender",
            "age_restriction",
        ]
        if column in feature_cols
    ]
    return feature_cols, cat_features


def materialize_reranker_matrix(
    frame: pd.DataFrame,
    feature_cols: list[str],
    cat_features: list[str],
) -> pd.DataFrame:
    matrix = frame[feature_cols].copy()
    for column in cat_features:
        matrix[column] = matrix[column].fillna(0).astype("int64").astype(str)
    return matrix


def fit_catboost_reranker(
    train_frame: pd.DataFrame,
    cfg: RerankerConfig,
    seed: int,
    quiet: bool,
) -> FittedReranker:
    try:
        from catboost import CatBoostClassifier, CatBoostRanker, Pool
    except ImportError as exc:
        raise ImportError(
            "catboost is required for --use-catboost-reranker. Install it first."
        ) from exc

    if train_frame.empty:
        raise ValueError("Reranker training frame is empty.")
    if "label" not in train_frame.columns:
        raise ValueError("Reranker training frame must contain label column.")

    positive_count = int(train_frame["label"].sum())
    if positive_count <= 0:
        raise ValueError("Reranker training frame does not contain positive labels.")

    ordered_train = train_frame.sort_values(
        ["user_id", "label", "final_score", "edition_id"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    feature_cols, cat_features = reranker_feature_spec(ordered_train)
    train_matrix = materialize_reranker_matrix(ordered_train, feature_cols, cat_features)
    negatives = max(1, int(len(train_frame) - positive_count))

    if cfg.mode == "ranker":
        train_pool = Pool(
            train_matrix,
            label=ordered_train["label"].astype("float32"),
            group_id=ordered_train["user_id"].astype("int64"),
            cat_features=cat_features,
        )
        params = {
            "iterations": cfg.iterations,
            "depth": cfg.depth,
            "learning_rate": cfg.learning_rate,
            "l2_leaf_reg": cfg.l2_leaf_reg,
            "loss_function": "YetiRank",
            "eval_metric": "NDCG:top=20",
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": -1,
            "task_type": cfg.task_type,
        }
        model = CatBoostRanker(**params)
        try:
            model.fit(train_pool)
        except Exception as exc:
            if cfg.task_type != "GPU":
                raise
            log(f"CatBoost ranker GPU training failed, retrying on CPU: {exc}", quiet)
            params["task_type"] = "CPU"
            model = CatBoostRanker(**params)
            model.fit(train_pool)
        prediction_kind = "ranker"
    else:
        train_pool = Pool(
            train_matrix,
            label=ordered_train["label"].astype("int8"),
            cat_features=cat_features,
        )
        params = {
            "iterations": cfg.iterations,
            "depth": cfg.depth,
            "learning_rate": cfg.learning_rate,
            "l2_leaf_reg": cfg.l2_leaf_reg,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": -1,
            "scale_pos_weight": float(min(50.0, negatives / max(1, positive_count))),
            "task_type": cfg.task_type,
        }
        model = CatBoostClassifier(**params)
        try:
            model.fit(train_pool)
        except Exception as exc:
            if cfg.task_type != "GPU":
                raise
            log(f"CatBoost classifier GPU training failed, retrying on CPU: {exc}", quiet)
            params["task_type"] = "CPU"
            model = CatBoostClassifier(**params)
            model.fit(train_pool)
        prediction_kind = "classifier"

    return FittedReranker(
        model=model,
        feature_cols=feature_cols,
        cat_features=cat_features,
        prediction_kind=prediction_kind,
    )


def apply_catboost_reranker(
    scored: pd.DataFrame,
    fitted: FittedReranker,
    cfg: RerankerConfig,
    quiet: bool,
) -> pd.DataFrame:
    try:
        from catboost import Pool
    except ImportError as exc:
        raise ImportError(
            "catboost is required for --use-catboost-reranker. Install it first."
        ) from exc

    reranked = add_reranker_features(scored)
    if reranked.empty:
        return reranked

    predictions = np.zeros(len(reranked), dtype="float32")
    batch_size = max(10_000, int(cfg.prediction_batch_size))
    for start_idx in range(0, len(reranked), batch_size):
        end_idx = min(len(reranked), start_idx + batch_size)
        batch = reranked.iloc[start_idx:end_idx]
        batch_matrix = materialize_reranker_matrix(
            batch,
            fitted.feature_cols,
            fitted.cat_features,
        )
        batch_pool = Pool(batch_matrix, cat_features=fitted.cat_features)
        if fitted.prediction_kind == "ranker":
            batch_pred = fitted.model.predict(batch_pool)
        else:
            batch_pred = fitted.model.predict_proba(batch_pool)[:, 1]
        predictions[start_idx:end_idx] = np.asarray(batch_pred, dtype="float32")
        if end_idx < len(reranked):
            log(f"  reranker predicted rows: {end_idx}/{len(reranked)}", quiet)

    reranked["catboost_score"] = predictions
    reranked["final_score"] = (
        cfg.blend_alpha * percentile_by_user(reranked, "catboost_score")
        + (1.0 - cfg.blend_alpha) * percentile_by_user(reranked, "final_score")
    ).astype("float32")
    return reranked


def run_validation(
    data: LoadedData,
    weights: BlendWeights,
    limits: SourceLimits,
    svd_cfg: SvdConfig,
    als_cfg: AlsConfig,
    validation_cfg: ValidationConfig,
    seeds: list[int],
    reranker_cfg: RerankerConfig | None,
    recovery_bonus_scale: float,
    self_train_cfg: SelfTrainConfig | None,
    quiet: bool,
) -> dict[str, object]:
    seed_metrics: list[dict[str, float | int]] = []
    prepared = prepare_pseudo_frames(
        data=data,
        limits=limits,
        svd_cfg=svd_cfg,
        als_cfg=als_cfg,
        validation_cfg=validation_cfg,
        weights=weights,
        recovery_bonus_scale=recovery_bonus_scale,
        self_train_cfg=self_train_cfg,
        seeds=seeds,
        quiet=quiet,
    )
    reranker_frames: list[pd.DataFrame] | None = None
    if reranker_cfg is not None and reranker_cfg.enabled:
        log("Preparing CatBoost reranker training frames...", quiet)
        reranker_frames = [
            select_reranker_training_rows(
                add_reranker_features(
                    score_frame(
                        bundle.frame,
                        weights,
                        recovery_bonus_scale=recovery_bonus_scale,
                    )
                ),
                reranker_cfg.train_topk_per_user,
            )
            for bundle in prepared
        ]

    for bundle_idx, bundle in enumerate(prepared):
        started = time.time()
        log(f"Pseudo-validation seed={bundle.seed}", quiet)
        scored = score_frame(bundle.frame, weights, recovery_bonus_scale=recovery_bonus_scale)
        if reranker_frames is not None:
            train_frames = [
                reranker_frames[idx]
                for idx in range(len(reranker_frames))
                if idx != bundle_idx and not reranker_frames[idx].empty
            ]
            if not train_frames:
                log(
                    "Only one pseudo-validation seed is available; reranker validation is optimistic.",
                    quiet,
                )
                train_frames = [reranker_frames[bundle_idx]]
            reranker_train = pd.concat(train_frames, ignore_index=True)
            fitted = fit_catboost_reranker(
                train_frame=reranker_train,
                cfg=reranker_cfg,
                seed=2026 + int(bundle.seed),
                quiet=quiet,
            )
            scored = apply_catboost_reranker(scored, fitted, reranker_cfg, quiet=quiet)
        predictions = finalize_topk(
            scored=scored,
            target_users=bundle.target_users,
            seen_by_user=bundle.builder.seen_by_user,
            popular_incident=bundle.builder.popular_incident,
            popular_recent=bundle.builder.popular_recent,
            popular_all=bundle.builder.popular_all,
            top_k=20,
        )
        ndcg = evaluate_predictions(predictions, bundle.relevant_by_user, bundle.target_users)
        recall = 0.0
        if not bundle.frame.empty:
            recall = float(bundle.frame["label"].sum()) / float(bundle.masked_size)
        seed_metrics.append(
            {
                "seed": int(bundle.seed),
                "mean_ndcg@20": float(ndcg),
                "candidate_recall": float(recall),
                "users": int(len(bundle.target_users)),
                "elapsed_sec": float(time.time() - started),
            }
        )

    scores = [row["mean_ndcg@20"] for row in seed_metrics]
    recalls = [row["candidate_recall"] for row in seed_metrics]
    return {
        "weights": weights.to_dict(),
        "incident_start": validation_cfg.incident_start,
        "incident_end": validation_cfg.incident_end,
        "validation_user_scope": validation_cfg.user_scope,
        "validation_mask_style": validation_cfg.mask_style,
        "recovery_bonus_scale": float(recovery_bonus_scale),
        "mean_ndcg@20": float(np.mean(scores)) if scores else 0.0,
        "std_ndcg@20": float(np.std(scores)) if scores else 0.0,
        "mean_candidate_recall": float(np.mean(recalls)) if recalls else 0.0,
        "reranker_enabled": bool(reranker_cfg is not None and reranker_cfg.enabled),
        "seeds": seed_metrics,
    }


def sample_dirichlet_weights(rng: np.random.Generator) -> BlendWeights:
    keys = list(BlendWeights().__dict__.keys())
    sampled = rng.dirichlet(np.ones(len(keys)))
    payload = {key: float(value) for key, value in zip(keys, sampled)}
    return BlendWeights(**payload)


def normalize_weights(weights: BlendWeights) -> BlendWeights:
    keys = list(weights.to_dict().keys())
    values = np.array([weights.to_dict()[key] for key in keys], dtype="float64")
    values = np.maximum(values, 1e-6)
    values /= values.sum()
    return BlendWeights(**{key: float(value) for key, value in zip(keys, values)})


def sample_local_weights(
    base_weights: BlendWeights,
    rng: np.random.Generator,
    scale: float = 0.22,
) -> BlendWeights:
    base = np.array(list(base_weights.to_dict().values()), dtype="float64")
    noise = rng.normal(loc=0.0, scale=scale, size=len(base))
    sampled = np.maximum(base * np.exp(noise), 1e-6)
    sampled /= sampled.sum()
    keys = list(base_weights.to_dict().keys())
    return BlendWeights(**{key: float(value) for key, value in zip(keys, sampled)})


def write_tune_checkpoint(
    out_json: Path | None,
    validation_cfg: ValidationConfig,
    best_weights: BlendWeights,
    best_metrics: dict[str, float],
    trial_rows: list[dict[str, object]],
) -> None:
    if out_json is None:
        return
    payload = {
        "incident_start": validation_cfg.incident_start,
        "incident_end": validation_cfg.incident_end,
        "validation_user_scope": validation_cfg.user_scope,
        "validation_mask_style": validation_cfg.mask_style,
        "best_weights": best_weights.to_dict(),
        "best_metrics": best_metrics,
        "trials": trial_rows,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_tune(
    data: LoadedData,
    base_weights: BlendWeights,
    limits: SourceLimits,
    svd_cfg: SvdConfig,
    als_cfg: AlsConfig,
    validation_cfg: ValidationConfig,
    seeds: list[int],
    trials: int,
    out_json: Path | None,
    recovery_bonus_scale: float,
    quiet: bool,
) -> dict[str, object]:
    log("Building pseudo-validation frames for tuning...", quiet)
    prepared = prepare_pseudo_frames(
        data=data,
        limits=limits,
        svd_cfg=svd_cfg,
        als_cfg=als_cfg,
        validation_cfg=validation_cfg,
        weights=None,
        recovery_bonus_scale=recovery_bonus_scale,
        self_train_cfg=None,
        seeds=seeds,
        quiet=quiet,
    )

    def evaluate(weights: BlendWeights) -> dict[str, float]:
        ndcgs: list[float] = []
        recalls: list[float] = []
        for bundle in prepared:
            scored = score_frame(bundle.frame, weights, recovery_bonus_scale=recovery_bonus_scale)
            predictions = finalize_topk(
                scored=scored,
                target_users=bundle.target_users,
                seen_by_user=bundle.builder.seen_by_user,
                popular_incident=bundle.builder.popular_incident,
                popular_recent=bundle.builder.popular_recent,
                popular_all=bundle.builder.popular_all,
                top_k=20,
            )
            ndcgs.append(
                evaluate_predictions(predictions, bundle.relevant_by_user, bundle.target_users)
            )
            recall = (
                float(bundle.frame["label"].sum()) / float(bundle.masked_size)
                if bundle.masked_size
                else 0.0
            )
            recalls.append(recall)
        return {
            "mean_ndcg@20": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "mean_candidate_recall": float(np.mean(recalls)) if recalls else 0.0,
        }

    best_weights = normalize_weights(base_weights)
    best_metrics = evaluate(best_weights)
    log(f"Base mean_ndcg@20={best_metrics['mean_ndcg@20']:.6f}", quiet)

    rng = np.random.default_rng(2026)
    trial_rows = [
        {
            "trial": 0,
            "weights": best_weights.to_dict(),
            **best_metrics,
        }
    ]
    write_tune_checkpoint(out_json, validation_cfg, best_weights, best_metrics, trial_rows)
    tune_started = time.time()

    for trial_idx in range(1, trials + 1):
        if trial_idx <= 3:
            weights = sample_local_weights(best_weights, rng, scale=0.08 * trial_idx)
        elif trial_idx % 5 == 0:
            weights = sample_dirichlet_weights(rng)
        else:
            weights = sample_local_weights(best_weights, rng)
        metrics = evaluate(weights)
        trial_rows.append({"trial": trial_idx, "weights": weights.to_dict(), **metrics})
        if metrics["mean_ndcg@20"] > best_metrics["mean_ndcg@20"]:
            best_weights = weights
            best_metrics = metrics
            log(
                f"Trial {trial_idx}: improved mean_ndcg@20={best_metrics['mean_ndcg@20']:.6f}",
                quiet,
            )
            write_tune_checkpoint(out_json, validation_cfg, best_weights, best_metrics, trial_rows)
        if (trial_idx % 10 == 0) or (trial_idx == trials):
            elapsed = time.time() - tune_started
            avg_trial_sec = elapsed / max(1, trial_idx)
            eta_sec = avg_trial_sec * max(0, trials - trial_idx)
            log(
                (
                    f"Trial {trial_idx}/{trials} | "
                    f"best={best_metrics['mean_ndcg@20']:.6f} | "
                    f"avg_trial_sec={avg_trial_sec:.2f} | "
                    f"eta_min={eta_sec / 60.0:.1f}"
                ),
                quiet,
            )
            write_tune_checkpoint(out_json, validation_cfg, best_weights, best_metrics, trial_rows)

    return {
        "incident_start": validation_cfg.incident_start,
        "incident_end": validation_cfg.incident_end,
        "validation_user_scope": validation_cfg.user_scope,
        "validation_mask_style": validation_cfg.mask_style,
        "recovery_bonus_scale": float(recovery_bonus_scale),
        "best_weights": best_weights.to_dict(),
        "best_metrics": best_metrics,
        "trials": trial_rows,
    }


def build_submission(
    data: LoadedData,
    weights: BlendWeights,
    limits: SourceLimits,
    svd_cfg: SvdConfig,
    als_cfg: AlsConfig,
    reranker_cfg: RerankerConfig | None,
    self_train_cfg: SelfTrainConfig | None,
    reranker_validation_cfg: ValidationConfig | None,
    reranker_seeds: list[int] | None,
    recovery_bonus_scale: float,
    quiet: bool,
) -> pd.DataFrame:
    positives = unique_positive_events(data.interactions)
    max_ts = positives["event_ts"].max()
    pairs = build_weighted_pairs(positives, max_ts=max_ts, cfg=svd_cfg)
    builder = CandidateBuilder(
        pairs=pairs,
        editions=data.editions,
        book_genres=data.book_genres,
        users=data.users,
        max_ts=max_ts,
        limits=limits,
        svd_cfg=svd_cfg,
        als_cfg=als_cfg,
        quiet=quiet,
    )
    target_users = data.targets["user_id"].astype("int64").tolist()
    frame = builder.generate_candidate_frame(target_users)
    scored = score_frame(frame, weights, recovery_bonus_scale=recovery_bonus_scale)
    if self_train_cfg is not None and self_train_cfg.enabled and not scored.empty:
        pairs = augment_pairs_with_pseudo_labels(pairs, scored, max_ts=max_ts, cfg=self_train_cfg)
        builder = CandidateBuilder(
            pairs=pairs,
            editions=data.editions,
            book_genres=data.book_genres,
            users=data.users,
            max_ts=max_ts,
            limits=limits,
            svd_cfg=svd_cfg,
            als_cfg=als_cfg,
            quiet=quiet,
        )
        frame = builder.generate_candidate_frame(target_users)
        scored = score_frame(frame, weights, recovery_bonus_scale=recovery_bonus_scale)
    if reranker_cfg is not None and reranker_cfg.enabled:
        log("Training CatBoost reranker on pseudo-masks...", quiet)
        pseudo_cfg = reranker_validation_cfg or ValidationConfig()
        pseudo_seeds = reranker_seeds or list(DEFAULT_SEEDS)
        prepared = prepare_pseudo_frames(
            data=data,
            limits=limits,
            svd_cfg=svd_cfg,
            als_cfg=als_cfg,
            validation_cfg=pseudo_cfg,
            weights=weights,
            recovery_bonus_scale=recovery_bonus_scale,
            self_train_cfg=self_train_cfg,
            seeds=pseudo_seeds,
            quiet=quiet,
        )
        reranker_train = pd.concat(
            [
                select_reranker_training_rows(
                    add_reranker_features(
                        score_frame(
                            bundle.frame,
                            weights,
                            recovery_bonus_scale=recovery_bonus_scale,
                        )
                    ),
                    reranker_cfg.train_topk_per_user,
                )
                for bundle in prepared
            ],
            ignore_index=True,
        )
        fitted = fit_catboost_reranker(
            train_frame=reranker_train,
            cfg=reranker_cfg,
            seed=2026,
            quiet=quiet,
        )
        scored = apply_catboost_reranker(scored, fitted, reranker_cfg, quiet=quiet)
    submission = finalize_topk(
        scored=scored,
        target_users=target_users,
        seen_by_user=builder.seen_by_user,
        popular_incident=builder.popular_incident,
        popular_recent=builder.popular_recent,
        popular_all=builder.popular_all,
        top_k=20,
    )
    submission = submission[["user_id", "edition_id", "rank"]].copy()
    submission["user_id"] = submission["user_id"].astype("int64")
    submission["edition_id"] = submission["edition_id"].astype("int64")
    submission["rank"] = submission["rank"].astype("int32")
    validate_submission_frame(submission, data.targets, top_k=20)
    return submission


def load_submission_frame(path: Path) -> pd.DataFrame:
    submission = pd.read_csv(path, usecols=["user_id", "edition_id", "rank"])
    submission["user_id"] = submission["user_id"].astype("int64")
    submission["edition_id"] = submission["edition_id"].astype("int64")
    submission["rank"] = submission["rank"].astype("int32")
    return submission


def build_blended_submission(
    submission_paths: list[Path],
    targets: pd.DataFrame,
    weights: list[float] | None,
    method: str,
    rrf_k: float,
) -> pd.DataFrame:
    if len(submission_paths) < 2:
        raise ValueError("Need at least two submissions to blend.")
    if method not in {"rrf", "inverse_rank", "borda"}:
        raise ValueError("Unsupported blend method.")

    if weights is None or len(weights) == 0:
        normalized_weights = np.full(len(submission_paths), 1.0 / len(submission_paths), dtype="float64")
    else:
        if len(weights) != len(submission_paths):
            raise ValueError("weights length must match input_submissions length.")
        normalized_weights = np.asarray(weights, dtype="float64")
        if np.any(normalized_weights < 0.0):
            raise ValueError("weights must be non-negative.")
        if not np.any(normalized_weights > 0.0):
            raise ValueError("At least one blend weight must be positive.")
        normalized_weights = normalized_weights / normalized_weights.sum()

    scored_frames: list[pd.DataFrame] = []
    for idx, submission_path in enumerate(submission_paths):
        frame = load_submission_frame(submission_path)
        validate_submission_frame(frame, targets, top_k=20)
        weight = float(normalized_weights[idx])
        if method == "rrf":
            score = weight / (float(rrf_k) + frame["rank"].astype("float64"))
        elif method == "inverse_rank":
            score = weight / frame["rank"].astype("float64")
        else:
            score = weight * (21.0 - frame["rank"].astype("float64")) / 20.0
        scored = frame.copy()
        scored["blend_score"] = score.astype("float32")
        scored["support_count"] = 1
        scored_frames.append(scored)

    combined = pd.concat(scored_frames, ignore_index=True)
    blended = (
        combined.groupby(["user_id", "edition_id"], as_index=False)
        .agg(
            blend_score=("blend_score", "sum"),
            support_count=("support_count", "sum"),
            best_rank=("rank", "min"),
        )
        .sort_values(
            ["user_id", "blend_score", "support_count", "best_rank", "edition_id"],
            ascending=[True, False, False, True, True],
        )
        .groupby("user_id", group_keys=False)
        .head(20)
        .copy()
    )
    blended["rank"] = blended.groupby("user_id").cumcount() + 1
    blended = blended[["user_id", "edition_id", "rank"]].copy()
    blended["user_id"] = blended["user_id"].astype("int64")
    blended["edition_id"] = blended["edition_id"].astype("int64")
    blended["rank"] = blended["rank"].astype("int32")
    validate_submission_frame(blended, targets, top_k=20)
    return blended


def main() -> None:
    args = parse_args()
    if args.command == "blend":
        data = load_data(args.data_dir)
        blended = build_blended_submission(
            submission_paths=list(args.input_submissions),
            targets=data.targets,
            weights=list(args.weights) if args.weights else None,
            method=str(args.method),
            rrf_k=float(args.rrf_k),
        )
        blended.to_csv(args.submission_path, index=False)
        print(
            json.dumps(
                {
                    "submission_path": str(args.submission_path),
                    "rows": int(len(blended)),
                    "users": int(blended["user_id"].nunique()),
                    "inputs": [str(path) for path in args.input_submissions],
                    "method": str(args.method),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    quiet = bool(getattr(args, "quiet", False))
    data = load_data(args.data_dir)

    weights = BlendWeights.from_path(getattr(args, "weights_json", None))
    limits = SourceLimits(
        candidate_cap=int(getattr(args, "candidate_cap", SourceLimits().candidate_cap)),
        user_history_cap=int(getattr(args, "user_history_cap", SourceLimits().user_history_cap)),
    )
    svd_cfg = SvdConfig(components=int(getattr(args, "svd_components", SvdConfig().components)))
    als_cfg = AlsConfig(
        enabled=bool(getattr(args, "use_implicit_als", False)),
        use_gpu=bool(getattr(args, "als_use_gpu", False)),
    )
    reranker_cfg = RerankerConfig(
        enabled=bool(getattr(args, "use_catboost_reranker", False)),
        mode=str(getattr(args, "catboost_mode", "ranker")),
        iterations=int(getattr(args, "catboost_iterations", 450)),
        depth=int(getattr(args, "catboost_depth", 8)),
        learning_rate=float(getattr(args, "catboost_learning_rate", 0.05)),
        l2_leaf_reg=float(getattr(args, "catboost_l2_leaf_reg", 6.0)),
        train_topk_per_user=int(getattr(args, "catboost_train_topk", 140)),
        blend_alpha=float(getattr(args, "catboost_blend_alpha", 0.82)),
        task_type=str(getattr(args, "catboost_task_type", "CPU")),
    )
    self_train_cfg = SelfTrainConfig(
        enabled=bool(getattr(args, "use_pseudo_refit", False)),
        topk_per_user=int(getattr(args, "pseudo_refit_topk", 1)),
        pair_weight_scale=float(getattr(args, "pseudo_refit_weight", 0.55)),
    )

    if args.command == "validate":
        result = run_validation(
            data=data,
            weights=weights,
            limits=limits,
            svd_cfg=svd_cfg,
            als_cfg=als_cfg,
            validation_cfg=ValidationConfig(
                incident_start=args.incident_start,
                incident_end=args.incident_end,
                mask_ratio=float(args.mask_ratio),
                user_limit=args.user_limit,
                user_scope=str(args.validation_user_scope),
                mask_style=str(args.validation_mask_style),
                time_bias_power=float(args.validation_time_bias),
                user_heterogeneity=float(args.validation_user_heterogeneity),
            ),
            seeds=list(args.seeds),
            reranker_cfg=reranker_cfg,
            recovery_bonus_scale=float(getattr(args, "recovery_bonus_scale", 0.0)),
            self_train_cfg=self_train_cfg,
            quiet=quiet,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "tune":
        result = run_tune(
            data=data,
            base_weights=weights,
            limits=limits,
            svd_cfg=svd_cfg,
            als_cfg=als_cfg,
            validation_cfg=ValidationConfig(
                incident_start=args.incident_start,
                incident_end=args.incident_end,
                mask_ratio=float(args.mask_ratio),
                user_limit=args.user_limit,
                user_scope=str(args.validation_user_scope),
                mask_style=str(args.validation_mask_style),
                time_bias_power=float(args.validation_time_bias),
                user_heterogeneity=float(args.validation_user_heterogeneity),
            ),
            seeds=list(args.seeds),
            trials=int(args.trials),
            out_json=args.out_json,
            recovery_bonus_scale=float(getattr(args, "recovery_bonus_scale", 0.0)),
            quiet=quiet,
        )
        args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result["best_metrics"], ensure_ascii=False, indent=2))
        print(json.dumps({"best_weights": result["best_weights"]}, ensure_ascii=False, indent=2))
        return

    if args.command == "submit":
        submission = build_submission(
            data=data,
            weights=weights,
            limits=limits,
            svd_cfg=svd_cfg,
            als_cfg=als_cfg,
            reranker_cfg=reranker_cfg,
            self_train_cfg=self_train_cfg,
            reranker_validation_cfg=ValidationConfig(
                incident_start=getattr(args, "reranker_incident_start", OFFICIAL_INCIDENT_START),
                incident_end=getattr(args, "reranker_incident_end", OFFICIAL_INCIDENT_END),
                mask_ratio=float(getattr(args, "reranker_mask_ratio", 0.2)),
                user_limit=getattr(args, "reranker_train_user_limit", None),
                user_scope="targets",
                mask_style="heterogeneous",
                time_bias_power=1.35,
                user_heterogeneity=0.75,
            ),
            reranker_seeds=list(getattr(args, "reranker_seeds", list(DEFAULT_SEEDS))),
            recovery_bonus_scale=float(getattr(args, "recovery_bonus_scale", 0.0)),
            quiet=quiet,
        )
        submission.to_csv(args.submission_path, index=False)
        print(
            json.dumps(
                {
                    "submission_path": str(args.submission_path),
                    "rows": int(len(submission)),
                    "users": int(submission["user_id"].nunique()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
