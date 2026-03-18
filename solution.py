"""
Solution for NTO AI Final - "Потеряшки" (Lost Interactions Recovery)

Recovers lost positive interactions (wishlist/read) from incomplete event logs.
Uses multiple candidate generation strategies + blended ranking.

Usage: python solution.py [data_dir]
"""

import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
OUTPUT_FILE = Path("submission.csv")
TOP_K = 20
INCIDENT_DAYS = 30


def load_data(data_dir: Path) -> dict:
    print("Loading data...")
    interactions = pd.read_csv(data_dir / "interactions.csv")
    targets = pd.read_csv(data_dir / "targets.csv")
    editions = pd.read_csv(data_dir / "editions.csv")
    book_genres = pd.read_csv(data_dir / "book_genres.csv")

    interactions["event_ts"] = pd.to_datetime(interactions["event_ts"])

    return {
        "interactions": interactions,
        "targets": targets,
        "editions": editions,
        "book_genres": book_genres,
    }


def build_sparse_matrix(user_ids, item_ids, values, user2idx, item2idx):
    """Build sparse matrix from arrays using vectorized ops."""
    row_idx = np.array([user2idx.get(u, -1) for u in user_ids])
    col_idx = np.array([item2idx.get(i, -1) for i in item_ids])
    mask = (row_idx >= 0) & (col_idx >= 0)
    return csr_matrix(
        (np.array(values)[mask], (row_idx[mask], col_idx[mask])),
        shape=(len(user2idx), len(item2idx)),
    )


def als_candidates(positives, target_users_set, user2idx, item2idx, idx2item, seen_set, n_cands=200):
    """ALS collaborative filtering candidates."""
    print("  ALS collaborative filtering...")
    try:
        from implicit.als import AlternatingLeastSquares
    except ImportError:
        print("  implicit not available, skipping ALS")
        return {}

    # Build weighted interaction matrix
    weights = np.where(positives["event_type"].values == 2, 2.0, 1.0)
    ratings = positives["rating"].values
    mask_rating = (~np.isnan(ratings)) & (positives["event_type"].values == 2)
    weights[mask_rating] += np.clip(ratings[mask_rating] / 5.0, 0, 1)

    mat = build_sparse_matrix(
        positives["user_id"].values,
        positives["edition_id"].values,
        weights,
        user2idx, item2idx,
    )

    model = AlternatingLeastSquares(
        factors=128, iterations=25, regularization=0.01,
        random_state=42, use_gpu=False,
    )
    model.fit(mat)

    # Batch recommend for all target users that exist in the matrix
    target_indices = []
    target_uids = []
    for uid in target_users_set:
        if uid in user2idx:
            target_indices.append(user2idx[uid])
            target_uids.append(uid)

    if not target_indices:
        return {}

    target_indices = np.array(target_indices)
    user_items = mat[target_indices]

    # Recommend in batch
    all_ids, all_scores = model.recommend(
        target_indices, user_items, N=n_cands, filter_already_liked_items=True
    )

    candidates = {}
    for i, uid in enumerate(target_uids):
        user_cands = {}
        for j in range(all_ids.shape[1]):
            eid = idx2item[int(all_ids[i, j])]
            if (uid, eid) not in seen_set:
                user_cands[eid] = float(all_scores[i, j])
        candidates[uid] = user_cands

    print(f"  ALS: {len(candidates)} users")
    return candidates


def item_sim_candidates(positives, target_users_set, user2idx, item2idx, idx2item, seen_set, user_items_map, n_cands=200):
    """Item-based collaborative filtering via cosine similarity."""
    print("  Item-based CF...")

    weights = np.where(positives["event_type"].values == 2, 2.0, 1.0)

    # Item-user matrix (items as rows)
    item_ids = positives["edition_id"].values
    user_ids = positives["user_id"].values
    row_idx = np.array([item2idx.get(i, -1) for i in item_ids])
    col_idx = np.array([user2idx.get(u, -1) for u in user_ids])
    mask = (row_idx >= 0) & (col_idx >= 0)

    n_items = len(item2idx)
    n_users = len(user2idx)
    item_user_mat = csr_matrix(
        (weights[mask], (row_idx[mask], col_idx[mask])),
        shape=(n_items, n_users),
    )
    item_user_norm = normalize(item_user_mat, norm="l2", axis=1)

    candidates = {}
    for uid in target_users_set:
        if uid not in user2idx:
            continue
        u_items = user_items_map.get(uid, set())
        if not u_items:
            continue

        item_indices = [item2idx[e] for e in u_items if e in item2idx]
        if not item_indices:
            continue

        # Similarity of user's items to all items
        user_vecs = item_user_norm[item_indices]
        sim = user_vecs.dot(item_user_norm.T).toarray()
        agg = sim.max(axis=0)

        # Get top candidates excluding seen
        top_idx = np.argsort(-agg)[:n_cands + len(item_indices)]
        user_cands = {}
        for idx in top_idx:
            eid = idx2item[idx]
            if eid in u_items or (uid, eid) in seen_set:
                continue
            user_cands[eid] = float(agg[idx])
            if len(user_cands) >= n_cands:
                break
        candidates[uid] = user_cands

    print(f"  Item-sim: {len(candidates)} users")
    return candidates


def genre_candidates(positives, editions, book_genres, target_users_set, seen_set, user_items_map, n_cands=200):
    """User-genre affinity candidates."""
    print("  Genre affinity...")

    user_ed = positives[["user_id", "edition_id"]].drop_duplicates()
    ed_book = editions[["edition_id", "book_id"]].drop_duplicates()
    ub = user_ed.merge(ed_book, on="edition_id")
    ugc = ub.merge(book_genres, on="book_id").groupby(
        ["user_id", "genre_id"]
    ).size().reset_index(name="count")
    totals = ugc.groupby("user_id")["count"].transform("sum")
    ugc["weight"] = ugc["count"] / totals

    # Edition popularity
    ed_pop = positives.groupby("edition_id")["user_id"].nunique()
    ed_pop_map = ed_pop.to_dict()

    # Genre -> editions (sorted by pop)
    eg = editions[["edition_id", "book_id"]].merge(book_genres, on="book_id")
    eg["pop"] = eg["edition_id"].map(ed_pop_map).fillna(0)
    genre_eds = {}
    for gid, grp in eg.groupby("genre_id"):
        top = grp.nlargest(500, "pop")
        genre_eds[gid] = list(zip(top["edition_id"].values, top["pop"].values))

    # User genre profiles (only for targets)
    ugc_target = ugc[ugc["user_id"].isin(target_users_set)]
    user_genre_profiles = defaultdict(dict)
    for row in ugc_target.itertuples(index=False):
        user_genre_profiles[row.user_id][row.genre_id] = row.weight

    candidates = {}
    for uid in target_users_set:
        profile = user_genre_profiles.get(uid)
        if not profile:
            continue
        u_items = user_items_map.get(uid, set())
        scores = {}
        for gid, w in profile.items():
            for eid, pop in genre_eds.get(gid, []):
                if eid in u_items or (uid, eid) in seen_set:
                    continue
                scores[eid] = scores.get(eid, 0.0) + w * (pop + 1.0)
        top = sorted(scores.items(), key=lambda x: -x[1])[:n_cands]
        candidates[uid] = dict(top)

    print(f"  Genre: {len(candidates)} users")
    return candidates


def author_candidates(positives, editions, target_users_set, seen_set, user_items_map, n_cands=200):
    """User-author affinity candidates."""
    print("  Author affinity...")

    user_ed = positives[["user_id", "edition_id"]].drop_duplicates()
    ed_auth = editions[["edition_id", "author_id"]].drop_duplicates()
    ua = user_ed.merge(ed_auth, on="edition_id")
    uac = ua.groupby(["user_id", "author_id"]).size().reset_index(name="count")
    totals = uac.groupby("user_id")["count"].transform("sum")
    uac["weight"] = uac["count"] / totals

    ed_pop = positives.groupby("edition_id")["user_id"].nunique()
    ed_pop_map = ed_pop.to_dict()

    # Author -> editions
    auth_eds = defaultdict(list)
    for row in editions[["edition_id", "author_id"]].itertuples(index=False):
        pop = ed_pop_map.get(row.edition_id, 0)
        auth_eds[row.author_id].append((row.edition_id, pop))
    for aid in auth_eds:
        auth_eds[aid].sort(key=lambda x: -x[1])
        auth_eds[aid] = auth_eds[aid][:500]

    uac_target = uac[uac["user_id"].isin(target_users_set)]
    user_auth_profiles = defaultdict(dict)
    for row in uac_target.itertuples(index=False):
        user_auth_profiles[row.user_id][row.author_id] = row.weight

    candidates = {}
    for uid in target_users_set:
        profile = user_auth_profiles.get(uid)
        if not profile:
            continue
        u_items = user_items_map.get(uid, set())
        scores = {}
        for aid, w in profile.items():
            for eid, pop in auth_eds.get(aid, []):
                if eid in u_items or (uid, eid) in seen_set:
                    continue
                scores[eid] = scores.get(eid, 0.0) + w * (pop + 1.0)
        top = sorted(scores.items(), key=lambda x: -x[1])[:n_cands]
        candidates[uid] = dict(top)

    print(f"  Author: {len(candidates)} users")
    return candidates


def popularity_candidates(positives, incident_start, target_users_set, seen_set, user_items_map, n_cands=200):
    """Global + recent popularity candidates."""
    print("  Popularity...")

    g_pop = positives.groupby("edition_id")["user_id"].nunique()
    recent = positives[positives["event_ts"] >= incident_start]
    r_pop = recent.groupby("edition_id")["user_id"].nunique()

    all_eds = set(g_pop.index) | set(r_pop.index)
    scores_list = []
    for eid in all_eds:
        g = g_pop.get(eid, 0)
        r = r_pop.get(eid, 0)
        scores_list.append((eid, g + 2.0 * r))
    scores_list.sort(key=lambda x: -x[1])
    top_popular = scores_list[:n_cands * 3]

    candidates = {}
    for uid in target_users_set:
        u_items = user_items_map.get(uid, set())
        user_cands = {}
        for eid, score in top_popular:
            if eid in u_items or (uid, eid) in seen_set:
                continue
            user_cands[eid] = score
            if len(user_cands) >= n_cands:
                break
        candidates[uid] = user_cands

    print(f"  Popularity: {len(candidates)} users")
    return candidates


def book_level_candidates(positives, editions, target_users_set, seen_set, user_items_map, n_cands=200):
    """Recommend other editions of books the user has interacted with at book level.

    If a user read edition A of book X, recommend other editions of book X,
    plus editions of books by the same author or in the same series.
    """
    print("  Book-level candidates...")

    ed_book = editions[["edition_id", "book_id"]].drop_duplicates()
    book_to_eds = defaultdict(set)
    for row in ed_book.itertuples(index=False):
        book_to_eds[row.book_id].add(row.edition_id)

    ed_pop = positives.groupby("edition_id")["user_id"].nunique()
    ed_pop_map = ed_pop.to_dict()

    # Map user -> books they've interacted with
    user_ed = positives[["user_id", "edition_id"]].drop_duplicates()
    user_books = user_ed.merge(ed_book, on="edition_id")
    user_book_map = defaultdict(set)
    for row in user_books.itertuples(index=False):
        user_book_map[row.user_id].add(row.book_id)

    candidates = {}
    for uid in target_users_set:
        u_items = user_items_map.get(uid, set())
        u_books = user_book_map.get(uid, set())
        scores = {}
        for bid in u_books:
            for eid in book_to_eds.get(bid, set()):
                if eid in u_items or (uid, eid) in seen_set:
                    continue
                pop = ed_pop_map.get(eid, 0)
                scores[eid] = scores.get(eid, 0.0) + pop + 1.0
        top = sorted(scores.items(), key=lambda x: -x[1])[:n_cands]
        candidates[uid] = dict(top)

    print(f"  Book-level: {len(candidates)} users")
    return candidates


def blend_and_rank(sources: dict, target_users, k=TOP_K):
    """Blend candidate sources with source-level normalization and weighting."""
    print("Blending and ranking...")

    source_weights = {
        "als": 4.0,
        "item_sim": 3.0,
        "genre": 1.5,
        "author": 1.5,
        "popularity": 0.8,
        "book_level": 2.0,
    }

    all_scores = defaultdict(lambda: defaultdict(float))

    for src_name, cands in sources.items():
        w = source_weights.get(src_name, 1.0)
        for uid, items in cands.items():
            if not items:
                continue
            vals = list(items.values())
            mx, mn = max(vals), min(vals)
            rng = mx - mn if mx > mn else 1.0
            for eid, score in items.items():
                norm = (score - mn) / rng
                all_scores[uid][eid] += w * norm

    results = []
    for uid in target_users:
        user_scores = all_scores.get(uid, {})
        if user_scores:
            sorted_items = sorted(user_scores.items(), key=lambda x: (-x[1], x[0]))
            for rank, (eid, _) in enumerate(sorted_items[:k], 1):
                results.append({"user_id": uid, "edition_id": eid, "rank": rank})

    return results


def fill_missing(results, target_users, positives, seen_set, user_items_map, k=TOP_K):
    """Fill users with < k recommendations using popular items."""
    print("Filling missing slots...")

    pop = positives.groupby("edition_id")["user_id"].nunique().reset_index(name="pop")
    pop = pop.sort_values(["pop", "edition_id"], ascending=[False, True])
    popular = pop["edition_id"].tolist()

    user_results = defaultdict(list)
    user_eds = defaultdict(set)
    for r in results:
        user_results[r["user_id"]].append(r)
        user_eds[r["user_id"]].add(r["edition_id"])

    final = []
    for uid in target_users:
        recs = sorted(user_results.get(uid, []), key=lambda x: x["rank"])
        cur_eds = set(user_eds.get(uid, set()))

        for r in recs[:k]:
            final.append(r)

        count = min(len(recs), k)
        if count < k:
            u_items = user_items_map.get(uid, set())
            rank = count + 1
            for eid in popular:
                if eid in cur_eds or eid in u_items or (uid, eid) in seen_set:
                    continue
                final.append({"user_id": uid, "edition_id": eid, "rank": rank})
                cur_eds.add(eid)
                rank += 1
                if rank > k:
                    break

    return final


def main():
    data_dir = DATA_DIR
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])

    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' not found.")
        print("Usage: python solution.py [data_dir]")
        sys.exit(1)

    data = load_data(data_dir)
    interactions = data["interactions"]
    targets = data["targets"]
    editions = data["editions"]
    book_genres = data["book_genres"]

    target_users = targets["user_id"].tolist()
    target_set = set(target_users)

    # Positive interactions only
    positives = interactions[interactions["event_type"].isin([1, 2])].copy()
    max_ts = interactions["event_ts"].max()
    incident_start = max_ts - pd.Timedelta(days=INCIDENT_DAYS)

    print(f"Total interactions: {len(interactions):,}")
    print(f"Positive interactions: {len(positives):,}")
    print(f"Target users: {len(target_users):,}")
    print(f"Unique editions: {editions['edition_id'].nunique():,}")
    print(f"Incident window: {incident_start} to {max_ts}")

    # Build index structures (vectorized)
    seen_pairs = positives[["user_id", "edition_id"]].drop_duplicates()
    seen_set = set(zip(seen_pairs["user_id"].values, seen_pairs["edition_id"].values))
    print(f"Seen positive pairs: {len(seen_set):,}")

    # User -> items map
    user_items_map = defaultdict(set)
    for uid, eid in zip(positives["user_id"].values, positives["edition_id"].values):
        user_items_map[uid].add(eid)

    # Build user/item index mappings
    all_users = sorted(positives["user_id"].unique())
    all_items = sorted(positives["edition_id"].unique())
    user2idx = {u: i for i, u in enumerate(all_users)}
    item2idx = {e: i for i, e in enumerate(all_items)}
    idx2item = {i: e for e, i in item2idx.items()}

    # Generate candidates
    print("\nGenerating candidates...")
    sources = {}

    sources["als"] = als_candidates(
        positives, target_set, user2idx, item2idx, idx2item, seen_set, n_cands=200,
    )

    sources["item_sim"] = item_sim_candidates(
        positives, target_set, user2idx, item2idx, idx2item, seen_set, user_items_map, n_cands=200,
    )

    sources["genre"] = genre_candidates(
        positives, editions, book_genres, target_set, seen_set, user_items_map, n_cands=200,
    )

    sources["author"] = author_candidates(
        positives, editions, target_set, seen_set, user_items_map, n_cands=200,
    )

    sources["popularity"] = popularity_candidates(
        positives, incident_start, target_set, seen_set, user_items_map, n_cands=200,
    )

    sources["book_level"] = book_level_candidates(
        positives, editions, target_set, seen_set, user_items_map, n_cands=200,
    )

    # Blend and rank
    results = blend_and_rank(sources, target_users, k=TOP_K)
    results = fill_missing(results, target_users, positives, seen_set, user_items_map, k=TOP_K)

    # Create submission
    submission = pd.DataFrame(results)
    submission = submission[["user_id", "edition_id", "rank"]]

    # Validate
    counts = submission.groupby("user_id").size()
    assert (counts == TOP_K).all(), f"Some users don't have exactly {TOP_K} recs"
    assert set(submission["user_id"].unique()) == set(target_users), "Missing target users"

    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSubmission saved to {OUTPUT_FILE}")
    print(f"Shape: {submission.shape}")
    print(f"Users: {submission['user_id'].nunique()}")
    print(submission.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
