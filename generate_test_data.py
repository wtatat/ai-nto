"""Generate synthetic test data mimicking the competition format for testing."""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

N_USERS = 500
N_AUTHORS = 100
N_BOOKS = 800
N_EDITIONS = 1000
N_GENRES = 20
TOTAL_DAYS = 180
INCIDENT_DAYS = 30

# Users
users = pd.DataFrame({
    "user_id": range(1, N_USERS + 1),
    "gender": np.random.choice([1, 2, np.nan], N_USERS, p=[0.4, 0.4, 0.2]),
    "age": np.random.choice(
        list(range(10, 70)) + [np.nan] * 10,
        N_USERS,
    ),
})
users.to_csv(DATA_DIR / "users.csv", index=False)

# Authors
authors = pd.DataFrame({
    "author_id": range(1, N_AUTHORS + 1),
    "author_name": [f"Author_{i}" for i in range(1, N_AUTHORS + 1)],
})
authors.to_csv(DATA_DIR / "authors.csv", index=False)

# Genres
genres = pd.DataFrame({
    "genre_id": range(1, N_GENRES + 1),
    "genre_name": [f"Genre_{i}" for i in range(1, N_GENRES + 1)],
})
genres.to_csv(DATA_DIR / "genres.csv", index=False)

# Editions
editions = pd.DataFrame({
    "edition_id": range(1, N_EDITIONS + 1),
    "book_id": np.random.randint(1, N_BOOKS + 1, N_EDITIONS),
    "author_id": np.random.randint(1, N_AUTHORS + 1, N_EDITIONS),
    "publication_year": np.random.randint(1950, 2025, N_EDITIONS),
    "age_restriction": np.random.choice([0, 6, 12, 16, 18], N_EDITIONS),
    "language_id": np.random.randint(1, 5, N_EDITIONS),
    "publisher_id": np.random.randint(1, 50, N_EDITIONS),
    "title": [f"Book Title {i}" for i in range(1, N_EDITIONS + 1)],
    "description": [f"Description for book {i}" for i in range(1, N_EDITIONS + 1)],
})
editions.to_csv(DATA_DIR / "editions.csv", index=False)

# Book genres
book_genre_pairs = set()
for bid in range(1, N_BOOKS + 1):
    n_genres = np.random.randint(1, 4)
    for gid in np.random.choice(range(1, N_GENRES + 1), n_genres, replace=False):
        book_genre_pairs.add((bid, int(gid)))
book_genres = pd.DataFrame(list(book_genre_pairs), columns=["book_id", "genre_id"])
book_genres.to_csv(DATA_DIR / "book_genres.csv", index=False)

# Interactions
start_date = pd.Timestamp("2025-06-01")
interactions = []
for uid in range(1, N_USERS + 1):
    n_events = np.random.randint(5, 50)
    for _ in range(n_events):
        eid = np.random.randint(1, N_EDITIONS + 1)
        event_type = np.random.choice([1, 2], p=[0.3, 0.7])
        day_offset = np.random.randint(0, TOTAL_DAYS)
        ts = start_date + pd.Timedelta(days=day_offset, hours=np.random.randint(0, 24))
        rating = round(np.random.uniform(1, 10), 1) if event_type == 2 else None
        interactions.append({
            "user_id": uid,
            "edition_id": eid,
            "event_type": event_type,
            "rating": rating,
            "event_ts": ts,
        })

interactions_df = pd.DataFrame(interactions)
interactions_df.to_csv(DATA_DIR / "interactions.csv", index=False)

# Targets - subset of users
target_users = np.random.choice(range(1, N_USERS + 1), size=200, replace=False)
targets = pd.DataFrame({"user_id": sorted(target_users)})
targets.to_csv(DATA_DIR / "targets.csv", index=False)

print(f"Generated test data in {DATA_DIR}/")
print(f"  Users: {N_USERS}")
print(f"  Authors: {N_AUTHORS}")
print(f"  Editions: {N_EDITIONS}")
print(f"  Genres: {N_GENRES}")
print(f"  Interactions: {len(interactions_df)}")
print(f"  Target users: {len(targets)}")
