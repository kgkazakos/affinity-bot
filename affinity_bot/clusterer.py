"""TF-IDF + KMeans clustering — the rule-based core of affinity-bot.

The pipeline is deliberately small and easy to reason about:

1. Vectorise notes with TF-IDF (English stopwords, unigrams + bigrams,
   ``min_df=1``, capped vocabulary).
2. L2-normalise so KMeans on the resulting space approximates cosine
   clustering on the original vectors.
3. Pick ``k`` automatically if not given — by inertia elbow, clamped to
   ``[min_clusters, max_clusters]`` and never larger than ``n_notes // 2``.
4. Run KMeans with ``random_state`` from config; assign each note to its
   nearest centroid.
5. Flag any note whose cosine distance to its centroid exceeds
   ``outlier_distance_threshold`` as an outlier (cluster_id == -1).

Outputs are intentionally light: just the cluster id per note, the
centroid weights, and the fitted vocabulary. The labeler turns those into
human-facing labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from affinity_bot.models import AffinityConfig, Note

_OUTLIER_ID = -1


@dataclass
class ClusterAssignment:
    """The raw output of the clustering step — pre-labelling."""

    note_ids: list[str]
    labels: list[int]                   # cluster_id per note; -1 for outliers
    centroids: np.ndarray               # (n_clusters, n_features), L2-normalised
    vocabulary: list[str]               # ordered feature names
    chosen_k: int


def cluster(notes: list[Note], config: AffinityConfig) -> ClusterAssignment:
    """Cluster ``notes`` into themes. Returns assignments + centroids."""
    if not notes:
        return ClusterAssignment([], [], np.zeros((0, 0)), [], 0)

    texts = [n.text for n in notes]
    note_ids = [n.id for n in notes]

    # 1) Vectorise — unigrams + bigrams, English stopwords, capped vocab
    max_features = min(2000, max(50, len(notes) * 30))
    vectoriser = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=max_features,
        min_df=1,
        lowercase=True,
        sublinear_tf=True,
    )
    try:
        matrix = vectoriser.fit_transform(texts)
    except ValueError:
        # All-stopword corpus or no usable tokens; cannot cluster meaningfully.
        # sklearn raises "empty vocabulary; perhaps the documents only contain stop words".
        return ClusterAssignment(
            note_ids=note_ids,
            labels=[_OUTLIER_ID] * len(notes),
            centroids=np.zeros((0, 0)),
            vocabulary=[],
            chosen_k=0,
        )

    if matrix.shape[1] == 0:
        return ClusterAssignment(
            note_ids=note_ids,
            labels=[_OUTLIER_ID] * len(notes),
            centroids=np.zeros((0, 0)),
            vocabulary=[],
            chosen_k=0,
        )

    # 2) Normalise so KMeans approximates cosine
    matrix_dense = normalize(matrix).toarray()

    # 3) Pick k
    k = _choose_k(matrix_dense, config)

    # 4) Cluster
    km = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=config.random_state,
    )
    raw_labels = km.fit_predict(matrix_dense).tolist()
    centroids = normalize(km.cluster_centers_)

    # 5) Flag outliers by cosine distance to their assigned centroid
    final_labels = list(raw_labels)
    for i, lab in enumerate(raw_labels):
        dist = 1.0 - float(np.dot(matrix_dense[i], centroids[lab]))
        if dist > config.outlier_distance_threshold:
            final_labels[i] = _OUTLIER_ID

    return ClusterAssignment(
        note_ids=note_ids,
        labels=final_labels,
        centroids=centroids,
        vocabulary=list(vectoriser.get_feature_names_out()),
        chosen_k=k,
    )


def _choose_k(matrix: np.ndarray, config: AffinityConfig) -> int:
    """Pick k automatically (if not pinned in config)."""
    n_notes = matrix.shape[0]
    if config.n_clusters is not None:
        return max(2, min(config.n_clusters, max(2, n_notes - 1)))

    hard_max = max(2, min(config.max_clusters, n_notes - 1, n_notes // 2 or 2))
    hard_min = max(2, min(config.min_clusters, hard_max))

    if hard_min == hard_max:
        return hard_min

    # Elbow on inertia: pick the k where marginal improvement falls below
    # 20% of the largest single-step improvement seen.
    inertias: list[Optional[float]] = []
    for k in range(hard_min, hard_max + 1):
        try:
            km = KMeans(n_clusters=k, n_init=5, random_state=config.random_state)
            km.fit(matrix)
            inertias.append(float(km.inertia_))
        except Exception:  # noqa: BLE001
            inertias.append(None)

    drops: list[float] = []
    for i in range(1, len(inertias)):
        prev, curr = inertias[i - 1], inertias[i]
        if prev is None or curr is None:
            drops.append(0.0)
        else:
            drops.append(max(0.0, prev - curr))
    if not drops or max(drops) == 0.0:
        return hard_min
    threshold = max(drops) * 0.2
    for i, d in enumerate(drops):
        if d < threshold:
            return hard_min + i
    return hard_max


def top_terms_for_cluster(
    cluster_id: int,
    assignment: ClusterAssignment,
    top_k: int,
) -> list[str]:
    """Return the highest-weighted features for a cluster centroid."""
    if cluster_id < 0 or cluster_id >= assignment.centroids.shape[0]:
        return []
    centroid = assignment.centroids[cluster_id]
    if centroid.size == 0:
        return []
    top_indices = np.argsort(-centroid)[:top_k]
    return [assignment.vocabulary[i] for i in top_indices if centroid[i] > 0]
