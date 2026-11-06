"""Checks whether the 215 QC-selected images are part of the 3,000-image complete pool, using the saved DINOv2
embeddings (afis_results/embeddings/dino_embeddings.npz, written by experiments/afis_analysis.embeddings).
An identical image would have cosine similarity ~1.0; the maximum is compared with the largest similarity between
two different images of the pool. Usage: python check_qc_selected_provenance.py <dino_embeddings.npz>"""
import json
import sys

import numpy as np

z = np.load(sys.argv[1], allow_pickle=True)


def unit(a):
    a = a.astype(np.float64)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


S, P = unit(z["sel_emb"]), unit(z["ntp_emb"])
m = (S @ P.T).max(1)
PP = P @ P.T
np.fill_diagonal(PP, -1)
print(json.dumps(dict(n_selected=len(S), n_pool=len(P), max_cosine_selected_to_pool=float(m.max()),
                      median_max_cosine_selected_to_pool=float(np.median(m)), n_selected_with_cosine_ge_0_9999=int((m >= 0.9999).sum()),
                      max_cosine_between_distinct_pool_images=float(PP.max())), indent=1))
