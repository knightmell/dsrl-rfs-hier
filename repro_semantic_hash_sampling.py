"""Reproduction: do two full buffers with identical bytes at different pos
sample differently?  (docstring claim in hierarchical_replay_buffer.py:517-526)
"""
import pickle

import numpy as np

from stable_baselines3.dsrl.hierarchical_replay_buffer import (
    BranchMode,
    HIERARCHY_SEMANTIC_ARRAY_NAMES,
)
from tests.three_critic_test_utils import make_model, populate_branch


def build_full_buffer():
    model, _ = make_model(n_envs=1, buffer_size=4)
    populate_branch(model, BranchMode.BASE, rows=2)
    populate_branch(model, BranchMode.JOINT, rows=2)
    buf = model.replay_buffer
    assert buf.full, "buffer should be full after 4 rows on buffer_size=4"
    return buf


A = build_full_buffer()
assert A.pos == 0, f"expected pos==0 after full wrap, got {A.pos}"

# Byte-identical copy via pickle (all numpy arrays copied verbatim).
B = pickle.loads(pickle.dumps(A))
B.pos = 2  # rotate next-write cursor without touching any stored byte
assert B.full

# Verify byte-identity of every semantic array + valid mask.
diffs = []
for name in HIERARCHY_SEMANTIC_ARRAY_NAMES:
    if not np.array_equal(getattr(A, name), getattr(B, name)):
        diffs.append(name)
for name in ("_row_valid",):
    if not np.array_equal(getattr(A, name), getattr(B, name)):
        diffs.append(name)
print("non-identical arrays:", diffs)
print("A.pos, B.pos, A.full, B.full:", A.pos, B.pos, A.full, B.full)

# The whole-buffer hash SHOULD differ (pos is in the header).
ha, hb = A.semantic_hash(), B.semantic_hash()
print("semantic_hash A == B:", ha == hb)

# Same RNG seed -> do samples differ?
np.random.seed(1234)
da = A.sample_any(200)
np.random.seed(1234)
db = B.sample_any(200)
print(
    "sample_any episode_id draws identical:",
    bool((da.episode_id == db.episode_id).all().item()),
)
print(
    "sample_any observations identical:",
    bool((da.observations == db.observations).all().item()),
)

np.random.seed(99)
ea = A.sample_branch(BranchMode.JOINT, 200)
np.random.seed(99)
eb = B.sample_branch(BranchMode.JOINT, 200)
print(
    "sample_branch(JOINT) episode_id draws identical:",
    bool((ea.episode_id == eb.episode_id).all().item()),
)

# Also confirm _eligible_pairs output is identical (row ordering, count).
pa = A._eligible_pairs(None)
pb = B._eligible_pairs(None)
print("_eligible_pairs shapes equal:", pa.shape == pb.shape,
      "| identical content:", bool((pa == pb).all()))

# Demonstrate the hash difference is *only* due to pos/full in the header:
# strip header influence by hashing with vector_rows (prefix mode omits pos/full).
print("prefix hash (vector_rows=4) equal:", A.semantic_hash(vector_rows=4) == B.semantic_hash(vector_rows=4))
