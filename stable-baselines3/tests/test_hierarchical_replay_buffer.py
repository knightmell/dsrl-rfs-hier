import pickle

import numpy as np
import pytest

from stable_baselines3.dsrl.hierarchical_replay_buffer import BranchMode
from tests.three_critic_test_utils import make_model, metadata_row, populate_branch


def test_tagged_replay_stores_all_fields_and_filters_flat_env_slots():
    model, _ = make_model(n_envs=2, buffer_size=12)
    metadata, dones, infos = metadata_row(model, BranchMode.BASE, row=1)
    metadata["branch_mode"][1] = int(BranchMode.JOINT)
    metadata["residual_applied"][1] = True
    metadata["residual_policy_version"][1] = 0
    model.replay_buffer.add_hierarchy(
        np.zeros((2, 3), np.float32),
        np.ones((2, 3), np.float32),
        metadata["action_exec"],
        np.array([1.0, 2.0], np.float32),
        dones,
        infos,
        metadata=metadata,
    )
    assert model.replay_buffer.branch_count(BranchMode.BASE) == 1
    assert model.replay_buffer.branch_count(BranchMode.JOINT) == 1
    base = model.replay_buffer.sample_branch(BranchMode.BASE, 16)
    joint = model.replay_buffer.sample_branch(BranchMode.JOINT, 16)
    assert set(base.branch_mode.flatten().tolist()) == {int(BranchMode.BASE)}
    assert set(joint.branch_mode.flatten().tolist()) == {int(BranchMode.JOINT)}
    assert base.actions.shape == base.action_exec.shape == (16, 4)


def test_standard_add_without_staged_metadata_fails():
    model, _ = make_model()
    with pytest.raises(RuntimeError, match="stage_metadata"):
        model.replay_buffer.add(
            np.zeros((1, 3), np.float32),
            np.zeros((1, 3), np.float32),
            np.zeros((1, 4), np.float32),
            np.zeros(1, np.float32),
            np.zeros(1, np.bool_),
            [{}],
        )


def test_base_invariants_reject_nonzero_residual():
    model, _ = make_model()
    metadata, dones, infos = metadata_row(model, BranchMode.BASE)
    metadata["residual_unit"][0, 0] = 0.1
    with pytest.raises(ValueError, match="exact-zero residual_unit"):
        model.replay_buffer.add_hierarchy(
            np.zeros((1, 3), np.float32),
            np.zeros((1, 3), np.float32),
            metadata["action_exec"],
            np.zeros(1, np.float32),
            dones,
            infos,
            metadata=metadata,
        )


@pytest.mark.parametrize("field", ["noise_log_prob", "beta"])
def test_float_scalar_metadata_rejects_non_finite_values(field):
    model, _ = make_model()
    metadata, dones, infos = metadata_row(model, BranchMode.BASE)
    metadata[field][0] = np.nan
    with pytest.raises(ValueError, match=field):
        model.replay_buffer.add_hierarchy(
            np.zeros((1, 3), np.float32),
            np.zeros((1, 3), np.float32),
            metadata["action_exec"],
            np.zeros(1, np.float32),
            dones,
            infos,
            metadata=metadata,
        )


def test_circular_overwrite_counts_and_pickle_restore():
    model, _ = make_model(buffer_size=4)
    populate_branch(model, BranchMode.BASE, rows=2)
    populate_branch(model, BranchMode.JOINT, rows=2)
    assert model.replay_buffer.full
    assert model.replay_buffer.branch_count(BranchMode.BASE) == 2
    assert model.replay_buffer.branch_count(BranchMode.JOINT) == 2
    populate_branch(model, BranchMode.JOINT, rows=1)
    assert model.replay_buffer.branch_count(BranchMode.BASE) == 1
    assert model.replay_buffer.branch_count(BranchMode.JOINT) == 3
    restored = pickle.loads(pickle.dumps(model.replay_buffer))
    assert restored.branch_counts == model.replay_buffer.branch_counts
    assert restored.semantic_hash() == model.replay_buffer.semantic_hash()


def test_semantic_hash_covers_metadata_tampering():
    model, _ = make_model()
    populate_branch(model, BranchMode.BASE, rows=2)
    before = model.replay_buffer.semantic_hash()
    model.replay_buffer.noise_policy_version[0, 0] += 1
    assert model.replay_buffer.semantic_hash() != before


def test_timeout_terminal_truth_is_canonical():
    model, _ = make_model()
    metadata, dones, infos = metadata_row(model, BranchMode.BASE, done=True)
    infos[0]["TimeLimit.truncated"] = True
    model.replay_buffer.add_hierarchy(
        np.zeros((1, 3), np.float32),
        np.ones((1, 3), np.float32),
        metadata["action_exec"],
        np.zeros(1, np.float32),
        dones,
        infos,
        metadata=metadata,
    )
    assert bool(model.replay_buffer.truncated[0, 0])
    assert not bool(model.replay_buffer.terminated[0, 0])
    sample = model.replay_buffer.sample_branch(BranchMode.BASE, 1)
    assert float(sample.dones.item()) == 0.0


def test_timeout_without_done_is_rejected():
    # Regression: the truncation guard used to derive `truncated` from
    # `done_bool`, so `truncated & ~done_bool` could never fire -- a timeout
    # whose done bit was unset was silently accepted and mis-stored as a
    # terminal transition.  `truncated` now carries the wrapper's
    # TimeLimit.truncated flag independently, making the invariant reachable.
    model, _ = make_model()
    metadata, dones, infos = metadata_row(model, BranchMode.BASE, done=False)
    infos[0]["TimeLimit.truncated"] = True
    with pytest.raises(ValueError, match="timeout_without_done"):
        model.replay_buffer.add_hierarchy(
            np.zeros((1, 3), np.float32),
            np.ones((1, 3), np.float32),
            metadata["action_exec"],
            np.zeros(1, np.float32),
            dones,
            infos,
            metadata=metadata,
        )


def test_semantic_hash_covers_row_valid_mask():
    # Regression: the digest covered only the stored arrays, so two buffers
    # holding identical data but valid on different rows hashed identically.
    # The valid-row mask is now part of replay semantics.
    model, _ = make_model(buffer_size=8)
    populate_branch(model, BranchMode.BASE, rows=2)
    before = model.replay_buffer.semantic_hash()
    # Same stored arrays, different valid-row mask within the hashed prefix.
    model.replay_buffer._row_valid[0] = False
    after = model.replay_buffer.semantic_hash()
    assert before != after


def test_whole_buffer_semantic_hash_covers_pos_rotation_on_full_buffer():
    # Regression: on a full buffer the physical row order is fixed while `pos`
    # rotates, so two buffers holding identical bytes at different `pos` sample
    # differently and must hash differently.  `pos`/`full` used to be absent
    # from the header, making every rotation of a full buffer hash identically.
    model, _ = make_model(buffer_size=4)
    populate_branch(model, BranchMode.BASE, rows=2)
    populate_branch(model, BranchMode.JOINT, rows=2)
    assert model.replay_buffer.full
    assert model.replay_buffer.pos == 0
    at_pos_zero = model.replay_buffer.semantic_hash()
    # Rotate the next-write index without touching any stored byte.
    model.replay_buffer.pos = 1
    at_pos_one = model.replay_buffer.semantic_hash()
    assert at_pos_zero != at_pos_one


def test_prefix_semantic_hash_is_rotation_invariant_across_pos_flip():
    # The P6 resume gate compares the immutable prefill prefix hash computed on
    # a non-full buffer at prefill time against the same prefix recomputed after
    # online training (still non-wrapped).  The prefix rows are physical rows
    # [0:vector_rows] and do not move when `pos` rotates, so the prefix digest
    # must NOT include pos/full.  Regression: a naive "add pos to the header"
    # fix broke that cross-state comparison.
    model, _ = make_model(buffer_size=8)
    populate_branch(model, BranchMode.BASE, rows=3)
    populate_branch(model, BranchMode.JOINT, rows=3)
    assert not model.replay_buffer.full
    assert model.replay_buffer.pos == 6
    before = model.replay_buffer.semantic_hash(vector_rows=3)
    model.replay_buffer.pos = 4
    after_flip = model.replay_buffer.semantic_hash(vector_rows=3)
    assert before == after_flip
    # The whole-buffer digest must still track pos/full (pos is part of replay
    # semantics), so flipping pos between two whole-buffer hashes changes the
    # digest.  Comparing against the 3-row prefix hash would be trivially true
    # (row counts differ), so compare whole-buffer against whole-buffer.
    at_pos_four = model.replay_buffer.semantic_hash()
    model.replay_buffer.pos = 6
    at_pos_six = model.replay_buffer.semantic_hash()
    assert at_pos_four != at_pos_six
