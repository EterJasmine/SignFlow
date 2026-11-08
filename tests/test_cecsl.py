"""Guard the continuous recognizer's masking, CTC decoding and batching."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.train_cecsl import ContinuousRecognizer, collate, decode, edit_distance, skeleton_vector


def test_missing_skeleton_remains_zero_and_finite():
    hand = SimpleNamespace(hand_landmarks=[], handedness=[])
    pose = SimpleNamespace(pose_landmarks=[])
    features = skeleton_vector(hand, pose, 16 / 9)
    assert features.shape == (309,)
    assert np.isfinite(features).all()
    assert not features.any()


def test_local_hand_shape_is_translation_invariant():
    def result(offset):
        points = [SimpleNamespace(x=.2 + index * .003 + offset, y=.3 + index * .004)
                  for index in range(21)]
        return SimpleNamespace(hand_landmarks=[points],
                               handedness=[[SimpleNamespace(category_name="Right")]])
    pose = SimpleNamespace(pose_landmarks=[])
    first = skeleton_vector(result(0), pose, 1.0)[99:].reshape(2, 21, 5)
    shifted = skeleton_vector(result(.1), pose, 1.0)[99:].reshape(2, 21, 5)
    np.testing.assert_allclose(first[1, :, 2:4], shifted[1, :, 2:4], atol=1e-5)
    assert np.all(first[1, :, 4] == 1)
    assert not first[0].any()


def test_ctc_blank_separates_repeated_words():
    assert decode([0, 2, 2, 0, 2, 3, 0], ["<blank>", "<unk>", "我", "好"]) == ["我", "我", "好"]
    assert edit_distance(["我", "好"], ["我", "我", "好"]) == 1
    assert edit_distance(["我"], []) == 1


def test_padded_frames_do_not_change_valid_predictions():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = ContinuousRecognizer(309, 5, hidden=8).eval()
    first = torch.randn(4, 309)
    second = torch.randn(7, 309)
    features, lengths, targets, target_lengths, _ = collate([
        (first, torch.tensor([2, 3]), {}), (second, torch.tensor([4]), {})])
    with torch.inference_mode():
        together = model(features, lengths)
        alone = model(first[None], torch.tensor([4]))
    assert together.shape == (7, 2, 5)
    assert targets.tolist() == [2, 3, 4]
    assert target_lengths.tolist() == [2, 1]
    torch.testing.assert_close(together[:4, 0], alone[:, 0], atol=1e-6, rtol=1e-5)
