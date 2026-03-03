"""
Tests for masking generators.
"""

import numpy as np
import pytest

from data.masking import (
    BlockMaskGenerator,
    FeatureMaskGenerator,
    RandomMaskGenerator,
    create_mask_generator,
)


@pytest.fixture
def sample_orig_mask():
    """Create a sample orig_mask with ~70% observed values."""
    rng = np.random.default_rng(123)
    mask = (rng.random((10, 48, 9)) > 0.3).astype(np.float32)
    return mask


class TestRandomMaskGenerator:
    def test_masks_only_observed(self, sample_orig_mask):
        gen = RandomMaskGenerator(p=0.5)
        art_mask = gen(sample_orig_mask)

        # Artificial mask should only be 1 where orig_mask is 1
        assert np.all(art_mask[sample_orig_mask == 0] == 0)

    def test_approximate_ratio(self, sample_orig_mask):
        gen = RandomMaskGenerator(p=0.3)
        rng = np.random.default_rng(42)
        art_mask = gen(sample_orig_mask, rng=rng)

        observed = sample_orig_mask.sum()
        masked = art_mask.sum()
        ratio = masked / observed

        assert 0.2 < ratio < 0.4, f"Expected ~0.3 ratio, got {ratio:.3f}"

    def test_deterministic_with_seed(self, sample_orig_mask):
        gen = RandomMaskGenerator(p=0.3)

        m1 = gen(sample_orig_mask, rng=np.random.default_rng(42))
        m2 = gen(sample_orig_mask, rng=np.random.default_rng(42))

        np.testing.assert_array_equal(m1, m2)

    def test_different_seeds_differ(self, sample_orig_mask):
        gen = RandomMaskGenerator(p=0.3)

        m1 = gen(sample_orig_mask, rng=np.random.default_rng(42))
        m2 = gen(sample_orig_mask, rng=np.random.default_rng(99))

        assert not np.array_equal(m1, m2)


class TestBlockMaskGenerator:
    def test_masks_only_observed(self, sample_orig_mask):
        gen = BlockMaskGenerator(block_len=5, n_blocks=2)
        art_mask = gen(sample_orig_mask, rng=np.random.default_rng(42))

        assert np.all(art_mask[sample_orig_mask == 0] == 0)

    def test_creates_contiguous_blocks(self):
        # All observed mask for easy verification
        orig_mask = np.ones((1, 48, 9), dtype=np.float32)
        gen = BlockMaskGenerator(block_len=10, n_blocks=1, mask_all_features=True)
        art_mask = gen(orig_mask, rng=np.random.default_rng(42))

        # Find masked timesteps — should be contiguous
        masked_times = np.where(art_mask[0, :, 0] == 1)[0]
        assert len(masked_times) <= 10
        if len(masked_times) > 1:
            diffs = np.diff(masked_times)
            assert np.all(diffs == 1), "Block should be contiguous"


class TestFeatureMaskGenerator:
    def test_masks_single_feature(self):
        orig_mask = np.ones((5, 48, 9), dtype=np.float32)
        gen = FeatureMaskGenerator(feature_idx=4, p_time=0.5)
        art_mask = gen(orig_mask, rng=np.random.default_rng(42))

        # Only feature 4 should be masked
        for d in range(9):
            if d != 4:
                assert art_mask[:, :, d].sum() == 0, f"Feature {d} should not be masked"

        assert art_mask[:, :, 4].sum() > 0, "Feature 4 should be masked"


class TestCreateMaskGenerator:
    def test_random(self):
        gen = create_mask_generator({"type": "random", "p": 0.5})
        assert isinstance(gen, RandomMaskGenerator)

    def test_block(self):
        gen = create_mask_generator({"type": "block", "block_len": 10, "n_blocks": 2})
        assert isinstance(gen, BlockMaskGenerator)

    def test_featurewise(self):
        gen = create_mask_generator({"type": "featurewise", "feature_idx": 3})
        assert isinstance(gen, FeatureMaskGenerator)

    def test_unknown(self):
        with pytest.raises(ValueError):
            create_mask_generator({"type": "unknown"})
