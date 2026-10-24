"""R-1: the committed split file is leak-free and reproducible from the split code."""

import pandas as pd
import pytest

from src import config
from src.data_utils import splits, store


@pytest.fixture(scope="module")
def manifest():
    return store.load_manifest()


@pytest.fixture(scope="module")
def v2():
    return pd.read_parquet(config.SPLITS_DIR / "v2.parquet")


def test_split_file_passes_checks(manifest, v2):
    splits.check_splits(v2, manifest)


def test_split_file_covers_the_corpus(manifest, v2):
    assert set(v2["id"]) == set(manifest["id"])


def test_split_file_matches_make_splits(manifest, v2):
    fresh = splits.make_splits(manifest)
    assert fresh["id"].tolist() == v2["id"].tolist()
    assert fresh["split"].astype(str).tolist() == v2["split"].astype(str).tolist()


def test_ood_is_whole_mixtures(manifest, v2):
    m = manifest.set_index("id").loc[v2.loc[v2["split"] == "ood", "id"], "mixture"]
    assert m.nunique() == config.SplitConfig().n_ood_mixtures


def test_fractions(v2):
    pool = v2[v2["split"] != "ood"]
    frac = pool["split"].astype(str).value_counts(normalize=True)
    assert frac["train"] == pytest.approx(0.8, abs=0.01)
    assert frac["val"] == pytest.approx(0.1, abs=0.01)
    assert frac["test"] == pytest.approx(0.1, abs=0.01)


def test_check_catches_a_split_group(manifest, v2):
    # Move one member of a multi-row train group into test: the group now spans two splits.
    groups = splits.group_ids(manifest.set_index("id").loc[v2["id"]].reset_index())
    train = (v2["split"] == "train").to_numpy()
    big = groups[train].value_counts()
    member = groups[train & (groups == big.index[0]).to_numpy()].index[0]
    broken = v2.copy()
    broken.loc[member, "split"] = "test"
    with pytest.raises(AssertionError, match="span two splits"):
        splits.check_splits(broken, manifest)
