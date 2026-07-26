"""Shared fixtures.

Every test runs against a throwaway warehouse under ``tmp_path``. ``GLASSBOX_ROOT``
is set for the duration so that any code path resolving the root implicitly
(rather than taking it as an argument) still lands in the sandbox — otherwise a
single missed argument would have tests writing into the developer's real
warehouse.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from glassbox.bootstrap import bootstrap
from glassbox.catalog import load_catalog

# A representative slice of Adult's grammar: the leading-space formatting, '?'
# for missing values, both labels, and — importantly — an exact duplicate row,
# which is what forces subject ids to carry an occurrence index.
_WORKCLASS = ["Private", "Self-emp-not-inc", "Local-gov", "?"]
_EDUCATION = [("Bachelors", 13), ("HS-grad", 9), ("Masters", 14), ("11th", 7)]
_MARITAL = ["Never-married", "Married-civ-spouse", "Divorced"]
_OCCUPATION = ["Tech-support", "Craft-repair", "Exec-managerial", "?"]
_RELATIONSHIP = ["Not-in-family", "Husband", "Wife", "Own-child"]
_RACE = ["White", "Black", "Asian-Pac-Islander"]
_SEX = ["Male", "Female"]
_COUNTRY = ["United-States", "Mexico", "?"]

DUPLICATE_ROW = (
    "39, Private, 77516, Bachelors, 13, Never-married, Tech-support, "
    "Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K"
)


def _synthetic_adult_lines(n: int = 300, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    lines = []
    for _ in range(n):
        edu, edu_num = rng.choice(_EDUCATION)
        sex = rng.choice(_SEX)
        # Give sex a real association with the label so fairness metrics computed
        # in later phases have signal rather than noise.
        p_high = 0.35 if sex == "Male" else 0.15
        label = ">50K" if rng.random() < p_high else "<=50K"
        lines.append(
            ", ".join(
                [
                    str(rng.randint(17, 78)),
                    rng.choice(_WORKCLASS),
                    str(rng.randint(20000, 300000)),
                    edu,
                    str(edu_num),
                    rng.choice(_MARITAL),
                    rng.choice(_OCCUPATION),
                    rng.choice(_RELATIONSHIP),
                    rng.choice(_RACE),
                    sex,
                    str(rng.choice([0, 0, 0, 2174, 5013])),
                    str(rng.choice([0, 0, 0, 1902])),
                    str(rng.randint(5, 70)),
                    rng.choice(_COUNTRY),
                    label,
                ]
            )
        )
    # Two byte-identical rows: distinct people who happen to share every feature.
    lines.append(DUPLICATE_ROW)
    lines.append(DUPLICATE_ROW)
    return lines


@pytest.fixture
def gb_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GLASSBOX_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture
def adult_file(gb_root: Path) -> Path:
    """A small synthetic adult.data, placed where the real downloader would cache it."""
    dest = gb_root / "data" / "raw" / "adult.data"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(_synthetic_adult_lines()) + "\n", encoding="utf-8")
    return dest


@pytest.fixture
def catalog(gb_root: Path):
    bootstrap(gb_root)
    return load_catalog(gb_root)
