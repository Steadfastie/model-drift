from datetime import date

import pandas as pd

from cross_model_drift.splits import FINAL_HOLDOUT, V1_TRAIN, V2_TEST, select_window, split_summary_rows


def test_v2_test_matches_final_holdout() -> None:
    assert V2_TEST.start == FINAL_HOLDOUT.start
    assert V2_TEST.end == FINAL_HOLDOUT.end


def test_select_window_is_inclusive_on_end_date() -> None:
    frame = pd.DataFrame(
        {
            "created": pd.to_datetime(
                [
                    "2021-05-20 23:59:59",
                    "2021-05-21 00:00:00",
                    "2021-07-21 23:59:59",
                    "2021-07-22 00:00:00",
                ]
            )
        }
    )
    selected = select_window(frame, V1_TRAIN)
    assert list(selected["created"].dt.date) == [date(2021, 5, 21), date(2021, 7, 21)]


def test_split_summary_covers_both_versions() -> None:
    rows = split_summary_rows()
    versions = {row["version"] for row in rows}
    splits = {row["split"] for row in rows}
    assert versions == {"v1", "v2"}
    assert {"train", "validation", "test", "holdout"} <= splits
