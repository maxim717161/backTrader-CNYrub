from pathlib import Path

import pandas as pd

import plot


def test_plot_writes_png(tmp_path: Path):
    stamps = pd.to_datetime(
        [
            "2022-04-21 10:00:00",
            "2022-04-21 10:01:00",
            "2022-04-22 10:00:00",
            "2022-06-17 10:00:00",
        ]
    )
    frame = pd.DataFrame(
        {
            "datetime": stamps,
            "high": [10.2, 10.4, 10.5, 11.0],
            "low": [10.0, 10.1, 10.2, 10.7],
            "close": [10.1, 10.3, 10.4, 10.9],
            "volume": [1, 2, 3, 4],
            "secid": ["CRM2", "CRM2", "CRM2", "CRU2"],
        }
    )
    output = tmp_path / "cny_front.png"
    written = plot.plot_series(frame, output)
    assert written == output
    assert output.stat().st_size > 1000
