from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from scipy.io import loadmat

from nstat.events import Events
from nstat.compat.matlab import Events as MatlabEvents

matplotlib.use("Agg")


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "Events" / "basic.mat"


def _vec(mat: dict[str, object], key: str) -> np.ndarray:
    return np.asarray(mat[key], dtype=float).reshape(-1)


def _cellstr(values: np.ndarray) -> list[str]:
    out: list[str] = []
    for value in np.asarray(values, dtype=object).reshape(-1):
        if isinstance(value, np.ndarray):
            if value.size == 1:
                out.append(str(value.reshape(-1)[0]))
            else:
                out.append("".join(str(v) for v in value.reshape(-1)))
        else:
            out.append(str(value))
    return out


def _cellvec(values: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(v, dtype=float).reshape(-1) for v in np.asarray(values, dtype=object).reshape(-1)]


def _load_fixture() -> dict[str, object]:
    return loadmat(str(FIXTURE), squeeze_me=False, struct_as_record=False)


def test_events_native_roundtrip_matches_matlab_fixture() -> None:
    m = _load_fixture()
    event_times = _vec(m, "event_times")
    event_labels = _cellstr(np.asarray(m["event_labels"], dtype=object))

    native = Events(times=event_times, labels=event_labels)
    assert np.array_equal(native.times, event_times)
    assert native.labels == event_labels
    assert native.color == "r"

    struct_payload = native.to_structure()
    assert np.array_equal(np.asarray(struct_payload["eventTimes"], dtype=float), event_times)
    assert [str(v) for v in struct_payload["eventLabels"]] == event_labels
    assert str(struct_payload["eventColor"]) == "r"

    restored = Events.from_structure(struct_payload)
    assert np.array_equal(restored.times, event_times)
    assert restored.labels == event_labels
    assert restored.color == "r"



def test_events_compat_plot_and_structure_match_matlab_fixture() -> None:
    m = _load_fixture()
    event_times = _vec(m, "event_times")
    event_labels = _cellstr(np.asarray(m["event_labels"], dtype=object))
    plot_axis = _vec(m, "plot_axis")

    compat = MatlabEvents(times=event_times, labels=event_labels)
    payload = compat.toStructure()

    assert set(payload.keys()) == {"eventTimes", "eventLabels", "eventColor"}
    assert np.array_equal(np.asarray(payload["eventTimes"], dtype=float), event_times)
    assert [str(v) for v in payload["eventLabels"]] == event_labels
    assert str(payload["eventColor"]) == "r"

    restored = MatlabEvents.fromStructure(payload)
    assert np.array_equal(restored.eventTimes, event_times)
    assert restored.eventLabels == event_labels
    assert restored.eventColor == "r"

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3), dpi=120)
    ax.axis(plot_axis.tolist())

    lines = compat.plot(ax)
    expected_count = int(np.asarray(m["plot_line_count"]).reshape(-1)[0])
    assert len(lines) == expected_count

    expected_x = _cellvec(np.asarray(m["plot_x_data"], dtype=object))
    expected_y = _cellvec(np.asarray(m["plot_y_data"], dtype=object))
    for idx, line in enumerate(lines):
        np.testing.assert_allclose(np.asarray(line.get_xdata(), dtype=float).reshape(-1), expected_x[idx], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(line.get_ydata(), dtype=float).reshape(-1), expected_y[idx], rtol=0.0, atol=1e-12)

    actual_text = sorted(ax.texts, key=lambda t: t.get_position()[0])
    expected_text = _cellstr(np.asarray(m["text_strings"], dtype=object))
    expected_pos = _cellvec(np.asarray(m["text_positions"], dtype=object))
    assert [t.get_text() for t in actual_text] == expected_text
    for idx, text_artist in enumerate(actual_text):
        np.testing.assert_allclose(
            np.asarray(text_artist.get_position(), dtype=float).reshape(-1),
            expected_pos[idx][:2],
            rtol=0.0,
            atol=1e-12,
        )

    plt.close(fig)
