"""
Headless UI integration tests for the Streamlit dashboard (`app.py`)
using `streamlit.testing.v1.AppTest`.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = str(REPO_ROOT / "app.py")


def test_streamlit_app_renders_without_exceptions() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 5


def test_streamlit_currency_toggle_and_slider_interaction() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    # Switch display currency from USD to CLP and adjust max supplier share slider
    at.sidebar.radio[0].set_value("CLP").run()
    assert not at.exception

    at.sidebar.slider[0].set_value(35).run()
    assert not at.exception

