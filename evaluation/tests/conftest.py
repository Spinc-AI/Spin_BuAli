"""Put the evaluation modules on the import path."""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from extractors import ClinicalTerms  # noqa: E402


@pytest.fixture(scope="session")
def terms():
    return ClinicalTerms()
