from pathlib import Path

REQUIRED_HELP_CLASSES = [
    "Analysis",
    "CIF",
    "ConfidenceInterval",
    "ConfigColl",
    "CovColl",
    "Covariate",
    "DecodingAlgorithms",
    "Events",
    "FitResSummary",
    "FitResult",
    "History",
    "SignalObj",
    "Trial",
    "TrialConfig",
    "nspikeTrain",
    "nstColl",
]


def test_class_help_pages_exist() -> None:
    base = Path("docs/help/classes")
    for klass in REQUIRED_HELP_CLASSES:
        page = base / f"{klass}.md"
        assert page.exists(), f"missing class help page: {page}"
