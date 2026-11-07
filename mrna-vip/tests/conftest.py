import pandas as pd
import pytest

from src import config
from src.data_pipeline import load_and_clean


@pytest.fixture(scope="session")
def clean_df() -> pd.DataFrame:
    return load_and_clean(config.RAW_CSV)


@pytest.fixture
def patient() -> dict:
    return {"Patient_ID": "T001", "Age": 58, "Biomarker_Level": 4.0, "Pre_Existing_Conditions": 2,
            "Vaccine_Dose_mcg": 60.0, "Prior_Reaction_History": "None"}


@pytest.fixture(scope="session")
def index(tmp_path_factory):
    from src.rag_index import GuidelineIndex
    d = tmp_path_factory.mktemp("store")
    return GuidelineIndex.build(persist_dir=d, backend="offline", log_path=d / "log.jsonl")
