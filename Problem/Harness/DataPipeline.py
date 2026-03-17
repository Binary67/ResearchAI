import io
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
from dotenv import dotenv_values


class KaggleDataLoader:
    API_BASE = "https://www.kaggle.com/api/v1/datasets/download"

    def __init__(self):
        env = dotenv_values(Path(__file__).resolve().parents[2] / ".env")
        token = env.get("KAGGLE_API_TOKEN")
        if not token:
            raise RuntimeError("KAGGLE_API_TOKEN not found in .env")
        self._headers = {"Authorization": f"Bearer {token}"}

    def download(self, dataset: str) -> dict[str, pd.DataFrame]:
        """Download a Kaggle dataset and return its tabular files as DataFrames.

        Args:
            dataset: Kaggle dataset slug, e.g. "zillow/zecon".

        Returns:
            Dict mapping filenames to DataFrames for all CSV/Excel files in the dataset.
        """
        url = f"{self.API_BASE}/{dataset}"
        req = Request(url, headers=self._headers)
        with urlopen(req) as resp:
            data = resp.read()

        frames = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(tmpdir)

            for path in Path(tmpdir).rglob("*"):
                if not path.is_file():
                    continue
                suffix = path.suffix.lower()
                if suffix == ".csv":
                    frames[path.name] = pd.read_csv(path)
                elif suffix in (".xls", ".xlsx"):
                    frames[path.name] = pd.read_excel(path)

        return frames
