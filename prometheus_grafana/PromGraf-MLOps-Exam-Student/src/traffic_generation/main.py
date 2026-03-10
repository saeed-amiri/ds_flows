"""
Generate traffic to the bike prediction API.
"""

import datetime
import io
import logging
import sys
import zipfile

import pandas as pd
import requests


logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


DATASET_URL: str = (
    "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
)

API_PREDICT_URL: str = "http://bike-api:8080/predict"

NUMERICAL_FEATURES: list[str] = [
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "mnth",
    "hr",
    "weekday",
]

CATEGORICAL_FEATURES: list[str] = [
    "season",
    "holiday",
    "workingday",
    "weathersit",
]

ALL_FEATURES: list[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

DATE_COLUMN: str = "dteday"


def fetch_data() -> pd.DataFrame:
    """Download the bike-sharing dataset."""

    logger.info("Fetching dataset from UCI archive")

    try:
        response: requests.Response = requests.get(
            DATASET_URL,
            verify=False,
            timeout=60,
        )

        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:

            data: pd.DataFrame = pd.read_csv(
                archive.open("hour.csv"),
                parse_dates=[DATE_COLUMN],
            )

        logger.info("Dataset successfully downloaded")

        return data

    except requests.RequestException as exc:

        logger.error("Dataset download failed: %s", exc)
        sys.exit(1)


def process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Create datetime index consistent with training pipeline."""

    logger.info("Processing dataset")

    raw_data["hr"] = raw_data["hr"].astype(int)

    raw_data.index = raw_data.apply(
        lambda row: datetime.datetime.combine(
            row[DATE_COLUMN].date(),
            datetime.time(row.hr),
        ),
        axis=1,
    )

    raw_data = raw_data.sort_index()

    logger.info("Dataset processing complete")

    return raw_data


def prepare_prediction_samples(
    data: pd.DataFrame,
    count: int,
) -> list[dict]:

    """Prepare feature samples for prediction requests."""

    subset: pd.DataFrame = data.loc[
        "2011-01-01 00:00:00":"2011-01-31 23:00:00"]

    if subset.empty:
        logger.warning("No samples available for prediction traffic")
        return []

    if len(subset) < count:

        logger.warning(
            "Requested %s samples but only %s available",
            count,
            len(subset),
        )

        samples = subset[ALL_FEATURES + [DATE_COLUMN]]

    else:

        samples = subset[ALL_FEATURES + [DATE_COLUMN]].sample(
            n=count,
            random_state=42,
        )

    return samples.to_dict(orient="records")


def send_prediction_requests(samples: list[dict]) -> None:
    """Send requests to the prediction API."""

    total: int = len(samples)

    logger.info(
        "Sending %s prediction requests to %s",
        total,
        API_PREDICT_URL,
    )

    for idx, sample in enumerate(samples):

        if idx % 10 == 0:
            logger.info("Request %s / %s", idx + 1, total)

        try:

            payload: Dict = sample.copy()

            date_value = payload.get(DATE_COLUMN)

            if isinstance(date_value, datetime.date):

                payload[DATE_COLUMN] = date_value.strftime("%Y-%m-%d")

            response: requests.Response = requests.post(
                API_PREDICT_URL,
                json=payload,
                timeout=10,
            )

            response.raise_for_status()

        except requests.RequestException as exc:

            logger.warning(
                "Request %s failed: %s",
                idx + 1,
                exc,
            )


def generate_traffic(count: int, dataset: pd.DataFrame) -> None:
    """Generate API traffic."""

    samples: List[Dict] = prepare_prediction_samples(dataset, count)

    if not samples:
        logger.warning("No traffic generated")
        return

    send_prediction_requests(samples)

    logger.info("Traffic generation completed")


def main() -> None:

    data: pd.DataFrame = process_data(fetch_data())

    generate_traffic(
        count=500,
        dataset=data,
    )


if __name__ == "__main__":
    main()
