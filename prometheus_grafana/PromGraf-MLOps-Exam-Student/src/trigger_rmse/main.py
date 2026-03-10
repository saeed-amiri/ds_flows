"""Script to trigger the high RMSE alert via the /evaluate API endpoint."""


import datetime
import io
import json
import sys
import warnings
import zipfile

import pandas as pd
import requests


warnings.filterwarnings("ignore", message="Unverified HTTPS request")


DATASET_URL: str = (
    "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
)

API_EVALUATE_URL: str = "http://bike-api:8080/evaluate"

EVALUATION_SAMPLE_SIZE: int = 1000

WEEKLY_PERIODS: dict[str, tuple[str, str]] = {
    "week1_february": ("2011-01-29 00:00:00", "2011-02-07 23:00:00"),
    "week2_february": ("2011-02-08 00:00:00", "2011-02-14 23:00:00"),
    "week3_february": ("2011-02-15 00:00:00", "2011-02-21 23:00:00"),
}

DEFAULT_PERIOD_NAME: str = "week3_february"
DEFAULT_PERIOD: tuple[str, str] = WEEKLY_PERIODS[DEFAULT_PERIOD_NAME]


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

TARGET: str = "cnt"

DATE_COLUMN: str = "dteday"

ALL_FEATURES: list[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

PAYLOAD_COLUMNS: list[str] = ALL_FEATURES + [TARGET, DATE_COLUMN]


def fetch_dataset() -> pd.DataFrame:
    """Download the bike sharing dataset."""

    print("Downloading dataset...")

    try:
        response: requests.Response = requests.get(
            DATASET_URL,
            verify=False,
            timeout=60,
        )

        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            dataset: pd.DataFrame = pd.read_csv(
                archive.open("hour.csv"),
                parse_dates=[DATE_COLUMN],
            )

        print("Dataset downloaded.")
        return dataset

    except requests.exceptions.RequestException as exc:
        print(f"Dataset download failed: {exc}")
        sys.exit(1)


def process_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Align dataset structure with the API pipeline."""

    print("Processing dataset...")

    data["hr"] = data["hr"].astype(int)

    data.index = data.apply(
        lambda row: datetime.datetime.combine(
            row[DATE_COLUMN].date(),
            datetime.time(row.hr),
        ),
        axis=1,
    )

    data.sort_index(inplace=True)

    print("Processing completed.")
    return data


def run_evaluation(
    dataset: pd.DataFrame,
    period_name: str,
    start_date: str,
    end_date: str,
) -> None:
    """Send evaluation data to the API."""

    print(
        f"\nRunning evaluation for period: {period_name} "
        f"({start_date} → {end_date})"
    )

    try:

        period_data: pd.DataFrame = dataset.loc[start_date:end_date]

        if period_data.empty:
            print("No data available for this period.")
            return

        evaluation_df: pd.DataFrame = period_data[PAYLOAD_COLUMNS]

        if evaluation_df.shape[0] > EVALUATION_SAMPLE_SIZE:
            evaluation_df = evaluation_df.sample(
                n=EVALUATION_SAMPLE_SIZE,
                random_state=42,
            )

        evaluation_df[DATE_COLUMN] = evaluation_df[DATE_COLUMN].astype(str)

        payload: list[dict] = evaluation_df.to_dict(orient="records")

        print(f"Sending {len(payload)} samples to {API_EVALUATE_URL}")

        response: requests.Response = requests.post(
            API_EVALUATE_URL,
            json={
                "data": payload,
                "evaluation_period_name": period_name,
            },
            timeout=300,
        )

        response.raise_for_status()

        result: dict = response.json()

        print("\nEvaluation completed")

        print(f"Message: {result.get('message')}")

        rmse: float | None = result.get("rmse")
        mape: float | None = result.get("mape")

        print(f"RMSE: {rmse:.4f}" if rmse else "RMSE: N/A")
        print(f"MAPE: {mape:.4f}" if mape else "MAPE: N/A")

        drift: int = result.get("drift_detected", 0)

        print(f"Data drift detected: {'Yes' if drift == 1 else 'No'}")

        print(f"Evaluated items: {result.get('evaluated_items')}")

    except requests.exceptions.RequestException as exc:
        print(f"Evaluation request failed: {exc}")

    except json.JSONDecodeError:
        print(f"Invalid JSON response: {response.text}")

    except Exception as exc:
        print(f"Unexpected error: {exc}")


def main() -> None:
    """Entry point."""

    dataset: pd.DataFrame = fetch_dataset()
    dataset = process_dataset(dataset)

    start_date, end_date = DEFAULT_PERIOD

    run_evaluation(
        dataset=dataset,
        period_name=DEFAULT_PERIOD_NAME,
        start_date=start_date,
        end_date=end_date,
    )


if __name__ == "__main__":
    main()
