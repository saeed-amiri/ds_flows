# Prometheus & Grafana Course Exam Submission

## Additional Information

For the custom ML monitoring metric, I chose **`data_drift_status`** (from Evidently).
This metric is important because detected drift can explain why model performance decreases.

In Grafana, I configured an alert on **high RMSE** with a threshold of **30**.

For traffic simulation and alert triggering, I implemented dedicated services similar to the evaluation service:

- `traffic-generation` runs `src/traffic_generation/generate_traffic.py`
- `fire-alert` runs `src/trigger_rmse/trigger.py`

To trigger the alert, I run evaluation on **week 3** data, where RMSE exceeds 30.
I reused the evaluation logic and adjusted only the time range.

## Makefile Targets

The following commands are defined:

- `all`: starts API and monitoring stack (`bike-api`, `node-exporter`, `prometheus`, `grafana`)
- `stop`: stops all running services
- `evaluation`: starts the evaluation service (`run_evaluation.py`)
- `traffic-generation`: starts the traffic generation service (`generate_traffic.py`)
- `fire-alert`: starts the RMSE trigger service (`trigger.py`)

## Repository Structure

```text
├── deployment/
│   ├── prometheus/
│   │   ├── prometheus.yml
│   │   └── rules/
│   │       └── alert_rules.yml
│   └── grafana/
│       ├── dashboards/
│       └── provisioning/
├── src/
│   ├── api/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   └── requirements.txt
│   ├── evaluation/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── run_evaluation.py
│   ├── traffic_generation/
│   │   └── generate_traffic.py
│   └── trigger_rmse/
│       └── trigger.py
├── docker-compose.yml
└── Makefile
```

## Exam Context

This project implements a complete monitoring setup for a bike-sharing regression model (`cnt`) based on the Bike Sharing UCI dataset.

Monitored variables:

- Target: `cnt`
- Numerical features: `temp`, `atemp`, `hum`, `windspeed`, `mnth`, `hr`, `weekday`
- Categorical features: `season`, `holiday`, `workingday`, `weathersit`

The objective is to continuously monitor model performance and drift, visualize results in Grafana, and configure alerts for incidents.

## What Was Implemented

### I. Environment and API

- FastAPI API in `src/api/main.py` for bike count prediction.
- Data loading and preprocessing with `_fetch_data` and `_process_data`.
- `RandomForestRegressor` training on January 2011 reference data (`_train_and_predict_reference_model`).
- Model trained once at startup and reused for inference.
- `/predict` endpoint for prediction requests using `BikeSharingInput` schema.
- API image and dependencies defined in `src/api/Dockerfile` and `src/api/requirements.txt`.

### II. Instrumentation and Prometheus

- Metrics implemented with `prometheus_client` and a custom `CollectorRegistry`:
	- `api_requests_total`
	- `api_request_duration_seconds`
	- `model_rmse_score`
	- `model_mae_score`
	- `model_r2_score`
	- `data_drift_status` (custom metric)
- `/evaluate` endpoint:
	- accepts current period data
	- predicts with trained model
	- runs Evidently report with drift/performance metrics
	- extracts RMSE, MAE, R2, and drift status
	- updates Prometheus Gauges/Counters
- `/metrics` endpoint exposed for Prometheus scraping.
- Prometheus configured to scrape:
	- `bike-api`
	- `node-exporter`
- Alert rule configured in `deployment/prometheus/rules/alert_rules.yml` (API down).

### III. Automated Grafana Dashboards

- Grafana runs in Docker Compose.
- Dashboards as code under `deployment/grafana/dashboards/`.
- Provisioning configured to auto-load dashboards at startup.
- Three dashboards included:
	- **API Performance**
	- **Model Performance & Drift**
	- **Infrastructure Overview**

### IV. Alerting

- Prometheus alert: API availability.
- Grafana UI alert: RMSE threshold set to `> 30`.

### V. Traffic Simulation and Evaluation

- `evaluation` service runs `run_evaluation.py` and updates model metrics.
- `traffic-generation` service simulates `/predict` usage.
- `fire-alert` service runs an evaluation slice that pushes RMSE above threshold.

## Reproducibility

Run the full stack with:

```bash
make all
```

Useful commands:

```bash
make evaluation
make traffic-generation
make fire-alert
make stop
```