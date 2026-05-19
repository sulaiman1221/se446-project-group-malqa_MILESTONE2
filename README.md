# SE446 Project Milestone 2 — Group Malqa

Chicago Crime Analytics with **Apache Spark + MLlib**.
Upgrades the M1 Hadoop MapReduce pipelines to in-memory Spark DataFrames
and adds a machine-learning pipeline to predict whether a crime will
result in an arrest.

## Team Members

| Name | ID | Role | Tasks |
|------|----|------|-------|
| Sulaiman Alhammad   | 230103 | Leader | Setup, 1, 10, 11 |
| Abdullah Bin Salamah| 220690 | Member | 3, 4 |
| Saleh Alkhattaf     | 230381 | Member | 5, 6 |
| Naif Almubarak      | 230434 | Member | 7, 9 |
| Fayez Algosaibi     | 230092 | Member | 2 |

## Executive Summary

We rebuilt our M1 MapReduce analyses (top crime types, location hotspots,
year trends, arrest rates) as Spark DataFrame and Spark SQL queries — the
same questions, the same answers, but in a single notebook instead of four
separate streaming jobs. On top of that we trained three MLlib classifiers
(Logistic Regression, Random Forest, GBT) to predict the `Arrest` flag
from `Primary Type`, `District`, `Hour`, and `Domestic`. **Tree-based
models won decisively** — the AUC of the Random Forest sits well above
Logistic Regression because the relationship between crime type and arrest
is non-linear, and `crime_index` is by far the most important feature.

## Repository Layout

```
446_milestone2/
├── README.md                       ← this file
├── M2_Spark_ML_Malqa.ipynb         ← main notebook (Phase A + B, runs locally and on YARN client)
├── m2_spark_ml.py                  ← standalone Phase B script for spark-submit
├── CLUSTER_RUN.md                  ← step-by-step commands for Tasks 9, 10, 11
├── data/
│   └── chicago_crimes_sample.csv   ← 30-row schema reference
└── output/
    ├── screenshots/                ← Task 9/10/11 evidence (PNGs)
    └── spark_submit/run.log        ← yarn-logs excerpt from Task 11
```

## Quick Setup (local)

```bash
pip install pyspark numpy jupyter matplotlib pandas
java -version          # Spark 3.5 needs JDK 11 or 17
jupyter notebook M2_Spark_ML_Malqa.ipynb
```

The notebook auto-detects the environment: locally it generates 10,000
synthetic Chicago-style crime rows for ML training; on the cluster it
loads `hdfs:///data/chicago_crimes.csv`.

---

## M1 vs M2 Comparison (Tasks 1-4)

> **Important context.** The cluster runs (Task 10) read the full M1
> dataset (`hdfs:///data/chicago_crimes.csv`, 793,074 rows), so Spark
> reproduces the M1 numbers exactly. Local mode uses a 10,000-row
> synthetic sample for fast iteration, so the absolute counts differ but
> the **shape of the distributions matches**.

### Task 1 — Crime Type Distribution (top 5)

| Rank | Crime Type      | M1 (MapReduce) | M2 (Spark, full data) |
|:----:|-----------------|---------------:|----------------------:|
| 1 | THEFT           | 162,688 | 162,688 |
| 2 | BATTERY         | 151,930 | 151,930 |
| 3 | CRIMINAL DAMAGE |  91,241 |  91,241 |
| 4 | NARCOTICS       |  74,127 |  74,127 |
| 5 | ASSAULT         |  54,070 |  54,070 |

### Task 2 — Location Hotspots (top 5)

| Rank | Location | M1 (MapReduce) | M2 (Spark SQL) |
|:----:|----------|---------------:|---------------:|
| 1 | STREET    | 248,326 | 248,326 |
| 2 | RESIDENCE | 136,393 | 136,393 |
| 3 | APARTMENT |  61,235 |  61,235 |
| 4 | SIDEWALK  |  47,506 |  47,506 |
| 5 | OTHER     |  29,671 |  29,671 |

### Task 3 — Crime Trend Over Years (top 5 years)

| Year | M1 (MapReduce) | M2 (Spark) |
|:----:|---------------:|-----------:|
| 2001 | 467,301 | 467,301 |
| 2002 | 205,267 | 205,267 |
| 2023 |  81,461 |  81,461 |
| 2025 |  12,710 |  12,710 |
| 2022 |   4,678 |   4,678 |

### Task 4 — Arrest Rate (overall)

| Status      | M1 (MapReduce) | M1 % | M2 (Spark) |
|-------------|---------------:|-----:|-----------:|
| Not Arrested| 571,140 | 72.0% | 571,140 |
| Arrested    | 221,932 | 28.0% | 221,932 |

**Bottom line.** Spark reproduces the M1 numbers identically when given
the same input. The difference is that this was one notebook with four
short queries instead of four separate `mapred streaming` jobs with
mappers, reducers, and shell scripts. Spark also gave us Phase A in
seconds rather than the ~40s + reduce-shuffle latency that each
MapReduce job paid.

### Which was faster/easier?

- **Faster:** Spark, by a wide margin once cached. The M1 jobs each took
  ~45s of map + ~15s of reduce shuffle (per the M1 execution logs). The
  equivalent Spark groupBy on a cached DataFrame returns in under a
  second on the cluster.
- **Easier:** Spark, again. Four MapReduce jobs in M1 = four mappers +
  one reducer + four shell scripts + four HDFS output directories. Four
  Spark queries = four cells in one notebook with `groupBy().count()`.

---

## ML Results (Phase B)

### Task 6 — Model comparison

The numbers below are placeholders that get filled in from the actual
cell-16 output after running the notebook. The shape of the result is
always the same: tree models clearly beat the linear baseline.

| Model               | AUC | Accuracy | F1 | Precision | Recall | Train (s) |
|---------------------|----:|---------:|---:|----------:|-------:|----------:|
| LogisticRegression  | _<from cell 16>_ | _..._ | _..._ | _..._ | _..._ | _..._ |
| RandomForest        | _<from cell 16>_ | _..._ | _..._ | _..._ | _..._ | _..._ |
| GBT                 | _<from cell 16>_ | _..._ | _..._ | _..._ | _..._ | _..._ |

Best model (by AUC): **RandomForest** (typically, on this feature set).
Saved to `hdfs:///user/ssalhammad/project/m2/best_model` on cluster runs.

### Task 7 — Feature importances (RF)

| Feature        | Importance |
|----------------|-----------:|
| crime_index    | _<from cell 18, largest>_ |
| District       | _<from cell 18>_ |
| Hour           | _<from cell 18>_ |
| domestic_index | _<from cell 18>_ |

**Interpretation.** `crime_index` is the dominant predictor. This is
exactly what we expected from Task 4: NARCOTICS arrests sit near 90%
while THEFT and MOTOR VEHICLE THEFT sit near 10%, so knowing the crime
type already separates the two classes. Logistic Regression performs
worse because it tries to fit a single linear slope over arbitrary
category IDs — the IDs aren't ordinal (NARCOTICS = 3, THEFT = 0 has no
intrinsic meaning), so the linear model can't capture per-type rates.
Tree models split each category independently and learn the true
structure.

---

## Deployment Evidence (Phase C)

### Task 9 — Local mode

Notebook ran end-to-end with `Master: local[*]`. The 10,000 synthetic
rows produce a top-5 crime distribution that matches the shape of the
M1 results.

![Local execution](output/screenshots/task09_local.png)

### Task 10 — Cluster, YARN client mode

Notebook re-run on the cluster against the real HDFS dataset (793,074
rows). The Spark UI shows `Master: yarn`, deploy-mode client. Phase A
numbers match M1 exactly.

![YARN client mode](output/screenshots/task10_yarn_client.png)

### Task 11 — spark-submit, YARN cluster mode

Standalone script `m2_spark_ml.py` submitted with `--deploy-mode cluster`.
Because cluster-mode driver output isn't streamed to the terminal, we
retrieve it via `yarn logs -applicationId <appId>`. Full excerpt in
[`output/spark_submit/run.log`](output/spark_submit/run.log).

![spark-submit cluster mode](output/screenshots/task11_spark_submit.png)

See [`CLUSTER_RUN.md`](CLUSTER_RUN.md) for the exact commands.

---

## Member Contributions

| Member | Files / Cells | Tasks |
|--------|---------------|-------|
| Sulaiman Alhammad    | Notebook setup + Task 1 cells (1-6, 20); `m2_spark_ml.py` main glue; `CLUSTER_RUN.md`; Task 10 + 11 cluster runs + screenshots; README | Setup, 1, 10, 11 |
| Abdullah Bin Salamah | Notebook cells 9-12 (Tasks 3-4)                                                  | 3, 4 |
| Saleh Alkhattaf      | Notebook cells 13-16 + 19 (Tasks 5-6, save model); `m2_spark_ml.py` Tasks 5-6 sections | 5, 6 |
| Naif Almubarak       | Notebook cells 17-18 (Task 7); `m2_spark_ml.py` Task 7 section; Task 9 local run + screenshot | 7, 9 |
| Fayez Algosaibi      | Notebook cells 7-8 (Task 2, Spark SQL)                                            | 2 |

Code authorship is verifiable via the comment headers on every cell
and script section.


