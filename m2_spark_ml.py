#!/usr/bin/env python3
# ============================================
# SE446 - Milestone 2: Spark ML Pipeline
# Group Malqa
#
# Tasks 5-6: Saleh Alkhattaf      (ID: 230381)
# Task 7:    Naif Almubarak       (ID: 230434)
# Task 11 (spark-submit glue): Sulaiman Alhammad (ID: 230103)
#
# Run on cluster:
#   spark-submit --master yarn --deploy-mode cluster \
#     --driver-memory 512m --num-executors 1 \
#     --executor-memory 1g --executor-cores 1 \
#     --conf spark.driver.maxResultSize=128m \
#     --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=python3.12 \
#     --conf spark.executorEnv.PYSPARK_PYTHON=python3.12 \
#     m2_spark_ml.py
# ============================================
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, to_timestamp, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)


# ============================================
# Task 6 helper: train one classifier, return metrics + model
# Author: Saleh Alkhattaf (ID: 230381)
# ============================================
def evaluate(name, model_cls, train, test, **params):
    t0 = time.time()
    model = model_cls(labelCol="label", featuresCol="features", **params).fit(train)
    train_s = time.time() - t0
    pred = model.transform(test)
    auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC").evaluate(pred)
    acc = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy").evaluate(pred)
    f1  = MulticlassClassificationEvaluator(labelCol="label", metricName="f1").evaluate(pred)
    pre = MulticlassClassificationEvaluator(labelCol="label", metricName="weightedPrecision").evaluate(pred)
    rec = MulticlassClassificationEvaluator(labelCol="label", metricName="weightedRecall").evaluate(pred)
    cm = pred.groupBy("label", "prediction").count().collect()
    m  = {(int(r["label"]), int(r["prediction"])): r["count"] for r in cm}
    return {
        "model": name,
        "AUC": auc, "Accuracy": acc, "F1": f1, "Precision": pre, "Recall": rec,
        "TN": m.get((0, 0), 0), "FP": m.get((0, 1), 0),
        "FN": m.get((1, 0), 0), "TP": m.get((1, 1), 0),
        "Train_sec": round(train_s, 2),
    }, model


def main():
    spark = SparkSession.builder.appName("M2-Malqa-MLlib").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    print("=== M2 Spark ML Pipeline (Group Malqa) ===")
    print(f"Spark {spark.version}, master={spark.sparkContext.master}")

    # 5% sample fits the cluster's 1g executor (per milestone hint).
    df_raw = (spark.read
              .option("header", True).option("inferSchema", True)
              .csv("hdfs:///data/chicago_crimes.csv")
              .sample(0.05, seed=42))

    df = (df_raw
          .withColumn("Hour", hour(to_timestamp(col("Date"), "MM/dd/yyyy hh:mm:ss a")))
          .withColumn("Hour", when(col("Hour").isNull(), 12).otherwise(col("Hour")))
          .withColumn("label", col("Arrest").cast("integer"))
          .withColumn("Domestic", col("Domestic").cast("string"))   # StringIndexer needs string/numeric
          .na.drop(subset=["Primary Type", "Domestic", "District", "Hour", "label"]))
    print(f"Rows after preprocessing: {df.count():,}")

    # ============================================
    # Task 5: Feature Engineering Pipeline
    # Author: Saleh Alkhattaf (ID: 230381)
    # ============================================
    feature_pipeline = Pipeline(stages=[
        StringIndexer(inputCol="Primary Type", outputCol="crime_index",   handleInvalid="skip"),
        StringIndexer(inputCol="Domestic",     outputCol="domestic_index", handleInvalid="skip"),
        VectorAssembler(
            inputCols=["District", "crime_index", "Hour", "domestic_index"],
            outputCol="features",
        ),
    ])
    prepared = feature_pipeline.fit(df).transform(df).select("features", "label")
    train, test = prepared.randomSplit([0.8, 0.2], seed=42)
    train.cache(); test.cache()
    print(f"Train rows: {train.count():,}    Test rows: {test.count():,}")

    # ============================================
    # Task 6: Train and evaluate LR / RF / GBT
    # Author: Saleh Alkhattaf (ID: 230381)
    # ============================================
    print("\n=== Task 6: Model comparison ===")
    lr_res,  lr_model  = evaluate("LogisticRegression", LogisticRegression,     train, test, maxIter=100, regParam=0.01)
    rf_res,  rf_model  = evaluate("RandomForest",       RandomForestClassifier, train, test, numTrees=100, maxDepth=5)
    gbt_res, gbt_model = evaluate("GBT",                GBTClassifier,          train, test, maxIter=50,  maxDepth=5)
    results = [lr_res, rf_res, gbt_res]
    models  = {"LogisticRegression": lr_model, "RandomForest": rf_model, "GBT": gbt_model}

    headers = ["model", "AUC", "Accuracy", "F1", "Precision", "Recall",
               "TN", "FP", "FN", "TP", "Train_sec"]
    print("  ".join(f"{h:<18}" for h in headers))
    for r in results:
        row = [
            r["model"],
            f"{r['AUC']:.4f}", f"{r['Accuracy']:.4f}", f"{r['F1']:.4f}",
            f"{r['Precision']:.4f}", f"{r['Recall']:.4f}",
            str(r["TN"]), str(r["FP"]), str(r["FN"]), str(r["TP"]),
            f"{r['Train_sec']}",
        ]
        print("  ".join(f"{c:<18}" for c in row))

    # ============================================
    # Task 7: Feature importances + interpretation
    # Author: Naif Almubarak (ID: 230434)
    # ============================================
    print("\n=== Task 7: Random Forest feature importances ===")
    feature_names = ["District", "crime_index", "Hour", "domestic_index"]
    importances = list(rf_model.featureImportances.toArray())
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        bar = "#" * int(imp * 40)
        print(f"  {name:<16} {imp:.4f}  {bar}")
    print(f"  (sum = {sum(importances):.4f})")
    print(
        "\nInterpretation: crime_index dominates -- matches the per-type arrest\n"
        "rates from Task 4 (NARCOTICS ~90% vs THEFT ~10%). Logistic Regression\n"
        "underperforms because the categorical crime IDs are not ordinal, so a\n"
        "linear slope cannot fit per-type behaviour; trees split categories\n"
        "independently and capture the true structure."
    )

    # Save the best model (by AUC) to HDFS
    best_res = max(results, key=lambda r: r["AUC"])
    best_model = models[best_res["model"]]
    save_path = "hdfs:///user/ssalhammad/project/m2/best_model"
    try:
        best_model.write().overwrite().save(save_path)
        print(f"\nBest model: {best_res['model']} (AUC={best_res['AUC']:.4f}) -> {save_path}")
    except Exception as e:
        print(f"Could not save model: {e}")

    spark.stop()
    print("\n=== Job complete ===")


if __name__ == "__main__":
    main()
