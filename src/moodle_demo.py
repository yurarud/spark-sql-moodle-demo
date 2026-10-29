"""Reproducible Spark SQL lecture demo with synthetic Moodle events."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
from datetime import date, datetime, timezone
import io
import json
from pathlib import Path
import statistics
from time import perf_counter
from typing import Any

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, Window, functions as F

ROOT = Path(__file__).resolve().parents[1]
EVENT_SCHEMA = (
    "event_id LONG, user_id LONG, course_id LONG, "
    "event_type STRING, event_time STRING"
)
COURSE_SCHEMA = "course_id LONG, course_name STRING"
DAILY_COLS = ["course_id", "day", "events", "users"]


def json_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def records(df: DataFrame, columns: list[str], order: list[str]) -> list[dict]:
    """Only for tiny, explicitly bounded lecture summaries."""
    return [
        {k: json_value(v) for k, v in row.asDict().items()}
        for row in df.select(*columns).orderBy(*order).collect()
    ]


def explain(df: DataFrame) -> str:
    stream = io.StringIO()
    with redirect_stdout(stream):
        df.explain(mode="formatted")
    return stream.getvalue()


def read_small(spark: SparkSession, input_dir: str | None = None):
    folder = input_dir or (ROOT / "data" / "small").as_uri()
    raw = (spark.read.schema(EVENT_SCHEMA).option("header", True)
           .option("mode", "FAILFAST").csv(folder.rstrip("/") + "/events.csv"))
    courses = (spark.read.schema(COURSE_SCHEMA).option("header", True)
               .option("mode", "FAILFAST").csv(folder.rstrip("/") + "/courses.csv"))
    return raw, courses


def generate_large(spark: SparkSession, rows: int, hot_percent: int, partitions: int):
    """All rows are generated on executors, without a Python list of big data."""
    source = spark.range(0, rows, 1, numPartitions=partitions)
    index = F.col("id")
    course = F.when(index % 100 < hot_percent, F.lit(10)).otherwise(
        11 + index % 49)
    epoch = F.lit(1793145600) + index % (14 * 86400)  # 2026-10-28 UTC
    raw = source.select(
        (index + 1).alias("event_id"),
        F.when(index % 101 != 0, 1000 + index % 10000).cast("long").alias("user_id"),
        course.cast("long").alias("course_id"),
        F.element_at(F.array(F.lit("login"), F.lit("view"), F.lit("quiz")),
                     (index % 3 + 1).cast("int")).alias("event_type"),
        F.when(index % 211 == 0, F.lit("bad"))
         .otherwise(F.from_unixtime(epoch, "yyyy-MM-dd HH:mm:ss"))
         .alias("event_time"),
    )
    # Exact duplicates only: every 1000th source event is appended once.
    raw = raw.unionByName(raw.filter((F.col("event_id") - 1) % 1000 == 0))
    courses = spark.range(10, 60).select(
        F.col("id").alias("course_id"),
        F.concat(F.lit("Course "), F.col("id")).alias("course_name"))
    return raw, courses


def clean_events(raw: DataFrame):
    typed = raw.withColumn("ts", F.try_to_timestamp(
        "event_time", F.lit("yyyy-MM-dd HH:mm:ss")))
    valid_condition = (
        F.col("event_id").isNotNull() & F.col("user_id").isNotNull()
        & F.col("course_id").isNotNull() & F.col("ts").isNotNull())
    rejected = typed.filter(~valid_condition)
    valid = typed.filter(valid_condition)
    clean = valid.dropDuplicates(["event_id"]).withColumn("day", F.to_date("ts"))
    return clean, rejected, valid


def daily_metrics(clean: DataFrame) -> DataFrame:
    return clean.groupBy("course_id", "day").agg(
        F.count("*").alias("events"), F.countDistinct("user_id").alias("users"))


def sql_metrics(spark: SparkSession, clean: DataFrame) -> DataFrame:
    clean.createOrReplaceTempView("events")
    return spark.sql("""
        SELECT course_id, day, COUNT(*) AS events,
               COUNT(DISTINCT user_id) AS users
        FROM events GROUP BY course_id, day
    """)


def rank_users(clean: DataFrame) -> DataFrame:
    activity = clean.groupBy("course_id", "user_id").agg(F.count("*").alias("events"))
    window = Window.partitionBy("course_id").orderBy(F.desc("events"), "user_id")
    return activity.withColumn("place", F.row_number().over(window)).filter("place <= 3")


def trends(daily: DataFrame) -> DataFrame:
    window = Window.partitionBy("course_id").orderBy("day")
    frame = window.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    return (daily.withColumn("previous", F.lag("events").over(window))
            .withColumn("cumulative", F.sum("events").over(frame)))


def assert_same(left: DataFrame, right: DataFrame, columns: list[str]):
    a, b = left.select(*columns), right.select(*columns)
    # CSV nullability differs from DDL, so compare the actual two computations.
    if a.schema != b.schema:
        raise AssertionError(f"Different schemas: {a.schema} != {b.schema}")
    if a.exceptAll(b).limit(1).count() or b.exceptAll(a).limit(1).count():
        raise AssertionError("Results differ, including duplicate multiplicities")


def verify_golden(daily: DataFrame, top3: DataFrame, trend: DataFrame):
    specifications = [
        (daily, "daily.csv", DAILY_COLS, ["course_id", "day"]),
        (top3, "top3.csv", ["course_id", "user_id", "events", "place"],
         ["course_id", "place"]),
        (trend, "trend.csv", DAILY_COLS + ["previous", "cumulative"],
         ["course_id", "day"]),
    ]
    result = {}
    for df, filename, columns, order in specifications:
        actual = records(df, columns, order)
        with (ROOT / "expected" / filename).open(encoding="utf-8", newline="") as stream:
            wanted = list(csv.DictReader(stream))
        normalized = [
            {key: "" if value is None else str(value) for key, value in row.items()}
            for row in actual]
        if normalized != wanted:
            raise AssertionError(f"Golden mismatch for {filename}: {normalized} != {wanted}")
        result[filename] = actual
    return result


def ensure_new_output(spark: SparkSession, output: str):
    """Use Spark's configured Hadoop filesystem for local, HDFS and other URIs."""
    path = spark._jvm.org.apache.hadoop.fs.Path(output)
    fs = path.getFileSystem(spark._jsc.hadoopConfiguration())
    if fs.exists(path):
        raise FileExistsError(f"Output already exists; choose a new path: {output}")


def benchmark_join(spark: SparkSession, clean: DataFrame, courses: DataFrame,
                   output: str, repeats: int, report_dir: Path):
    """Controlled strategy comparison. Writes force the full JOIN result."""
    keys = ["spark.sql.adaptive.enabled", "spark.sql.autoBroadcastJoinThreshold"]
    old = {key: spark.conf.get(key) for key in keys}
    facts = clean.select("event_id", "user_id", "course_id", "ts").persist(
        StorageLevel.MEMORY_AND_DISK)
    dim = courses.persist(StorageLevel.MEMORY_AND_DISK)
    try:
        facts.count()
        dim.count()
        # Teaching isolation: both automatic broadcast and AQE are disabled.
        spark.conf.set(keys[0], "false")
        spark.conf.set(keys[1], "-1")
        queries = {
            "merge": facts.hint("merge").join(dim.hint("merge"), "course_id", "left"),
            "broadcast": facts.join(F.broadcast(dim), "course_id", "left"),
        }
        columns = ["course_id", "event_id", "user_id", "ts", "course_name"]
        assert_same(queries["merge"], queries["broadcast"], columns)
        for name, query in queries.items():
            (report_dir / f"benchmark_{name}_plan.txt").write_text(explain(query), encoding="utf-8")
            query.write.mode("errorifexists").parquet(f"{output}/benchmark/{name}/warmup")
        timings = {name: [] for name in queries}
        sequence = []
        for repeat in range(repeats):
            order = ["merge", "broadcast"] if repeat % 2 == 0 else ["broadcast", "merge"]
            for name in order:
                start = perf_counter()
                queries[name].write.mode("errorifexists").parquet(
                    f"{output}/benchmark/{name}/trial-{repeat + 1}")
                elapsed = perf_counter() - start
                timings[name].append(elapsed)
                sequence.append({"trial": repeat + 1, "strategy": name, "seconds": elapsed})
        return {
            "rows": facts.count(), "repeats": repeats, "timings": sequence,
            "median_seconds": {k: statistics.median(v) for k, v in timings.items()},
            "configuration": {key: spark.conf.get(key) for key in keys},
            "equal_results": True,
            "method": "Precached inputs, per-strategy warmup, alternating order, full Parquet writes",
            "interpretation": "No guaranteed speedup. Includes filesystem I/O. Local mode is one machine.",
        }
    finally:
        facts.unpersist()
        dim.unpersist()
        for key, value in old.items():
            spark.conf.set(key, value)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["small", "large"], default="small")
    parser.add_argument("--master", help="Use local[2] for local Python, omit under spark-submit")
    parser.add_argument("--input-dir", help="Small CSV folder, supports hdfs:// or file://")
    parser.add_argument("--rows", type=int, default=100_000, help="Base rows before duplicate injection")
    parser.add_argument("--hot-percent", type=int, default=80, help="Share assigned to course 10")
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--output", required=True, help="NEW data directory, local path or Hadoop URI")
    parser.add_argument("--report-dir", default=None, help="NEW local directory on the driver")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pause", action="store_true", help="Keep Spark UI alive until Enter")
    args = parser.parse_args()
    if args.rows < 1000 or args.partitions < 1 or args.repeats < 1:
        parser.error("rows >= 1000, partitions >= 1 and repeats >= 1 are required")
    if not 0 <= args.hot_percent <= 100:
        parser.error("hot-percent must be in 0..100")
    if args.input_dir and args.mode != "small":
        parser.error("input-dir applies to mode=small only")
    return args


def run(args):
    report_dir = Path(args.report_dir or (
        ROOT / "reports" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))).resolve()
    if report_dir.exists():
        raise FileExistsError(f"Report directory already exists: {report_dir}")
    builder = (SparkSession.builder.appName("Lecture8MoodleDemo")
               .config("spark.sql.session.timeZone", "UTC")
               .config("spark.sql.shuffle.partitions", str(args.partitions)))
    if args.master:
        builder = builder.master(args.master)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    spark.sparkContext.setJobDescription("Lecture 8: synthetic Moodle analytics")
    report_dir.mkdir(parents=True)
    try:
        ensure_new_output(spark, args.output)
        version = tuple(int(x) for x in spark.version.split(".")[:2])
        if version < (3, 5):
            raise RuntimeError("This demo requires Spark 3.5+ (try_to_timestamp)")
        print(f"Spark {spark.version}; master={spark.sparkContext.master}")
        print(f"Spark UI: {spark.sparkContext.uiWebUrl}")
        if args.mode == "small":
            raw, courses = read_small(spark, args.input_dir)
        else:
            raw, courses = generate_large(spark, args.rows, args.hot_percent, args.partitions)
        clean, rejected, valid = clean_events(raw)
        clean = clean.persist(StorageLevel.MEMORY_AND_DISK)
        raw_count, valid_count, clean_count = raw.count(), valid.count(), clean.count()
        audit = {"raw_rows": raw_count, "rejected_rows": raw_count - valid_count,
                 "duplicate_valid_rows": valid_count - clean_count, "clean_rows": clean_count}
        if courses.filter("course_id IS NULL").limit(1).count():
            raise AssertionError("NULL course key")
        if courses.groupBy("course_id").count().filter("count > 1").limit(1).count():
            raise AssertionError("Course dimension keys must be unique")
        daily = daily_metrics(clean)
        sql = sql_metrics(spark, clean)
        assert_same(daily, sql, DAILY_COLS)
        top3, trend = rank_users(clean), trends(daily)
        enriched = clean.join(courses, "course_id", "left")
        missing_count = enriched.filter("course_name IS NULL").count()
        audit["missing_course_rows"] = missing_count
        if missing_count:
            raise AssertionError("Events without course dimension entries")
        # Additional lecture operations: filter, select, when, withColumn, unionByName.
        views = (clean.filter(F.col("event_type") == "view")
                 .select("event_id", "user_id", "course_id", "ts")
                 .withColumn("hour", F.hour("ts"))
                 .withColumn("kind", F.when(F.col("hour") < 12, "morning").otherwise("later")))
        batch_a, batch_b = raw.filter("event_id <= 4"), raw.filter("event_id > 4")
        if batch_a.unionByName(batch_b).count() != raw_count:
            raise AssertionError("Batch union changed row count")
        golden = None
        if args.mode == "small":
            if audit != {"raw_rows": 10, "rejected_rows": 2, "duplicate_valid_rows": 1,
                         "clean_rows": 7, "missing_course_rows": 0}:
                raise AssertionError(f"Unexpected small data quality: {audit}")
            golden = verify_golden(daily, top3, trend)
        for name, df in [("daily", daily), ("top3", top3), ("trend", trend), ("views", views)]:
            print(f"\n=== {name} (bounded preview) ===")
            df.orderBy(*({"daily": ["course_id", "day"], "top3": ["course_id", "place"],
                         "trend": ["course_id", "day"], "views": ["event_id"]}[name])).show(12, False)
        for name, df in [("daily", daily), ("join", enriched), ("window", top3)]:
            (report_dir / f"{name}_plan.txt").write_text(explain(df), encoding="utf-8")
        for name, df in [("events", clean), ("daily", daily), ("trend", trend)]:
            df.write.mode("errorifexists").partitionBy("day").parquet(f"{args.output}/{name}")
        top3.write.mode("errorifexists").parquet(f"{args.output}/top3")
        rejected.write.mode("errorifexists").parquet(f"{args.output}/rejected")
        restored = spark.read.parquet(f"{args.output}/events")
        if restored.count() != clean_count:
            raise AssertionError("Parquet row count changed")
        pruning = (restored.filter(F.col("day") == F.lit("2026-10-28"))
                   .select("course_id", "user_id", "day"))
        (report_dir / "parquet_pruning_plan.txt").write_text(explain(pruning), encoding="utf-8")
        # Demonstrate cache reuse, then release it. The ETL uses the same cached clean.
        cache_stats = {"by_event_type": records(clean.groupBy("event_type").count(),
                                               ["event_type", "count"], ["event_type"])}
        benchmark = None
        if args.benchmark:
            benchmark = benchmark_join(spark, clean, courses, args.output, args.repeats, report_dir)
        summary = {
            "mode": args.mode, "spark_version": spark.version,
            "master": spark.sparkContext.master, "timezone": "UTC",
            "partitions": args.partitions, "hot_percent": args.hot_percent if args.mode == "large" else None,
            "output": args.output, "audit": audit, "golden_results": golden,
            "sql_equals_dataframe": True, "parquet_row_count_verified": True,
            "cache_summary": cache_stats, "benchmark": benchmark,
        }
        (report_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nQUALITY:", json.dumps(audit, ensure_ascii=False))
        print("CHECKS: SQL/DataFrame and Parquet passed")
        print("REPORT:", report_dir / "summary.json")
        print("DATA:", args.output)
        if args.pause:
            input("Spark UI is active. Press Enter to stop Spark... ")
        clean.unpersist()
        return summary
    finally:
        spark.stop()


if __name__ == "__main__":
    run(parse_args())
