"""Integration contracts against lecture's fixed expected data."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from moodle_demo import (SparkSession, F, read_small, clean_events, daily_metrics,
                         sql_metrics, rank_users, trends, verify_golden,
                         assert_same, generate_large)


class LectureDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (SparkSession.builder.master("local[2]").appName("Lecture8Tests")
                     .config("spark.sql.shuffle.partitions", "2")
                     .config("spark.sql.session.timeZone", "UTC").getOrCreate())
        cls.spark.sparkContext.setLogLevel("ERROR")
        cls.raw, cls.courses = read_small(cls.spark)
        cls.clean, cls.rejected, cls.valid = clean_events(cls.raw)
        cls.clean = cls.clean.cache()

    @classmethod
    def tearDownClass(cls):
        cls.clean.unpersist()
        cls.spark.stop()

    def test_quality_contract(self):
        self.assertEqual(self.raw.count(), 10)
        self.assertEqual(self.rejected.count(), 2)
        self.assertEqual(self.valid.count(), 8)
        self.assertEqual(self.clean.count(), 7)

    def test_lecture_golden_results(self):
        daily = daily_metrics(self.clean)
        result = verify_golden(daily, rank_users(self.clean), trends(daily))
        self.assertEqual(len(result["daily.csv"]), 4)

    def test_sql_and_dataframe_equivalence(self):
        columns = ["course_id", "day", "events", "users"]
        assert_same(daily_metrics(self.clean), sql_metrics(self.spark, self.clean), columns)

    def test_null_aggregates(self):
        frame = self.spark.createDataFrame([(101,), (101,), (None,)], "user_id LONG")
        values = frame.agg(F.count("*").alias("rows"), F.count("user_id").alias("known"),
                           F.countDistinct("user_id").alias("unique")).first()
        self.assertEqual(tuple(values), (3, 2, 1))

    def test_join_duplicate_dimension_is_visible(self):
        extra = self.spark.createDataFrame([(10, "Duplicate")], "course_id LONG, course_name STRING")
        bad = self.courses.unionByName(extra)
        self.assertEqual(self.clean.join(bad, "course_id", "left").count(), 11)

    def test_large_generator_repeatable_and_distinct_key(self):
        raw, courses = generate_large(self.spark, 1000, 80, 2)
        clean, _, _ = clean_events(raw)
        self.assertEqual(raw.count(), 1001)
        # 10 missing users, 5 invalid times, one overlap: 986 valid unique events.
        self.assertEqual(clean.count(), 986)
        self.assertEqual(courses.count(), 50)
        self.assertEqual(clean.select("event_id").distinct().count(), 986)
        self.assertEqual(str(clean.agg(F.min("day")).first()[0]), "2026-10-28")


if __name__ == "__main__":
    unittest.main()
