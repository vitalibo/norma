import pytest
from pyspark.sql import SparkSession
from pyxis.pyspark import LocalTestSpark


@pytest.fixture(scope='module', name='spark')
def spark_fixture():
    session = SparkSession.builder \
        .appName('PyTest') \
        .config('spark.sql.session.timeZone', 'UTC') \
        .config('spark.sql.jsonGenerator.ignoreNullFields', False) \
        .getOrCreate()

    with LocalTestSpark(session) as spark:
        yield spark


@pytest.fixture(scope='module', name='spark_session')
def spark_session(spark):
    yield spark.spark_session
