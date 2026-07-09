import pytest

pytest.register_assert_rewrite('pyxis.pyspark')

from pyspark.sql import SparkSession  # noqa: E402
from pyxis.pyspark import LocalTestSpark  # noqa: E402


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
    return spark.spark_session
