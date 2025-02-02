import pytest

pytest.register_assert_rewrite('pyxis.pyspark')  # noqa isort:skip

# pylint: disable=wrong-import-position
from pyspark.sql import SparkSession
from pyxis.pyspark import LocalTestSpark


@pytest.fixture(scope='module', name='spark')
def spark_fixture():
    session = SparkSession.builder \
        .appName('PyTest') \
        .config('spark.sql.session.timeZone', 'UTC') \
        .getOrCreate()

    with LocalTestSpark(session) as spark:
        yield spark
