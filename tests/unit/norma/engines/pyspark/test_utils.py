import pytest
from pyspark.sql import functions as fn
from pyspark.sql.types import StructType
from pyxis import resources

from norma.engines.pyspark.utils import with_nested_column


@pytest.mark.parametrize('case, column', [
    pytest.param(f'case{i}', column, id=f'Case #{i}: {name}') for i, (name, column) in enumerate([
        ('update field', 'f1'),
        ('new field', 'f2'),
        ('new nested field', 'f2.f3.f4'),
        ('update nested field', 'f2.f3.f4'),
        ('update array', 'f1[]'),
        ('new array', 'f1[]'),
        ('update nested field in array', 'f1.f2[].f3.f4'),
        ('new nested field in array ', 'f1.f2[].f3.f4'),
        ('crate an empty array', 'f1[].f3')
    ])
])
def test_with_nested_column(case, column, spark):
    df = spark.create_dataframe_from_resource(
        __file__,
        f'data/with_nested_column/{case}/in.json',
        f'data/with_nested_column/{case}/in_schema.json'
    )
    expected = spark.create_dataframe_from_resource(
        __file__,
        f'data/with_nested_column/{case}/exp.json',
        f'data/with_nested_column/{case}/exp_schema.json'
    )
    schema = StructType.from_json(  # pylint:disable=no-member
        resources.resource_as_json(
            __file__,
            f'data/with_nested_column/{case}/exp_schema.json'
        )
    )

    actual = df.transform(
        with_nested_column(column, fn.lit('new_value')))

    spark.assert_dataframe_equals(
        actual, expected, schema=schema)
