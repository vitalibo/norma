import pytest
from pyspark.sql import functions as fn
from pyspark.sql.types import ArrayType, IntegerType, StringType, StructField, StructType
from pyxis import resources

from norma.engines.pyspark.utils import dtype_at, dtype_drop, dtype_set, nested_set_expr


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
def test_nested_set_expr(case, column, spark):
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

    root = column.split('.')[0].removesuffix('[]')
    exists = root in df.columns
    actual = df.withColumn(root, nested_set_expr(
        column, fn.lit('new_value'),
        fn.col(root) if exists else None,
        df.schema[root].dataType if exists else None
    ))

    spark.assert_dataframe_equals(
        actual, expected, schema=schema)


def test_dtype_at():
    dtype = StructType([
        StructField('f1', StringType()),
        StructField('f2', StructType([
            StructField('f3', ArrayType(StructType([
                StructField('f4', IntegerType())
            ])))
        ]))
    ])

    assert dtype_at(dtype, '') is dtype
    assert dtype_at(dtype, 'f1') == StringType()
    assert dtype_at(dtype, 'f2.f3[].f4') == IntegerType()
    assert dtype_at(dtype, 'f2.f3').elementType.names == ['f4']
    with pytest.raises(KeyError):
        dtype_at(dtype, 'f0')
    with pytest.raises(ValueError):
        dtype_at(dtype, 'f1.f0')


def test_dtype_set():
    dtype = StructType([
        StructField('f1', StringType()),
    ])

    assert dtype_set(dtype, 'f1', IntegerType()) == StructType([
        StructField('f1', IntegerType()),
    ])
    assert dtype_set(dtype, 'f2', IntegerType()) == StructType([
        StructField('f1', StringType()),
        StructField('f2', IntegerType()),
    ])
    assert dtype_set(dtype, 'f2.f3', IntegerType()) == StructType([
        StructField('f1', StringType()),
        StructField('f2', StructType([
            StructField('f3', IntegerType()),
        ])),
    ])
    assert dtype_set(dtype, 'f2[].f3', IntegerType()) == StructType([
        StructField('f1', StringType()),
        StructField('f2', ArrayType(StructType([
            StructField('f3', IntegerType()),
        ]))),
    ])
    assert dtype_set(None, '', IntegerType()) == IntegerType()


def test_dtype_drop():
    dtype = StructType([
        StructField('f1', StringType()),
        StructField('f2', StructType([
            StructField('f3', ArrayType(StructType([
                StructField('f4', IntegerType()),
                StructField('f5', IntegerType()),
            ])))
        ]))
    ])

    assert dtype_drop(dtype, 'f1') == StructType([
        StructField('f2', StructType([
            StructField('f3', ArrayType(StructType([
                StructField('f4', IntegerType()),
                StructField('f5', IntegerType()),
            ])))
        ]))
    ])
    assert dtype_drop(dtype, 'f2.f3') == StructType([
        StructField('f1', StringType()),
        StructField('f2', StructType([]))
    ])
    assert dtype_drop(dtype, 'f2.f3[].f5') == StructType([
        StructField('f1', StringType()),
        StructField('f2', StructType([
            StructField('f3', ArrayType(StructType([
                StructField('f4', IntegerType()),
            ])))
        ]))
    ])
