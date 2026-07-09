import pytest
from pyspark.sql import Row
from pyspark.sql import functions as fn
from pyspark.sql.types import ArrayType, IntegerType, StringType, StructField, StructType
from pyxis import resources

from norma.engines.pyspark.utils import (
    dtype_at, dtype_drop, dtype_set, nested_drop_expr, nested_get_expr, nested_set_expr
)


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
        f'data/nested_set_expr/{case}/in.json',
        f'data/nested_set_expr/{case}/in_schema.json'
    )
    expected = spark.create_dataframe_from_resource(
        __file__,
        f'data/nested_set_expr/{case}/exp.json',
        f'data/nested_set_expr/{case}/exp_schema.json'
    )
    schema = StructType.from_json(  # pylint:disable=no-member
        resources.load_json(
            __file__,
            f'data/nested_set_expr/{case}/exp_schema.json'
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


@pytest.mark.parametrize('column, root_expr, expected', [
    pytest.param(column, args[0] if args else lambda: None, expected,
                 id=f'Case #{i}: {name}') for i, (name, column, expected, *args) in enumerate([
        ('top-level', 'f1', ['v1', 'v2']),
        ('nested field', 'f2.f3.f4', ['a', None]),
        ('array column itself', 'f5[]', [[Row(f6='x1', f7=Row(f8='y1')), Row(f6='x2', f7=Row(f8='y2'))], None]),
        ('field of array elements', 'f5[].f6', [['x1', 'x2'], None]),
        ('deeper field of array elements', 'f5[].f7.f8', [['y1', 'y2'], None]),
        ('use root_expr instead of fn.col of the root', 'f2.f3.f4', ['A', None], lambda: fn.col('f2').withField(
            'f3', fn.col('f2').getField('f3').withField('f4', fn.upper(fn.col('f2').getField('f3').getField('f4'))))),
    ])
])
def test_nested_get_expr(column, root_expr, expected, spark):
    df = spark.create_dataframe_from_resource(
        __file__,
        'data/nested_get_expr/in.json',
        'data/nested_get_expr/in_schema.json'
    )

    rows = df.select(nested_get_expr(column, root_expr()).alias('out')).collect()
    assert [row['out'] for row in rows] == expected


@pytest.mark.parametrize('case, column', [
    pytest.param(f'case{i}', column, id=f'Case #{i}: {name}') for i, (name, column) in enumerate([
        ('drop nested field', 'f1.f2'),
        ('drop deeply nested field', 'f1.f2.f3'),
        ('drop field of array element structs', 'f1.f2[].f4'),
        ('drop nested field inside array element structs', 'f1.f2[].f3.f5'),
        ('drop field with array root', 'f1[].f2'),
    ])
])
def test_nested_drop_expr(case, column, spark):
    df = spark.create_dataframe_from_resource(
        __file__,
        f'data/nested_drop_expr/{case}/in.json',
        f'data/nested_drop_expr/{case}/in_schema.json'
    )
    expected = spark.create_dataframe_from_resource(
        __file__,
        f'data/nested_drop_expr/{case}/exp.json',
        f'data/nested_drop_expr/{case}/exp_schema.json'
    )
    schema = StructType.from_json(  # pylint:disable=no-member
        resources.load_json(
            __file__,
            f'data/nested_drop_expr/{case}/exp_schema.json'
        )
    )

    root = column.split('.')[0].removesuffix('[]')
    actual = df.withColumn(root, nested_drop_expr(
        column, fn.col(root), df.schema[root].dataType))

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
