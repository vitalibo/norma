import fnmatch
import functools
import json
import re
from argparse import Namespace

from pyspark.sql import DataFrame
from pyspark.sql import functions as fn
from pyspark.sql.types import ArrayType, MapType, StringType, StructType
from pyxis import pyspark as pyxis

FOR = re.compile(r'^!for\((?P<var>[A-Za-z_]\w*) in (?P<bounds>.+)\)$')
ITER = re.compile(r'^!iter\((?P<var>[A-Za-z_]\w*) in (?P<bounds>.+)\)$')
SETTINGS = {'rows': 0, 'schema': False, 'data': 0, 'analyzed_plan': False, 'physical_plan': False}
EXPRESSION = re.compile(r'{(?P<expression>[^{}]+)}')


def render(value, scope):
    if not isinstance(value, str):
        return value
    return EXPRESSION.sub(lambda match: str(evaluate(match.group('expression'), scope)), value)


@functools.cache
def compiled(expression):
    return compile(expression, '<dsl>', 'eval')


def evaluate(expression, scope):
    try:
        return eval(compiled(expression), {}, scope)  # noqa: S307
    except NameError as e:
        name = str(e).split("'")[1]
        raise ValueError(f'unknown parameter {name!r} in {{{expression}}}') from e
    except Exception as e:
        raise ValueError(f'cannot evaluate {{{expression}}}: {e}') from e


def bounds(spec, scope):
    interval, _, step = spec.partition(':')
    start, separator, stop = interval.partition('..')
    if not separator:
        start, stop = '0', start

    limits = [render(value, scope) for value in (start, stop, step or '1')]
    try:
        return range(*(int(limit) for limit in limits))
    except ValueError as e:
        raise ValueError(f'invalid bounds in !for(... in {spec})') from e


def unroll(node, scope):
    obj, array = {}, []
    for key, value in node.items():
        directive = FOR.fullmatch(key)
        if not directive:
            obj[render(key, scope)] = expand(value, scope)
            continue

        for index in bounds(directive.group('bounds'), scope):
            child = expand(value, {**scope, directive.group('var'): index})
            if isinstance(child, dict):
                obj.update(child)
            elif isinstance(child, list):
                array.extend(child)
            else:
                array.append(child)

    if obj and array:
        raise ValueError(f'{node} expands into both an object and an array')
    return array or obj


def splice(value, scope):
    expanded = expand(value, scope)
    if isinstance(expanded, list) and not isinstance(value, list):
        return expanded
    return [expanded]


def expand(node, scope):
    if isinstance(node, str):
        return render(node, scope)
    if isinstance(node, list):
        return [item for value in node for item in splice(value, scope)]
    if not isinstance(node, dict):
        return node
    if set(node) == {'!json'}:
        return json.dumps(expand(node['!json'], scope))
    if set(node) == {'!expr'}:
        return evaluate(node['!expr'], scope)
    return unroll(node, scope)


class Run(Namespace):
    def materialize_state(self, spark, rows=None):
        state = self.state if rows is None else expand(self.template, {**self.params, 'rows': rows})
        return generate(spark, state['data'], pyxis.StructType.from_json(state['schema']))


def runs(options):
    with open(options.file, encoding='utf-8') as f:
        benchmarks = json.loads(f.read())

    missing = [
        pattern for pattern in options.benchmarks
        if not any(fnmatch.fnmatchcase(benchmark['benchmark'], pattern) for benchmark in benchmarks)
    ]
    if missing:
        raise ValueError(f'benchmark not found: {", ".join(missing)}')

    def selected(name):
        return any(fnmatch.fnmatchcase(name, pattern) for pattern in options.benchmarks)

    def parameters(spec):
        params = {}
        for item in spec.split(','):
            name, separator, literal = (part.strip() for part in item.partition('='))
            if not separator:
                raise ValueError(f'expected name=value in run {spec!r}, got {item!r}')
            try:
                params[name] = json.loads(literal)
            except json.JSONDecodeError:
                params[name] = literal

        settings = {name: params.pop(name, default) for name, default in SETTINGS.items()}
        return {**params, 'rows': settings['rows']}, settings

    result = []
    for benchmark in benchmarks:
        if not selected(benchmark['benchmark']):
            continue

        for spec in benchmark.pop('runs'):
            params, settings = parameters(spec)
            result.append(
                Run(**expand(benchmark, params), template=benchmark['state'], params=params, spec=spec,
                    warmup=options.warmup, iterations=options.iterations, **settings)
            )

    return result


def fields(node, dtype):
    if not isinstance(node, dict) or set(node) != set(dtype.fieldNames()):
        raise ValueError(f'data {node!r} does not match the fields of {dtype.simpleString()}')
    return [column(node[field.name], field.dataType).alias(field.name) for field in dtype.fields]


def scalar(node):
    return fn.expr(node) if isinstance(node, str) else fn.lit(node)


def json_value(node):
    # without a schema for the document, its values are strings like every other input leaf
    if isinstance(node, dict):
        return fn.struct(*(json_value(value).alias(name) for name, value in node.items()))
    if isinstance(node, list):
        return fn.array(*map(json_value, node))
    return scalar(node).cast(StringType())


def column(node, dtype):
    if isinstance(dtype, StructType):
        return fn.struct(*fields(node, dtype))
    if isinstance(dtype, ArrayType):
        return fn.array(*(column(item, dtype.elementType) for item in node)).cast(dtype)
    if isinstance(dtype, MapType):
        return fn.map_from_arrays(
            fn.array(*map(fn.lit, node)).cast(ArrayType(dtype.keyType)),
            fn.array(*(column(value, dtype.valueType) for value in node.values())).cast(ArrayType(dtype.valueType)),
        )
    if isinstance(node, (dict, list)) and isinstance(dtype, StringType):
        return fn.to_json(json_value(node))
    return scalar(node).cast(dtype)


def generate(spark, data, schema):
    (key, templates), = data.items()
    directive = ITER.fullmatch(key)
    if not directive or len(templates) != 1:
        raise ValueError('state.data must look like {"!iter(row in 0..{rows})": [<row template>]}')

    span = bounds(directive.group('bounds'), {})
    rows = spark.range(span.start, span.stop, span.step).select(fn.col('id').alias(directive.group('var')))
    generated = rows.select(*fields(templates[0], schema))

    jspark = spark._jsparkSession  # noqa: SLF001
    jdf = jspark.createDataFrame(generated._jdf.rdd(), jspark.parseDataType(schema.json()))  # noqa: SLF001
    return DataFrame(jdf, spark)
