#!/usr/bin/env python
import functools
import re
import time

import dsl
import helper
from pyspark.sql import SparkSession

from norma import Schema


@functools.cache
def session():
    spark = SparkSession.builder \
        .master('local[*]') \
        .appName('Norma Benchmarks') \
        .config('spark.ui.enabled', 'false') \
        .config('spark.ui.showConsoleProgress', 'false') \
        .config('spark.sql.debug.maxToStringFields', str(2 ** 31 - 1)) \
        .config('spark.sql.adaptive.enabled', 'false') \
        .config('spark.driver.memory', '4g') \
        .getOrCreate()

    spark.sparkContext.setLogLevel('ERROR')
    return spark


def create_dataframe(run):
    if run.data:
        helper.debug(f'  sampling {run.data} rows of the input')
        run.sample = run.materialize_state(session(), rows=run.data)._jdf.toJSON().takeAsList(run.data)  # noqa: SLF001

    helper.debug(f'  building pyspark dataframe of {run.rows} rows')
    df = run.materialize_state(session())
    helper.debug('  dataframe ready')
    return df.cache()


def measure(run, df):
    helper.debug('  building validation plan')
    started = time.perf_counter()
    schema = Schema.from_json_schema(run.json_schema)
    validated = schema.validate(df)
    build_time = time.perf_counter() - started

    run_time = None
    if run.rows > 0:
        helper.debug(f'  plan built in {build_time:.3f}s, writing to the noop sink')
        started = time.perf_counter()
        validated.write.format('noop').mode('overwrite').save()
        run_time = time.perf_counter() - started

    helper.debug('  rendering the analyzed plan')
    query = validated._jdf.queryExecution()  # noqa: SLF001
    analyzed, executed = query.analyzed(), query.executedPlan()
    plans = {  # exprIds and lambda names drift within a session
        kind: re.sub(r'(?:_\d+)?#\d+', '', plan.toString()).rstrip()
        for kind, plan in (('analyzed', analyzed), ('physical', executed))
    }
    run.plans = {}
    if run.schema:
        run.plans['schema'] = df._jdf.schema().treeString().rstrip()  # noqa: SLF001
    if run.data:
        run.plans['data'] = '\n'.join(run.sample)
    run.plans.update({kind: plan for kind, plan in plans.items() if getattr(run, f'{kind}_plan')})
    helper.debug(f'  analyzed plan is {len(plans["analyzed"])} chars, counting physical operators')
    analyzed_nodes, _ = helper.plan_nodes(analyzed)
    executed_nodes, executed_codegen = helper.plan_nodes(executed)
    helper.debug(f'  {executed_codegen} of {executed_nodes} operators in whole-stage codegen')
    if not hasattr(run, 'cse_missed'):  # deterministic per run but slow on wide plans, so count once
        run.cse_missed = helper.missed_subexpressions(executed, session())

    return {
        'build': build_time, 'run': run_time, 'plan': len(plans['analyzed']),
        'plan_nodes': analyzed_nodes, 'nodes': executed_nodes, 'codegen': executed_codegen,
        'cse_missed': run.cse_missed,
    }


def main():
    runs = dsl.runs(helper.parse_args(__doc__))

    helper.info(f'Running {len(runs)} benchmarks...')
    for index, run in enumerate(runs, start=1):
        helper.info(f'[{index}/{len(runs)}] {run.benchmark} {run.spec}')

        df = create_dataframe(run)
        try:
            helper.debug('  materializing cache')
            df.count()

            for iteration in range(run.warmup):
                helper.debug(f' warmup {iteration + 1}/{run.warmup}')
                measure(run, df)

            run.results = []
            for iteration in range(run.iterations):
                helper.debug(f' iteration {iteration + 1}/{run.iterations}')
                run.results.append(measure(run, df))
        finally:
            helper.debug('  unpersisting')
            df.unpersist(blocking=True)

    helper.render(runs)


if __name__ == '__main__':
    main()
