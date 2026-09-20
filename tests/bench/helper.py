import argparse
import datetime
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess  # noqa: S404
import sys

VERBOSE = False


def debug(message):
    if not VERBOSE:
        return

    info(message)


def info(message):
    stamp = datetime.datetime.now().astimezone().strftime('%H:%M:%S.%f')[:-3]
    print(f'[{stamp}] {message}', file=sys.stderr, flush=True)


def parse_args(desc):
    def values(literal):
        def value(o):
            try:
                return json.loads(o)
            except json.JSONDecodeError:
                return o

        return [value(item) for item in literal.split(',')]

    options = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    options.add_argument('file')
    options.add_argument('-b', '--benchmarks', type=values, default='*', metavar='pattern[,pattern]')
    options.add_argument('-i', '--iterations', type=int, default=3)
    options.add_argument('-w', '--warmup', type=int, default=1)
    options.add_argument('-v', '--verbose', action='store_true')
    parsed_args = options.parse_args()

    global VERBOSE  # noqa: PLW0603
    VERBOSE = parsed_args.verbose
    return parsed_args


def plan_nodes(plan, codegen=False):
    name = plan.nodeName()
    if name.startswith('WholeStageCodegen') or name == 'InputAdapter':
        total, fused, codegen = 0, 0, name != 'InputAdapter'
    elif plan.children().isEmpty():  # leaves only read the input
        total, fused = 0, 0
    else:
        total, fused = 1, int(codegen)

    if name != 'InMemoryTableScan':
        children = plan.children().iterator()
        while children.hasNext():
            child_total, child_fused = plan_nodes(children.next(), codegen)
            total, fused = total + child_total, fused + child_fused
    return total, fused


def missed_subexpressions(plan, spark):  # noqa: C901
    def iterate(seq):
        it = seq.iterator()
        while it.hasNext():
            yield it.next()

    def cheapness():
        # the notion CollapseProject relies on when it inlines an expression into several places
        def load(*names):
            return [spark._jvm.java.lang.Class.forName(f'org.apache.spark.sql.catalyst.expressions.{o}') for o in names]

        leaves, wrappers = load('Attribute', 'OuterReference', 'PythonUDF'), load('Alias', 'ExtractValue')

        def cheap(expr):
            if expr.foldable() or any(cls.isInstance(expr) for cls in leaves):
                return True
            return any(cls.isInstance(expr) for cls in wrappers) and all(
                cheap(child) for child in iterate(expr.children()))

        return cheap

    shortcut = spark.conf.get('spark.sql.subexpressionElimination.skipForShortcutExpr') == 'true'
    cheap = cheapness()

    def projects(node):
        if node.nodeName() == 'Project':
            yield node
        for child in iterate(node.children()):
            yield from projects(child)

    def eliminated(exprs):
        # the same pass ProjectExec codegen runs before emitting the projection
        equivalence = spark._jvm.org.apache.spark.sql.catalyst.expressions.EquivalentExpressions(shortcut)
        target = getattr(equivalence, 'addExprTree$default$2')()
        for expr in exprs:
            equivalence.addExprTree(expr, target)
        return {
            state.expr().canonicalized().toString(): state.useCount()
            for state in iterate(equivalence.getAllExprStates(1))
        }

    def repeated(exprs):
        # the same walk without its blind spots: CodegenFallback children and lazily evaluated branches
        counts, samples = {}, {}

        def visit(expr):
            if expr.nodeName() == 'LambdaFunction' or not expr.deterministic():
                return
            if not expr.children().isEmpty():
                key = expr.canonicalized().toString()
                counts[key] = counts.get(key, 0) + 1
                if counts[key] > 1:
                    return
                samples[key] = expr
            for child in iterate(expr.children()):
                visit(child)

        for expr in exprs:
            visit(expr)
        return {key: (count, samples[key]) for key, count in counts.items() if count > 1}

    missed = 0
    for project in projects(plan):
        exprs = list(iterate(project.projectList()))
        done = eliminated(exprs)
        missed += sum(
            max(count - done.get(key, 1), 0)
            for key, (count, expr) in repeated(exprs).items()
            if not cheap(expr)
        )
    return missed


def environment(runs):
    def command(*args, merge=False):
        try:
            done = subprocess.run(  # noqa: S603
                args, check=False, capture_output=True, text=True, timeout=5,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if done.returncode != 0:
            return None
        return (done.stdout + done.stderr).strip() if merge else done.stdout.strip()

    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return '-'

    def cpu():
        if sys.platform == 'darwin':
            return command('sysctl', '-n', 'machdep.cpu.brand_string') or platform.machine()
        try:
            with open('/proc/cpuinfo', encoding='utf-8') as f:
                model = next((line for line in f if line.startswith('model name')), None)
        except OSError:
            model = None
        return model.split(':', 1)[1].strip() if model else platform.processor() or platform.machine()

    def memory():
        if sys.platform == 'darwin':
            size = command('sysctl', '-n', 'hw.memsize')
            return f'{int(size) / 2 ** 30:.0f} GiB' if size else '-'
        try:
            with open('/proc/meminfo', encoding='utf-8') as f:
                total = next((line for line in f if line.startswith('MemTotal')), None)
        except OSError:
            total = None
        return f'{int(total.split()[1]) / 2 ** 20:.0f} GiB' if total else '-'

    def jvm():
        home = os.environ.get('JAVA_HOME')
        reported = command(os.path.join(home, 'bin', 'java') if home else 'java', '-version', merge=True)
        return reported.split('\n')[0] if reported else '-'

    commit = command('git', 'rev-parse', '--short', 'HEAD') or '-'
    if command('git', 'status', '--porcelain', '--untracked-files=no'):
        commit = f'{commit} (dirty)'

    return {
        'date': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        'commit': commit,
        'cpu': f'{cpu()} ({os.cpu_count()} cores)',
        'os': platform.platform(),
        'memory': memory(),
        'python': platform.python_version(),
        'jvm': jvm(),
        'pyspark': version('pyspark'),
        'iterations': runs[0].iterations,
        'warmup': runs[0].warmup
    }


def render(runs):
    def stats(results, name, f='3'):
        values = [result[name] for result in results if result.get(name) is not None]
        if not values:
            return ['-'] * 2
        return [
            f'{min(values):.{f}f}',
            f'{statistics.stdev(values):.{f}f}' if len(values) > 1 else '-',
        ]

    facts = environment(runs)
    width = max(len(key) for key in facts)
    print('-- env --')
    for key, value in facts.items():
        print(f'{key.ljust(width)}  {value}')

    header = [
        'benchmark', 'parameters', 'build.min', 'build.sd', 'run.min', 'run.sd',
        'plan_chars', 'plan_nodes', 'physical_nodes', 'codegen_nodes', 'cse_missed',
    ]

    rows = [
        [
            run.benchmark,
            run.spec,
            *stats(run.results, 'build'),
            *stats(run.results, 'run'),
            *[str(run.results[-1][name]) for name in ('plan', 'plan_nodes', 'nodes', 'codegen', 'cse_missed')],
        ]
        for run in runs
    ]

    widths = [max(len(cell) for cell in column) for column in zip(header, *rows)]
    print('\n-- benchmarks --')
    for cells in (header, *rows):
        print('  '.join(
            cell.ljust(width) if i < 2 else cell.rjust(width)
            for i, (cell, width) in enumerate(zip(cells, widths))
        ).rstrip())

    planned = [run for run in runs if run.plans]
    if planned:
        print('\n-- plans --')
    for index, run in enumerate(planned):
        print('\n' * (3 if index else 1) + f'=== {run.benchmark} :: {run.spec} ===')
        for kind, plan in run.plans.items():
            print(f'\n--- {kind} ---\n{plan}')
