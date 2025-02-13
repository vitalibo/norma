import inspect

from norma import rules
from norma.engines.pandas import rules as pandas_rules
from norma.engines.pyspark import rules as pyspark_rules
from norma.rules import Rule


def _generate_test(method, argv):
    def test_rule():
        rule = method(**argv)

        assert rule is not None
        assert rule.name == method.__name__
        assert rule.priority >= 0
        assert rule.priority <= 10
        assert rule.kwargs == argv

    def test_rule_pandas():
        rule_func = getattr(pandas_rules, method.__name__)

        sig = inspect.signature(rule_func)

        assert sig.return_annotation == Rule
        assert set(sig.parameters.keys()) == set(argv.keys())

    def test_rule_pyspark():
        rule_func = getattr(pyspark_rules, method.__name__)

        sig = inspect.signature(rule_func)

        assert sig.return_annotation == Rule
        assert set(sig.parameters.keys()) == set(argv.keys())

    globals()[f'test_{name}'] = test_rule
    globals()[f'test_pandas_{name}'] = test_rule_pandas
    globals()[f'test_pyspark_{name}'] = test_rule_pyspark


for name, func in inspect.getmembers(rules, inspect.isfunction):
    signature = inspect.signature(func)
    if signature.return_annotation is Rule:
        _generate_test(func, dict(zip(signature.parameters.keys(), 'qwertyuiop')))
