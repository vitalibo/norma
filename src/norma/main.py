import json

import numpy as np
import pandas as pd

import norma.schema as nm
from norma import rules

schema = nm.Schema({
    'id': nm.Column(int, rules=[
        rules.required(),
        rules.int_parsing(),
        rules.MaskRule(lambda x: x['id'] % 2 == 0, error_type='even_id', error_msg='id must be even')
    ]),
    'age': nm.Column(int, rules=[
        rules.required(),
        rules.int_parsing(),
        rules.greater_than(29),
        rules.less_than(31)
    ]),
})

df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'email': ['alice@example.com', 'bob@example.com', 'invalid-email', 'neiman_@outreach.cjk', 'brodi@mouth.qfw'],
    'age': [25, np.nan, "N/A", 30, 35]
})

validated_df = schema.validate(df)

print(validated_df.dtypes)
dump = json.dumps(validated_df.to_dict(orient='records'), indent=2, default=str)
print(dump)
with open('actual.json', 'w') as f:
    f.write(dump)
