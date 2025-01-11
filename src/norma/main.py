import json

import numpy as np
import pandas as pd

from norma import rules
from norma.engine import Validator

data = {
    'id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'email': ['alice@example.com', 'bob@example.com', 'invalid-email', 'foo@fo.com', 'as@bar.com'],
    'age': [25, np.nan, "N/A", 30, 35]
}

df = pd.DataFrame(data)

validator = Validator({
    # 'id': [
    #     rules.verify_number('id', gt=1, nullable=False),
    # ],
    'age': [
        rules.verify_number('age', nullable=False, ge=29, le=31)
    ],
})

df['errors'] = validator.validate(df)

dump = json.dumps(df.to_dict(orient='records'), indent=2, default=str)
print(dump)
with open('actual.json', 'w') as f:
    f.write(dump)
