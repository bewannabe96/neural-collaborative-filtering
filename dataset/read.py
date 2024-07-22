from typing import Optional

import pandas as pd
from pandas.errors import EmptyDataError


def read_multiple_csv_as_df(filenames: list[str], **kwargs) -> pd.DataFrame:
    df: Optional[pd.DataFrame] = None
    for filename in filenames:
        try:
            df_part = pd.read_csv(filename, **kwargs)
            df = df_part if df is None else pd.concat([df, df_part], ignore_index=True)
        except (EmptyDataError, ):
            continue
    return df
