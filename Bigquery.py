import google.auth
from google.cloud import bigquery
from google.cloud import bigquery_storage
import pandas as pd

def to_dataframe(rows):
    arr = []
    column_names = []
    for r in rows:
        if not column_names: column_names = list(r.keys())
        arr.insert(-1, list(r.values()))
    df = pd.DataFrame(arr)
    df.columns = column_names
    return df


bqclient = bigquery.Client.from_service_account_json("Staffing-Projections-425ff1698984.json")
bqstorageclient = bigquery_storage.BigQueryReadClient(
    credentials=google.auth.load_credentials_from_file(filename="Staffing-Projections-425ff1698984.json",
                                                       quota_project_id="staffing-projections")
)
query_string = """
SELECT
*
FROM `staffing-projections.test_sample_dataset.test_sample_training_table`
ORDER BY `hour`
"""

rows = (
    bqclient.query(query_string)
        .result()
)