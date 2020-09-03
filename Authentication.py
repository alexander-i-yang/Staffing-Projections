from google.oauth2 import service_account
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1 as bigquery_storage

key_path = "Staffing-Projections-425ff1698984.json"

credentials = service_account.Credentials.from_service_account_file(
    key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

client = bigquery.Client(credentials=credentials, project=credentials.project_id,)
storage_client = bigquery_storage.BigQueryReadClient(credentials=credentials)

# import google.auth
# from google.cloud import bigquery
# from google.cloud import bigquery_storage
#
# # Explicitly create a credentials object. This allows you to use the same
# # credentials for both the BigQuery and BigQuery Storage clients, avoiding
# # unnecessary API calls to fetch duplicate authentication tokens.
# credentials, your_project_id = google.auth.default()
# # Make clients.
# bqclient = bigquery.Client(
#     credentials=credentials,
#     project=your_project_id,
# )
# bqstorage_client = bigquery_storage.BigQueryReadClient(credentials=google.auth.load_credentials_from_file("Staffing-Projections-425ff1698984.json"))
#
# Download query results.
query_string = """
SELECT
CONCAT(
    'https://stackoverflow.com/questions/',
    CAST(id as STRING)) as url,
view_count
FROM `bigquery-public-data.stackoverflow.posts_questions`
WHERE tags like '%google-bigquery%'
ORDER BY view_count DESC
LIMIT 10
"""
dataframe = (
    client.query(query_string)
    .result()
    .to_dataframe(bqstorage_client=storage_client)
)
print("done")
for row in dataframe:
    print(row)
