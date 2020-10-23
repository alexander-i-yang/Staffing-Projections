import google.auth
from google.cloud import bigquery
from google.cloud import bigquery_storage
import pandas as pd
import datetime


def to_dataframe(rows):
    arr = []
    column_names = []
    for r in rows:
        if not column_names: column_names = list(r.keys())
        arr.insert(-1, list(r.values()))
    df = pd.DataFrame(arr)
    df.columns = column_names
    return df


def get_data_by_bu(rawdf, bu_id):
    return (rawdf.loc[rawdf.bu_id == bu_id])


bqclient = bigquery.Client.from_service_account_json("Staffing-Projections-425ff1698984.json")
bqstorageclient = bigquery_storage.BigQueryReadClient(
    credentials=google.auth.load_credentials_from_file(filename="Staffing-Projections-425ff1698984.json",
                                                       quota_project_id="staffing-projections")
)

# bqconfig = bigquery.QueryJobConfig(destination='staffing-projections.bucket_data.inbound_monthly_test_bucketized')

query_string = """
WITH
  inbound_monthly AS (SELECT * FROM staffing-projections.raw_data.inbound_monthly),
  program_ids AS (SELECT * FROM staffing-projections.raw_data.programs),
  report AS (SELECT program_id, bu_id, business_unit FROM staffing-projections.raw_data.report)
SELECT
  *
FROM(
  SELECT
    bu_id,
    MAX(business_unit) AS business_unit,
    SUM(TIMESTAMP_DIFF(time_terminated, time_answered, SECOND)) AS call_time,
    TIMESTAMP_SECONDS(60*60 * DIV(UNIX_SECONDS(time_answered), 60*60)) AS time_interval,
  FROM
    inbound_monthly
  FULL OUTER JOIN
    report
  USING(program_id)
  WHERE
    time_answered IS NOT NULL
  GROUP BY bu_id, time_interval
)
"""

rows = (
    bqclient.query(
        query_string,
    )
        .result()
)

rawdf = to_dataframe(rows).sort_index()

print(rawdf)

rawdf['bu_id'] = rawdf['bu_id'].fillna(0)
rawdf = rawdf.astype({'bu_id': 'int32'})
rawdf['time_interval'] = pd.to_datetime(rawdf['time_interval'])
bu_ids = rawdf['bu_id'].unique().tolist()
bu_units = rawdf['business_unit'].unique().tolist()
print(bu_ids)
print(bu_units)
rawdf.sort_values(by=["bu_id", "time_interval"], inplace=True)
rawdf.set_index(keys=["bu_id"], drop=False, inplace=True)

timestamps = rawdf['time_interval'].unique().to_numpy()
timestamps.sort()
start_timestamp = timestamps[0]
end_timestamp = timestamps[-1]
points_per_id = int((end_timestamp - start_timestamp).total_seconds() / (60 * 60))

print(rawdf)

finaldf = pd.DataFrame()
for counter in range(len(bu_ids)):
    # if counter > 3: break
    cur_id = bu_ids[counter]
    cur_bu_unit = bu_units[counter]
    if counter % 10 == 1:
        print("%i: %i/%i" % (cur_id, counter, len(bu_ids)))
    program_df = get_data_by_bu(rawdf, cur_id)

    rows_to_add = []
    for i in range(points_per_id + 1):
        cur_timestamp = start_timestamp + datetime.timedelta(0, i * 60 * 60)
        if cur_timestamp not in program_df.values:
            rows_to_add.append([cur_id, cur_bu_unit, 0, cur_timestamp])
    df_to_add = pd.DataFrame(rows_to_add, columns=program_df.columns)
    program_df = program_df.append(df_to_add, ignore_index=True)
    program_df.sort_values(by="time_interval", inplace=True, ignore_index=True)
    finaldf = finaldf.append(program_df)

f = open("inbound_monthly.csv", "w")
f.write(finaldf.to_csv())
f.close()
