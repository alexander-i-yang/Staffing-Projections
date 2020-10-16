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

# bqconfig = bigquery.QueryJobConfig(destination='staffing-projections.bucket_data.inbound_monthly_test_bucketized')

query_string = """
#standardSQL
WITH
  inbound_monthly AS (SELECT * FROM staffing-projections.raw_data.inbound_monthly),
  program_ids AS (SELECT * FROM staffing-projections.raw_data.programs)
SELECT
  time_interval,
  inbound_monthly.program_id,
  program_ids.p_name,
  call_time
FROM(
  SELECT
    program_id,
    SUM(TIMESTAMP_DIFF(time_terminated, time_answered, SECOND)) AS call_time,
    TIMESTAMP_SECONDS(30*60 * DIV(UNIX_SECONDS(time_answered), 30*60)) AS time_interval,
  FROM
    inbound_monthly
  WHERE
    time_answered IS NOT NULL
  GROUP BY program_id, time_interval
  ORDER BY time_interval) AS inbound_monthly
LEFT JOIN
  program_ids
ON
  program_ids.program_id = inbound_monthly.program_id
ORDER BY time_interval
"""

rows = (
    bqclient.query(
        query_string,
    )
        .result()
)

processed_data = to_dataframe(rows).sort_index()
f = open("inbound_monthly.csv", "w")
f.write(processed_data.to_csv())
f.close()
# cut_data = processed_data["call_time"]
# num_test = 100
# train = cut_data.head(600 - num_test)
# test = cut_data.tail(num_test)

# history = [x for x in train]
# predictions = list()
# extra = 50
# for t in range(len(test)+1):
#     model = ARIMA(history, order=(6, 1, 0))
#     model_fit = model.fit()
#     output = model_fit.forecast(steps=1)
#     test_index = t+600-num_test
#     yhat = output[0]
#     if test_index < 600:
#         obs = test[test_index]
#         history.append(obs)
#         predictions.append(yhat)
#     else:
#         yhat = model_fit.forecast(steps=extra)
#         predictions.extend(yhat)
#     print('predicted=%f, expected=%f' % (yhat, obs))
# error = mean_squared_error(test, predictions)
# print('Test MSE: %.3f' % error)
# print(len(predictions))
# predictions = pd.DataFrame(predictions)
# predictions['interval'] = range(600-num_test, 600+extra)
# predictions = predictions.set_index('interval')
# print(predictions)

# min = 10000000
# min_thing = ()
# for p in range(0, 14):
#     for d in range(0, 2):
#         for q in range(0, 2):
#             model = ARIMA(train, order=(p,d,q))
#             results = model.fit()
#             predictions = results.forecast(steps=100)
#             aic = results.aic
#             if aic < min:
#                 min = aic
#                 min_thing = (p, d, q)
#             print("%i%i%i - %f" % (p, d, q, aic))
# print(min, min_thing)

# Best: (6, 0, 1) or (13, 1, 1)
# model = sp.SARIMAX(cut_data, order=(13, 1, 1))
# model_fit = model.fit()
# # print(model_fit.summary())
# predictions = model_fit.forecast(steps=100)
#
# pyplot.plot(cut_data)
# pyplot.plot(predictions)
# pyplot.show()