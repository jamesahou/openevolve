import pandas as pd
import pickle

df = pd.read_csv("ks_law_dataset_ned_redshift.csv")

df = df[["log_gas", "Redshift_NED", "log_SFR"]]

df_train = df[df["Redshift_NED"] <  0.005]
df_test = df[df["Redshift_NED"] >= 0.005]

df_train.to_csv("ks_dataset_train.csv", index=False)
df_test.to_csv("ks_dataset_test.csv", index=False)

df_train = list(df_train.itertuples(index=False, name=None))
df_test = list(df_test.itertuples(index=False, name=None))

with open("ks_dataset_train.pickle", "wb") as f:
    pickle.dump(df_train, f)

with open("ks_dataset_test.pickle", "wb") as f:
    pickle.dump(df_test, f)