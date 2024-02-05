import pandas as pd
import datetime as dt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width',1000)
df_=pd.read_csv("C:/Users/Ali/Desktop/PROJECTS/FLOMusteriSegmentasyonu/flo_data_20k.csv")
df=df_.copy()

def data_obs(dataframe): #Function for observing the dataset to decide if feature engineering is necessary or not.
    print(dataframe.head(10))
    print(dataframe.columns)
    print(dataframe.shape)
    print(dataframe.isnull().sum())
    print(dataframe.describe().T)
    return df

data_obs(df)


#Creating new variables to see the total contribution of any customer
df["order_num_total"]=df["order_num_total_ever_online"]+df["order_num_total_ever_offline"]
df["customer_value_total"]=df["customer_value_total_ever_offline"]+df["customer_value_total_ever_online"]
print(df.columns)

#Changing the data type as date for necessary columns
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)


print(df.groupby("order_channel").agg({"master_id":"count",
                                 "order_num_total":"sum",
                                 "customer_value_total":"sum"}))


print(df.sort_values("customer_value_total", ascending=False)[0:10])
print(df.sort_values("order_num_total", ascending=False)[0:10])


print(df["last_order_date"].max()) #2021-05-30
analysis_date=dt.datetime(2021,6,1)

#Creating RFM scores in a new DataFrame
rfm=pd.DataFrame()
rfm["customer_id"]=df["master_id"]
rfm["recency"]= (analysis_date - df["last_order_date"]).astype("timedelta64[s]")
rfm["frequency"]=df["order_num_total"]
rfm["monetary"]=df["customer_value_total"]

print(rfm.head())


rfm["recency_score"]=pd.qcut(rfm["recency"],5,labels=[5,4,3,2,1])
rfm["monetary_score"]=pd.qcut(rfm["monetary"],5,labels=[1,2,3,4,5])
rfm["frequency_score"]=pd.qcut(rfm["frequency"].rank(method="first"),5,labels=[1,2,3,4,5])
rfm["RF_Score"]=rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)
rfm["RFM_SCORE"] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str)

print(rfm.head())

#Customer Identification
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_Score'].replace(seg_map, regex=True)

print(rfm.head())


#Average RFM values of segments
print(rfm.groupby("segment").agg({"recency":["mean", "count"],
                            "monetary":["mean", "count"],
                            "frequency":["mean", "count"]}))

