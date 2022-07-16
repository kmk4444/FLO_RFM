# Bussines Problem
#Segmenting the customers of FLO, an online shoe store, wants to make sense according to these segments.
#It will be designed accordingly and will be created according to this particular clustering.
#FLO, Wants to determine marketing strategies according to these segments.

# Variables
# master_id : Unique Customer Number
# order_channel : Which channel of the shopping platform is used (Android, IOS, Desktop, Mobile)
# last_order_channel : The channel where the most recent purchase was made
# first_order_date : Date of the customer's first purchase
# last_order_channel : Customer's previous shopping history
# last_order_date_offline : The date of the last purchase made by the customer on the offline platform
# order_num_total_ever_online : Total number of purchases made by the customer on the online platform
# order_num_total_ever_offline : Total number of purchases made by the customer on the offline platform
# customer_value_total_ever_offline : Total fees paid for the customer's offline purchases
# customer_value_total_ever_online : Total fees paid for the customer's online purchases
# interested_in_categories_12 : List of categories the customer has shopped in the last 12 months

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
from matplotlib import pyplot as plt
import seaborn as sns

df_ = pd.read_csv("WEEK_3/Ödevler/FLO_RFM/flo_data_20k.csv")
df = df_.copy()
df.head()

###################################### Task 1 ###############################

# Step 1 :Analyze data

def check_df(dataframe, head=5):
    print("############### shape #############")
    print(dataframe.shape)
    print("############### types #############")
    print(dataframe.dtypes)
    print("############### head #############")
    print(dataframe.head())
    print("############### tail #############")
    print(dataframe.tail())
    print("############### NA #############")
    print(dataframe.isnull().sum())
    print("############### Quantiles #############")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Step 2: omnichannel means that customer had bought a product both online and offline platforms.
# we need to find total number of shopping and cost

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# Step 3: find the date variable and convert them.
df.loc[:,df.columns.str.contains("date")] = df.loc[:,df.columns.str.contains("date")].astype("datetime64")

# Step 4: analyze the total amount of product and spending by orders_channel

df.groupby("order_channel").agg({"master_id": "count",
                                 "order_num_total": "sum",
                                 "customer_value_total":"sum"})

# Step 5: Rank the top 10 customers who are the highest revenue.

df.sort_values(by="customer_value_total", ascending= False).head(10)

# Step 6: # Step 5: Rank the top 10 customers who are the highest orders.

df.sort_values(by="order_num_total", ascending= False).head(10)

# Step 7: create a function to  the data preparation process.

def data_preparation(dataframe):

    # Total number of customers' orders and spend
    dataframe["order_num_total"] = dataframe.loc[:, dataframe.columns.str.contains("order_num_total")].sum(axis=1)
    dataframe["customer_value_total"] = dataframe.loc[:, dataframe.columns.str.contains("customer_value_total")].sum(axis=1)

    # date (column)
    dataframe.loc[:, dataframe.columns.str.contains("date")] = dataframe.loc[:, dataframe.columns.str.contains("date")].apply(lambda x : x.astype('datetime64[ns]'))

data_preparation(df)

################################## Task 2 #######################
# Step 1: find recency, frequency and monetary
df["last_order_date"].max()
today_date = today_date = dt.datetime(2021,6,1)


rfm = df.groupby("master_id").agg({"last_order_date": lambda x: (today_date - x.max()).days,
                             "order_num_total": lambda x: x.sum(),
                             "customer_value_total": lambda x: x.sum()})
rfm.columns = ["Recency", "Frequency", "Monetary"]
rfm. head()

################################## Task 3 #######################
# Step 1: find point of recency, frequency and monetary.

rfm["recency_score"] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1])
rfm["frequency_score"] = pd.qcut(rfm["Frequency"].rank(method = "first"), 5, labels=[1,2,3,4,5])
rfm["monetary_score"] = pd.qcut(rfm["Monetary"],  5, labels=[1,2,3,4,5])

rfm["RF_SCORE"] = (rfm["recency_score"].astype("str") + rfm["frequency_score"].astype("str"))

rfm.head()

################################## Task 4 #######################
# Step 1: create segments according to rf score

seg_map = {
    r"[1-2][1-2]" : "hibernating",
    r"[1-2][3-4]" : "at_Risk",
    r"[1-2]5" : "cant_loose",
    r"3[1-2]" : "about_to_sleep",
    r"33" : "need_attention",
    r"[3-4][4-5]" :"loyal_customers",
    r"41" : "promising",
    r"51" : "new_customers",
    r"[4-5][2-3]" : "potential_loyalists",
    r"5[4-5]" : "champions"
}

rfm["Segments"] = rfm["RF_SCORE"].replace(seg_map, regex=True)
rfm.head()

################################## Task 5 #######################
# Analyze rfm segments
print(rfm.groupby("Segments").agg({"Recency":"mean",
                            "Frequency": "mean",
                            "Monetary":"mean"}))

# Case 1

#A new women's shoe brand will be included.
# The target audience (champions,loyal_customers) and women are determined as shoppers.' \
# We need access to the id numbers of these customers.

new_df = pd.merge(df, rfm, on="master_id")[["master_id","RF_SCORE","Segments","interested_in_categories_12"]]

for i in range(len(new_df["interested_in_categories_12"])):
    if "KADIN" not in new_df["interested_in_categories_12"][i]:
        new_df = new_df.drop(i, axis=0)

new_df = new_df.loc[(new_df["Segments"] == "champions") | (new_df["Segments"] == "loyal_customers"),:]
new_df = new_df["master_id"]
new_df.to_csv("rfm_woman.csv")

# Graph - 1

colors  = ("darkorange", "darkseagreen", "orange", "cyan", "cadetblue", "hotpink", "lightsteelblue", "coral",  "mediumaquamarine","palegoldenrod")
explodes = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]

rfm["Segments"].value_counts(sort=False).plot.pie(colors=colors,
                                                 textprops={'fontsize': 12},
                                                 autopct = '%4.1f',
                                                 startangle= 90,
                                                 radius =2,
                                                 rotatelabels=True,
                                                 shadow = True,
                                                 explode = explodes)
plt.ylabel("");

# Graph - 2 

rfm_coordinates = {"champions": [3, 5, 0.8, 1],
                   "loyal_customers": [3, 5, 0.4, 0.8],
                   "cant_loose": [4, 5, 0, 0.4],
                   "at_Risk": [2, 4, 0, 0.4],
                   "hibernating": [0, 2, 0, 0.4],
                   "about_to_sleep": [0, 2, 0.4, 0.6],
                   "promising": [0, 1, 0.6, 0.8],
                   "new_customers": [0, 1, 0.8, 1],
                   "potential_loyalists": [1, 3, 0.6, 1],
                   "need_attention": [2, 3, 0.4, 0.6]}

fig, ax = plt.subplots(figsize=(20, 10))

ax.set_xlim([0, 5])
ax.set_ylim([0, 5])

plt.rcParams["axes.facecolor"] = "white"
palette = ["#282828", "#04621B", "#971194", "#F1480F", "#4C00FF",
           "#FF007B", "#9736FF", "#8992F3", "#B29800", "#80004C"]

for key, color in zip(rfm_coordinates.keys(), palette[:10]):
    coordinates = rfm_coordinates[key]
    ymin, ymax, xmin, xmax = coordinates[0], coordinates[1], coordinates[2], coordinates[3]

    ax.axhspan(ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax, facecolor=color)

    users = rfm[rfm.Segments == key].shape[0]
    users_percentage = (rfm[rfm.Segments == key].shape[0] / rfm.shape[0]) * 100
    avg_monetary = rfm[rfm.Segments == key]["Monetary"].mean()

    user_txt = "\n\nTotal Users: " + str(users) + "(" + str(round(users_percentage, 2)) + "%)"
    monetary_txt = "\n\n\n\nAverage Monetary: " + str(round(avg_monetary, 2))

    x = 5 * (xmin + xmax) / 2
    y = (ymin + ymax) / 2

    plt.text(x=x, y=y, s=key, ha="center", va="center", fontsize=18, color="white", fontweight="bold")
    plt.text(x=x, y=y, s=user_txt, ha="center", va="center", fontsize=14, color="white")
    plt.text(x=x, y=y, s=monetary_txt, ha="center", va="center", fontsize=14, color="white")

    ax.set_xlabel("Recency Score")
    ax.set_ylabel("Frequency Score")

sns.despine(left=True, bottom=True)
plt.show()

