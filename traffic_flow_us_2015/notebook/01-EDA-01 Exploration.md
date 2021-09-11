---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Objective
To find the most obvious 5 patterns.

# Data Source
## Context
Traffic management is a critical concern for policymakers, and a fascinating data question. This ~2gb dataset contains daily volumes of traffic, binned by hour. Information on flow direction and sensor placement is also included.

## Content
Two datasets are included:

- dot_traffic_2015.txt.gz
    - daily observation of traffic volume, divided into 24 hourly bins
station_id, location information (geographical place), traffic flow direction, and type of road
- dot_traffic_stations_2015.txt.gz
    - deeper location and historical data on individual observation stations, cross-referenced by station_id

## Acknowledgements
This dataset was compiled by the US Department of Transportation and available on Google BigQuery

## Inspiration
- Where are the heaviest traffic volumes? 
    - By day of the year?
    - By station id?
    - By time of day?
    - By type of road?
- Any interesting seasonal patterns to traffic volumes?
- What affects the traffic situation? Holidays? Weather?


# Setup


## Import Libraries

```python
import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly

import holidays
import folium
# from dataprep.eda import plot, plot_correlation, plot_missing

from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from sklearn.metrics import silhouette_samples
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from datetime import date
from IPython.display import HTML
```

## Path Configuration

```python
DIRECTORY = '../data/'
TRAFFIC_PATH = DIRECTORY + 'dot_traffic_2015.txt.gz'
STATION_PATH = DIRECTORY + 'dot_traffic_stations_2015.txt.gz'
```

## Display Configuration

```python
pd.options.display.max_columns = None
FIGSIZE = (12, 10)
```

## Global Configuration

```python
SEED = 888
```

## Global Function

```python
def plot_missing_bar(df, figsize):
    (df.isna().sum() / df.shape[0]).sort_values().plot(kind='barh', 
                                                       figsize=figsize, 
                                                       title='Missing Value Percentage'
                                                      )
    plt.show()
```

```python
def plot_silhouette_ts_kmeans(n_clusters, X_ts, X, metric, seed, col_1, col_2):
    for n in tqdm(n_clusters):
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X_ts) + (n + 1) * 10])

        # Initialize the clusterer with n value and a random generator
        # seed of 10 for reproducibility.
        clusterer = TimeSeriesKMeans(n_clusters=n, metric=metric, random_state=seed)
        cluster_labels = clusterer.fit_predict(X_ts)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n =", n,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n)
        ax2.scatter(X.iloc[:, col_1], X.iloc[:, col_2], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n = %d" % n),
                     fontsize=14, fontweight='bold')

    plt.show()    
```

# Data


## Read Data

```python
df_traffic = pd.read_csv(TRAFFIC_PATH)
df_station = pd.read_csv(STATION_PATH)
```

## Basic Data  Exploration

```python
df_traffic.info()
```

```python
df_station.info()
```

```python
print(f'Traffic data shape: {df_traffic.shape}')
print(f'Station data shape: {df_station.shape}')
```

### Adding Columns and Changing Data Type

```python
df_traffic.loc[:, 'date'] = pd.to_datetime(df_traffic.date, format='%Y-%m-%d')
df_traffic.loc[:, 'fips_state_code']= df_traffic.fips_state_code.astype('category')

us_holidays = holidays.UnitedStates(years=2015)
df_traffic.loc[:, 'holiday'] = df_traffic.date.isin(us_holidays['2015-01-01':'2015-12-31'])
```

```python
# since we at us region, the longitude need to multiply by -1 to map in folium
df_station.loc[:, 'longitude_1'] = df_station.longitude * - 1
df_station.loc[:, 'fips_state_code']= df_station.fips_state_code.astype('category')
df_station.loc[:, 'fips_county_code']= df_station.fips_county_code.astype('category')
```

```python

```

```python
mask = df_station.station_id.isin(df_traffic.station_id)
df_station1 = df_station.loc[mask, :]
```

```python
df_traffic.columns
```

```python
df_station1.columns
```

```python
cols = ['latitude', 'longitude_1']
df_station1.loc[:, cols].describe()
```

We knew that latitude is from range -90 to 90 and longitude is from range -180 to 80. So possibly we will need to remove some of the station because of the location is not really traceable.


### Cardinality


#### Stations

```python
station_nunique = df_station1.nunique().sort_values()
station_nunique
```

```python
drop_cols = station_nunique[station_nunique==1].index.tolist()
print(drop_cols)

df_station.drop(columns=drop_cols, inplace=True)
```

#### Traffic

```python
traffic_nunique = df_traffic.nunique().sort_values()
traffic_nunique
```

```python
drop_cols = traffic_nunique[traffic_nunique <= 1].index.tolist()
print(drop_cols)

df_traffic.drop(columns=drop_cols, inplace=True)
```

### Missing Values


#### Station

```python
plot_missing_bar(df_station, figsize=(12,10))
```

Seems like either:
- there are station id change to another station id.
- or the previous station id is known just the indicating the previous station.

```python
df_station.station_id.isin(df_station.previous_station_id.dropna()).any()
```

```python
mask = df_station.previous_station_id.notna()
df_station.loc[mask, :].head()
```

```python
mask = df_station.station_id == '000000'
df_station.loc[mask, :]
```

So, the previous station id meant that the station that is before this station. So they are interconnected instead of the id change from old to new.


#### Traffic

```python
plot_missing_bar(df_traffic, figsize=(10,8))
```

No missing values in traffic data.


# EDA


## General EDA


### Traffic

```python
pattern = 'traffic_volume_counted'
tv_cols = [x for x in df_traffic.columns if re.findall(pattern, x) != []]
print(tv_cols)
```

```python
for i in range(0, len(tv_cols), 6):
    df_traffic.loc[:, f'before_{i+6}'] = df_traffic.loc[:, tv_cols[i:(i+6)]].sum(axis=1)
```

```python
df_traffic.describe(include='all')
```

From the descriptive statistics, we can observe that
- the lane of travel can up to 9 which is matching with the direction of travel
- fips_state_code, 12 is having the most entries.
- functional classification name, urban: principal arterial - interstate have the most entries
- station_id 000050 having the most entries.

```python
df_traffic
```

From the table, we found out that
- there are lane of travel of 0, yet there are having car to pass by.
- some of the station have up to direction of travel.
- day of week is given, so possibly we can compare both weekdays and weekend (day of week start with 1 which is a sunday) - might require some checking.

```python
# validate if the given day of week column is correct
assert ((df_traffic.date.dt.dayofweek + 2).replace(8, 1) == df_traffic.day_of_week).all()
```

```python
df_traffic.loc[:, tv_cols].plot(kind='hist', figsize=(15,20), subplots=True, layout=(12, 2), bins=200)
plt.show()
```

Generally it's all right skewed.


Based on these findings, we can
- aggregate traffic volume mean or median or sum by date while retaining per hour columns
- finding out what date is having higher traffic volume compare to other days
- aggregate traffic volumn mean or median or sum by station id while retaining per hour columns

The feature engineering can be
- aggregate the hours on 6 hours basis
- checking if it's holiday
- weather (season)

```python
index = 'date'
df_temp = df_traffic.groupby(index)[tv_cols].sum()
plt.figure(figsize=(15,8))
df_temp.mean().plot(rot=45, label='mean')
df_temp.median().plot(rot=45, label='median')
df_temp.std().plot(rot=45, label='std')
plt.title('Aggregated Sum of Traffic Volume by Hours')
plt.ylabel('Aggregated Sum of Traffic Volume')
plt.xlabel('Time of the Day')
plt.legend()
plt.show()
```

#### Finding 1
This is the most obvious general trend where there are always a peak around 6am-7am and 4pm-6pm, which we can also see the standard deviationis fluctuated around that time but the deviation is lower than the mean and median.

```python
index = 'station_id'
df_temp = df_traffic.groupby(index)[tv_cols].sum()
# df_temp.sum().sum().plot(rot=45)
plt.figure(figsize=(15,8))
df_temp.mean().plot(rot=45, label='mean')
df_temp.median().plot(rot=45, label='median')
df_temp.std().plot(rot=45, label='std')
plt.title('Aggregated Sum of Traffic Volume by Station Id')
plt.ylabel('Aggregated Sum of Traffic Volume')
plt.xlabel('Time of the Day')
plt.legend()
plt.show()
```

#### Finding 2
We can observe the trend is similar with the plot by date. However, the standard deviation is way higher than mean and median, which indicate the traffic volume have a huge deviation when compare station by station, which is worth to investigate which stations are the one contribute the most.


## Top 10 Station ID by Traffic Volume

```python
index = 'station_id'
df_temp = df_traffic.groupby(index)[tv_cols].sum()
traffic_volume_by_station_id = df_temp.sum(axis=1).sort_values(ascending=False)
traffic_volume_by_station_id.head(10)
```

```python
index = ['station_id', 'direction_of_travel_name', 'fips_state_code']
df_traffic.groupby(index)[tv_cols].sum().sum(axis=1).sort_values(ascending=False).head(10)
```

From here itself we knew that some of the station id having a high traffic flow is due to the interconnection nature of their station. However, when we are comparing a certain direction, the top 10 station id change to another set except a few of them.

```python
# plotting folium map to have middle location
lat = df_station1.latitude.mean()
lng = df_station1.longitude_1.mean()

map_station = folium.Map(location=(lat, lng), zoom_start=3)
df_temp = df_station1.dropna(subset=['latitude', 'longitude_1'])

# taking only top 10 station id to check where is it located
# filtered those invalid longitude that is shown in Basic EDA
mask1 = df_temp.station_id.isin(traffic_volume_by_station_id.head(10).index)
mask2 = df_temp.longitude_1 >= -180
mask = mask1 & mask2
df_temp = df_temp.loc[mask, :]
df_temp.apply(lambda row:folium.CircleMarker(location=[row["latitude"], row["longitude_1"]], 
                                             radius=10, popup=row[['station_id','station_location', 'latitude', 'longitude_1']])
              .add_to(map_station), axis=1)
display(map_station)
```

From the map itself we can see 
- It is from the united state central region. 
- Notice that the circle plot on the map is more than 10. This meant that one station have multiple data points and connecting to other routes which can be validate.

```python
df_temp.groupby('station_id')['latitude'].count()
```

```python
# plotting folium map to have middle location
lat = df_station1.latitude.mean()
lng = df_station1.longitude_1.mean()

map_station = folium.Map(location=(lat, lng), zoom_start=3)
df_temp = df_station1.dropna(subset=['latitude', 'longitude_1'])

# taking only last 10 station id to check where is it located
# filtered those invalid longitude that is shown in Basic EDA
mask1 = df_temp.station_id.isin(traffic_volume_by_station_id.tail(10).index)
mask2 = df_temp.longitude_1 >= -180
mask = mask1 & mask2
df_temp = df_temp.loc[mask, :]
df_temp.apply(lambda row:folium.CircleMarker(location=[row["latitude"], row["longitude_1"]], 
                                             radius=10, popup=row[['station_id','station_location', 'latitude', 'longitude_1']])
              .add_to(map_station), axis=1)
display(map_station)
```

Based on the circle plot here, we noticed that the there are 11 circle, which then these indicates that the map that showing top 10 station id just now is affected by directions itself.


## Top 10 Station Id with Direction by Traffic Volume

```python
index = ['station_id', 'direction_of_travel_name', 'fips_state_code']
top_10_station_id_direction = df_traffic.groupby(index)[tv_cols].sum().sum(axis=1).sort_values(ascending=False).head(10).reset_index()
top_10_station_id_direction = top_10_station_id_direction.rename(columns={0:'traffic_volume_sum'})
```

```python
# plotting folium map to have middle location
lat = df_station1.latitude.mean()
lng = df_station1.longitude_1.mean()

map_station = folium.Map(location=(lat, lng), zoom_start=3)
df_temp = df_station1.dropna(subset=['latitude', 'longitude_1'])

# taking only last 10 station id to check where is it located
# filtered those invalid longitude that is shown in Basic EDA
df_temp = pd.merge(df_temp, top_10_station_id_direction, on=index)
mask = df_temp.longitude_1 >= -180
df_temp = df_temp.loc[mask, :]
df_temp.apply(lambda row:folium.CircleMarker(location=[row["latitude"], row["longitude_1"]], 
                                             radius=10, popup=row[['station_id','station_location', 'latitude', 'longitude_1']])
              .add_to(map_station), axis=1)

display(map_station)
```

## Top 5 State by Traffic Volume

```python
index = ['fips_state_code']
traffic_volume_by_state = df_traffic.groupby(index)[tv_cols].sum().sum(axis=1).sort_values(ascending=False)
```

```python
# plotting folium map to have middle location
lat = df_station1.latitude.mean()
lng = df_station1.longitude_1.mean()

map_station = folium.Map(location=(lat, lng), zoom_start=4)
df_temp = df_station1.dropna(subset=['latitude', 'longitude_1'])

# taking only top 10 station id to check where is it located
# filtered those invalid longitude that is shown in Basic EDA
mask1 = df_temp.fips_state_code.isin(traffic_volume_by_state.head(5).index)
mask2 = df_temp.longitude_1 >= -180
mask = mask1 & mask2
df_temp = df_temp.loc[mask, :]
df_temp.apply(lambda row:folium.CircleMarker(location=[row["latitude"], row["longitude_1"]], 
                                             radius=1, popup=row[['station_id','station_location', 'latitude', 'longitude_1']])
              .add_to(map_station), axis=1)

display(map_station)
```

### Finding 3
We can see there are possibly a lot vehicle within these area which is more to central area.
- Los Angles
- Phoenix
- Washington
- Ohio
- Florida


## Top 5 County by Traffic Volume

```python
index = ['functional_classification_name']
traffic_volume_by_fcn = df_traffic.groupby(index)[tv_cols].sum().sum(axis=1).sort_values(ascending=False)
```

```python
traffic_volume_by_fcn
```

### Finding 4
There's a lot of traffic flow on urban compare to rural area. But possibly due to there are more route in urban area compared to rural area. Hence we count the number of station id (not unique because we are counting how many connections are there as well)

```python
df_station.groupby('functional_classification_name')['station_id'].count().sort_values(ascending=False)
```

Apparently, the station that they are having seems to be having lesser difference than expected. So, the traffic flow is probably caused by the population instead of the road built in rural area.


## Traffic Volume by Date

```python
index = ['date']
group_tv_by_date = df_traffic.groupby(index)[tv_cols]

df_agg_tv_by_date_mean = group_tv_by_date.mean()
df_agg_tv_by_date_median = group_tv_by_date.median()
```

### Visualization - Line Plot


#### Mean

```python
df_agg_tv_by_date_mean.plot(figsize=FIGSIZE)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.show()
```

It seems there's cyclic and seasonal pattern in the plot.


#### Median

```python
df_agg_tv_by_date_median.plot(figsize=FIGSIZE)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.show()
```

Seems like abit similar to what we have on mean graph except that the maximum range is lower comparitively.
However, we couldn't make a conclusive analysis here. Hence we will move towards to subplot to have a better understanding.


### Visualization - SubPlot for Line


#### Mean

```python
df_agg_tv_by_date_mean.plot(figsize=(15,20), subplots=True, layout=(12, 2))
plt.show()
```

<!-- #region -->
Due to it's smaller size, hence we might misinterpret certain plot. However this is just to provide an intuition on what we can do next.

From the y axis, we can observe that generally 
- The traffic volume is generally lower from 12am to 5am.
- It started to increase and having a peak around 7am to 8am.
- The trend fell slightly on 8am to 11am then increase again.
- Thre traffic started to decrease from 6pm onwards.


We can observe the trend shape across the date are 
- Generally the same for the traffic volume on 12am to 3am
- However the 4am looks slightly different which seems like a combination of 2am-3am and 4am to 5am.
- 5am to 10am are generally the same as well which we can see there's are relatively flat high peak which are suspected to be weekdays.
- 10am to 6pm are generally the same where there are random peaks, however coincidentally the peak are sometimes fell in the lowest point of 5am to 10am. It's suspected that this part of trend might be influence by holiday and weekend where the traffic volume increased at that time.
- 6pm to 12am have a relatively similar trend.

The fall within every hours of the graph are suspected to be
- weekend where majority of people is not going to work.
<!-- #endregion -->

#### Median

```python
df_agg_tv_by_date_median.plot(figsize=(15,20), subplots=True, layout=(12, 2))
plt.show()
```

Generally the median traffic volume by date exhibit the same trend, except that the maximum scale value is lower compare to mean traffic volume by date.


### Visualization - Histogram


#### Mean

```python
df_agg_tv_by_date_mean.plot(kind='hist', figsize=(15,20), subplots=True, layout=(12, 2))
plt.show()
```

From this plot, we have more idea on how the traffic volume is distributed.
- we can observe the distribution is shifting to the right slowly until 6pm then shift back to left.


#### Median

```python
df_agg_tv_by_date_median.plot(kind='hist', figsize=(15,20), subplots=True, layout=(12, 2))
plt.show()
```

We can observe that the major trend moving from 12am to 12pm is generally the same, but there are certain hours of median traffic volume by date is different with mean traffic volume by date. Majority of them will is telling them there are more frequency in a lower bin, eg. 4am-6am. This probably meant that there are stations on a certain date have a very high amount of traffic volume which then dragging the mean value towards the right of the graph.


### Time Series Clustering

```python
# standardize to reduce the time taken
sc = StandardScaler()
sc_tv_by_date_mean = sc.fit_transform(df_agg_tv_by_date_mean)
print(sc_tv_by_date_mean.shape)
```

```python
ts_tv_by_date_mean = to_time_series_dataset(df_agg_tv_by_date_mean)
print(ts_tv_by_date_mean.shape)
```

Due to time constraint, we make several assumptions here to shorten the process of evaluation:
- testing cluster number from 2 to 12 instead of 2 to 365 is because that we think one year have 12 months have possibly there wont be such a deviate pattern on every day.
- even though there are other methods such as elbow test, looking at distance between cluster and distance within cluster to measure cluster information, we picked on silhouette score.
- our standardization apparently performed on column basis, which meant it standardize accross the same hour for everyday instead of standardize across everyday. We might need to test on this standardize / checking on difference on traffic flow when we have the time.
- we did not split the train test as we does not have any supervised label here to predict but just to find out the pattern.

```python
n_clusters = range(2, 6)
plot_silhouette_ts_kmeans(n_clusters, ts_tv_by_date_mean, df_agg_tv_by_date_mean, 'softdtw', SEED, 0, 1)
```

From the silhoutte plot, it seems cluster number 2 and 3 will be a good pick judging based on
- presense of cluster are above average silhouette score.
- the size of silhouette seems a bit uneven, but based on the visualization the clustering is okay to have it.
- the most even will be size = 3 in this case


```python
plot_silhouette_ts_kmeans(n_clusters, ts_tv_by_date_mean, df_agg_tv_by_date_mean, 'dtw', SEED, 0, 1)
```

We came to the same conclusion while using dtw or dtw. Hence, the final n_cluster that we pick is 2 as silhouette average score is higher and also it take lesser computation time.

```python
choosen_n = 2
tskm_tv_by_date = TimeSeriesKMeans(n_clusters=choosen_n, metric='softdtw', random_state=SEED)
tskm_tv_by_date.fit(ts_tv_by_date_mean)
labels_tv_by_date = tskm_tv_by_date.predict(ts_tv_by_date_mean)
```

```python
for label in range(0, choosen_n):
    print(f'Cluster {label}')
    mask = labels_tv_by_date == label
    df_temp = df_agg_tv_by_date_mean.loc[mask, :].transpose()
    df_temp.plot(figsize=(15,8), rot=45, legend=False,)    
    plt.show()
```

Judging from here,
- We can observe the cluster 0 seems to be having some similar pattern, which is two peak and one peak is around 6am to 8am and the other peak will be around 4pm - 6pm. Seems like the typical working hours.
- Cluster 1 is the only one obvious that is having peak hours from 10am-4pm, which seems like the holiday. However there are still part of the cluster 0 data within it.

```python
df_agg_tv_by_date_mean.loc[:, 'holiday'] = df_agg_tv_by_date_mean.index.isin(us_holidays['2015-01-01':'2015-12-31'])
df_agg_tv_by_date_mean.loc[:, 'cluster'] = labels_tv_by_date
```

```python
mask = df_agg_tv_by_date_mean.cluster == 1
holiday_percentage = df_agg_tv_by_date_mean.loc[mask, 'holiday'].sum() / len(us_holidays['2015-01-01':'2015-12-31'])
print(f'Cluster 1 consist of {holiday_percentage*100:.2f}% of total holidays')
```

```python
df_temp = df_agg_tv_by_date_mean.reset_index()

# the day of week here start with monday as 0, sunday = 6
df_temp.loc[:, 'day_of_week'] = df_temp.date.dt.dayofweek
df_temp.loc[:, 'weekend'] = df_temp.day_of_week >= 4
```

```python
df_temp.groupby('cluster').weekend.value_counts(normalize=True)
```

#### Findings 5
We can conclude that when we are looking only on traffic volume by date by hours, then we generally will have 2 patterns which are
- cluster 0 - the normal working hour pattern where it's having peak hour on 6am to 8am and 4pm to 6pm
- cluster 1 - the holiday kind of moment where people went out on around 10am to 12pm


# Conclusion

The most obvious 5 pattern:
- Finding 1 - the peak hour is around 6am-8am and 4pm-6pm in general.
- Finding 2 - the traffic volume is affected by their location and also how many direction that the station can go to.
- Finding 3 - the top 5 states that is having alot of traffic - Los Angles,Phoenix, Washington, Ohio, Florida
- Finding 4 - the traffic flow in urban area is more than rural area but is not caused by the route build. Possibly due to the population.
- Finding 5 - the further differentiateble pattern is that there are two patterns, working days and holiday which having different peak.

<!-- #region -->
# Suggestion for Improvement


We can try to 
- utilize station data, since there are different type of data collection which potentially impact our analysis
    - different algorithm of vehicle classification
    - different calibration of weighing system
    - different classification system for vehicle
    - lane of travel and direction of travel
    - method of traffic volume counting
- try to link the station route together to see how many person stop at each station since we have previous station id with the direction.
<!-- #endregion -->
