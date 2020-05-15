import pandas as pd

data = pd.read_csv('q1.csv')
# Notice that there are duplicates in the database

data.ts = pd.to_datetime(data.ts)
data_bdv = data[data.country_id == 'BDV']

# Q1 - get the unique values using nunique
res_q1 = data_bdv.groupby('site_id')['user_id'].nunique()

# Q2
# Filter the data
data_dates = data[(data.ts >= '2019-02-03 00:00:00') & (data.ts <= ' 2019-02-04 23:59:59')]

# First count the number of times each user visited each site
user_site = data_dates.groupby(['site_id', 'user_id'])['ts'].count().reset_index()

# Next we filter by number of visits
res_q2 = user_site[user_site.ts > 10].sort_values(by = 'ts', ascending = False)

# Q3
# First find the last visit of each user
last_visit = data.groupby('user_id')['ts'].last().reset_index()

# Filter the data to only have the last users - like a subquery
last_users = data.merge(last_visit, how = 'inner')

# Count unique values
res_q3 = last_users.groupby('site_id')['user_id'].nunique().sort_values(ascending = False)

# Q4
# Get the first visit
first_visit = data.groupby('user_id')['ts'].first().reset_index()

# Filter the data to only have the first users
first_users = data.merge(first_visit, how = 'inner')[['user_id', 'site_id']]
last_users_sub = last_users[['user_id', 'site_id']]

res_q4 = first_users.merge(last_users_sub, how = 'inner').drop_duplicates()
# Notice that a lot of these cases are users with only one visit