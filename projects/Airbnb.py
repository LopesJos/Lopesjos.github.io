#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import widgets, interactive


# In[2]:


df = pd.read_csv('Airbnb_Open_Data.csv',low_memory=False)


# In[3]:


df.head(1)


# In[4]:


df.dtypes


# In[5]:


df.rename(columns={"host id": "host_id", 
                   "host name": "host_name",
                   "neighbourhood group": "neighbourhood_group",
                   "service fee": "service_fee",
                    "minimum nights": "min_nights",
                    "number of reviews": "number_of_reviews",
                    "last review": "last_review",
                    "reviews per month": "reviews_per_month",
                   "review rate number": "avg_review",
                   "calculated host listings count": "calculated_host_listings_count",
                   "availability 365": "availability_365"
                  },inplace=True)


# In[6]:


df.rename(str.upper, axis='columns',inplace=True)


# In[7]:


df['NEIGHBOURHOOD_GROUP'].unique()


# In[8]:


df.replace({'brookln': 'Brooklyn', 'manhatan': 'Manhattan'}, inplace=True)


# In[9]:


df.isnull().sum()


# In[10]:


df.nunique()


# In[11]:


df.LICENSE.unique()


# In[12]:


df.shape


# In[13]:


df.drop_duplicates(inplace=True)


# In[14]:


df.drop(['LICENSE','COUNTRY','COUNTRY CODE'],inplace=True,axis=1)


# In[15]:


df['duplicado']= df.duplicated(subset=['HOST_ID'],keep=False)


# In[16]:


df_2= df[df['duplicado']==True]


# In[17]:


df_2.value_counts('HOST_ID')


# In[18]:


df[df['HOST_ID']==29531702698]


# In[19]:


df.dropna(subset=['AVG_REVIEW','NUMBER_OF_REVIEWS','LAT','LONG'],inplace=True)

df.head(1)


# In[20]:


df.drop(['duplicado'],inplace=True,axis=1)


# In[21]:


df.isnull().sum()


# In[22]:


df.PRICE


# In[23]:


price_review=df.groupby(['PRICE'])['AVG_REVIEW'].agg('mean').reset_index()
price_review['PRICE'] = price_review['PRICE'].str.replace(',', '').str.replace('$', '').astype(int)
price_review.sort_values(by='PRICE',
                         inplace=True,
                         ignore_index=True)
price_review


# In[24]:


plt.figure(figsize=(20,6))
sns.lineplot(data=price_review, 
                x="PRICE",
                y="AVG_REVIEW"
            )


# In[25]:


plt.figure(figsize=(20,6))
df['PRICE'] = df['PRICE'].str.replace(',', '').str.replace('$', '').astype(float)
sns.barplot(data=df,
       x='NEIGHBOURHOOD_GROUP',
       y='PRICE', 
       hue='AVG_REVIEW'
       )
plt.show()


# In[26]:


plt.figure(figsize=(20,6))
sns.boxplot(data=df,
       x='NEIGHBOURHOOD_GROUP',
       y='AVG_REVIEW', 
       )
plt.show()


# In[27]:


df.head(1)


# In[28]:


#df_Brooklyn=df[df['NEIGHBOURHOOD_GROUP'] == 'Brooklyn']
df_byBrooklyn=df.groupby(['NEIGHBOURHOOD_GROUP','NEIGHBOURHOOD'])['AVG_REVIEW','PRICE'].agg('mean').reset_index()
df_byBrooklyn.head(1)


# In[29]:


from IPython.display import display


# In[39]:


# Define the function to update the plot
def update_plot(NEIGHBOURHOOD_GROUP,PARAMETERS):
    
    fig, ax = plt.subplots(figsize=(30, 8))
    
    # Clear the previous plot
    ax.clear()
    
    # Filter the data based on the selected day
    filtered_df = df_byBrooklyn[df_byBrooklyn['NEIGHBOURHOOD_GROUP']== NEIGHBOURHOOD_GROUP ]
    
    Mean = filtered_df.groupby('NEIGHBOURHOOD_GROUP')[PARAMETERS].mean().reset_index()
    Mean = Mean[Mean['NEIGHBOURHOOD_GROUP']== NEIGHBOURHOOD_GROUP]
    Mean = Mean[PARAMETERS][0]
    
    highlight_colour = 'm'
    non_highlight_colour = 'c'

    #filtered_df['colours'] = filtered_df[PARAMETERS].apply(lambda x: highlight_colour if x >= Mean else non_highlight_colour)
    filtered_df.loc[filtered_df[PARAMETERS] >= Mean, 'colours'] = highlight_colour
    filtered_df.loc[filtered_df[PARAMETERS] < Mean, 'colours'] = non_highlight_colour
    #print(filtered_df['colours'])
    
    color_palette = {c: c for c in filtered_df['colours'].unique()}
    
    sns.barplot(data=filtered_df,
                x="NEIGHBOURHOOD", 
                y=df_byBrooklyn[PARAMETERS],
                palette=filtered_df['colours'],
               )
    
    ax.spines[['right', 'top']].set_visible(False) 
    
    # label each bar in barplot
    for p in ax.patches:
    
        # get the height of each bar
        height = p.get_height()
        # adding text to each bar
        ax.text(
        x = p.get_x()+(p.get_width()/2),         # x-coordinate position of data label, padded to be in the middle of the bar
        y = height* 1.01,                          # y-coordinate position of data label, padded 100 above bar
        s = '{:.1f}'.format(height),             # data label, formatted to ignore decimals
        ha = 'center',                           # sets horizontal alignment (ha) to center
        color = 'black'
        )
        
    ax.axhline(y= Mean, 
               zorder=0, 
               color='grey', 
               ls='--', 
               lw=2,
              alpha = 0.4)    
    
    
    plt.xlabel('Neibur')
    plt.ylabel('Average Review')
    #plt.title(df_byBrooklyn[NEIGHBOURHOOD_GROUP])
    plt.xticks(rotation=80)
    plt.show()


List_NEIGHBOURHOOD_GROUP = list(df_byBrooklyn['NEIGHBOURHOOD_GROUP'].unique())
PARAMETERS_Values = ['AVG_REVIEW','PRICE']

Neibur_slide=widgets.Dropdown(
    options=List_NEIGHBOURHOOD_GROUP,
    value=List_NEIGHBOURHOOD_GROUP[0],
    description='City:',
    disabled=False,
)
Parameters_slide=widgets.Dropdown(
    options=PARAMETERS_Values,
    value=PARAMETERS_Values[0],
    description='Parameters:',
    disabled=False,
)
# Link slider to update function
widgets.interact(update_plot, 
                 NEIGHBOURHOOD_GROUP=Neibur_slide,
                 PARAMETERS= Parameters_slide)


# In[31]:


df.head(5)


# In[32]:


coord_mean = df.groupby('NEIGHBOURHOOD_GROUP')['LAT','LONG','AVG_REVIEW','PRICE'].mean().reset_index()
coord_mean


# In[33]:


#df['Lat_Mean'] = df['NEIGHBOURHOOD_GROUP'].map(coord_mean['LAT'])
#df['Long_Mean'] = df['NEIGHBOURHOOD_GROUP'].map(coord_mean['LONG'])


# In[36]:


import geopandas as gpd

# Read the shapefile containing world country data
#worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

# Plot the world map
fig, ax = plt.subplots(figsize=(20, 20))
#worldmap.plot(ax=ax,color="lightgrey")

streets = gpd.read_file("C:/Users/Zé/HAVARD/EDA's/Airbnb/new_york_hope.shp")

streets.plot(ax=ax, color='c',alpha=0.2)

# Plot the filtered data
x = coord_mean['LONG']
y = coord_mean['LAT']
z = coord_mean['AVG_REVIEW']
scatter = ax.scatter(x,
           y,
           s = z*100,
           c = z, 
           alpha = 0.9, 
           #cmap='blue'
          )

# label each point on the map
for i, row in coord_mean.iterrows():
    ax.text(x = row['LONG'],
            y = row['LAT'] -0.007,
            s = 'AVG_REVIEW: {:.2f}'.format(row['AVG_REVIEW']),
            ha = 'center', 
            color = 'black'
            #fontsize=12
           )
for i, row in coord_mean.iterrows():
    ax.text(x = row['LONG'],
            y = row['LAT'] -0.011,
            s = 'AVG_PRICE: {:.2f}'.format(row['PRICE']),
            ha = 'center', 
            color = 'black'
            #fontsize=12
           )
    
for i, row in coord_mean.iterrows():
    ax.text(x = row['LONG'],
            y = row['LAT'] + 0.005,
            s = row['NEIGHBOURHOOD_GROUP'],
            ha = 'center', 
            color = 'black'
            #fontsize=12
           )
plt.colorbar(scatter,shrink=0.5)
# Show the map
ax.set_title("Airbnb's in New York")
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.xlim([-74.15, -73.65])
plt.ylim([40.5, 40.95])
#plt.gcf().set_size_inches(20, 10)
plt.show()


# In[37]:


# Read the shapefile containing world country data
#worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

# Plot the world map
fig, ax = plt.subplots(figsize=(20, 20))
#worldmap.plot(ax=ax,color="lightgrey")

streets = gpd.read_file("C:/Users/Zé/HAVARD/EDA's/Airbnb/new_york_hope.shp")

streets.plot(ax=ax, color='c',alpha=0.2)

# Plot the filtered data
x = df['LONG']
y = df['LAT']
z = df['AVG_REVIEW']
ax.scatter(x,
            y,
            s=z*2,
            c=z, 
            alpha=0.9, 
            #cmap='blue'
          )

# Show the map
ax.set_title("Airbnb's in New York")
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
#plt.title("GTA's")
#plt.xlabel('Longitude')
#plt.ylabel('Latitude')
#plt.xlim([-74.15, -73.65])
#plt.ylim([40.5, 40.95])
#plt.gcf().set_size_inches(20, 10)
plt.show()


# In[ ]:




