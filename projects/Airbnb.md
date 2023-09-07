```python
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import widgets, interactive
```


```python
df = pd.read_csv('Airbnb_Open_Data.csv',low_memory=False)
```


```python
df.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>NAME</th>
      <th>host id</th>
      <th>host_identity_verified</th>
      <th>host name</th>
      <th>neighbourhood group</th>
      <th>neighbourhood</th>
      <th>lat</th>
      <th>long</th>
      <th>country</th>
      <th>...</th>
      <th>service fee</th>
      <th>minimum nights</th>
      <th>number of reviews</th>
      <th>last review</th>
      <th>reviews per month</th>
      <th>review rate number</th>
      <th>calculated host listings count</th>
      <th>availability 365</th>
      <th>house_rules</th>
      <th>license</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001254</td>
      <td>Clean &amp; quiet apt home by the park</td>
      <td>80014485718</td>
      <td>unconfirmed</td>
      <td>Madaline</td>
      <td>Brooklyn</td>
      <td>Kensington</td>
      <td>40.64749</td>
      <td>-73.97237</td>
      <td>United States</td>
      <td>...</td>
      <td>$193</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>10/19/2021</td>
      <td>0.21</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>286.0</td>
      <td>Clean up and treat the home the way you'd like...</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 26 columns</p>
</div>




```python
df.dtypes
```




    id                                  int64
    NAME                               object
    host id                             int64
    host_identity_verified             object
    host name                          object
    neighbourhood group                object
    neighbourhood                      object
    lat                               float64
    long                              float64
    country                            object
    country code                       object
    instant_bookable                   object
    cancellation_policy                object
    room type                          object
    Construction year                 float64
    price                              object
    service fee                        object
    minimum nights                    float64
    number of reviews                 float64
    last review                        object
    reviews per month                 float64
    review rate number                float64
    calculated host listings count    float64
    availability 365                  float64
    house_rules                        object
    license                            object
    dtype: object




```python
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
```


```python
df.rename(str.upper, axis='columns',inplace=True)
```


```python
df['NEIGHBOURHOOD_GROUP'].unique()
```




    array(['Brooklyn', 'Manhattan', 'brookln', 'manhatan', 'Queens', nan,
           'Staten Island', 'Bronx'], dtype=object)




```python
df.replace({'brookln': 'Brooklyn', 'manhatan': 'Manhattan'}, inplace=True)
```


```python
df.isnull().sum()
```




    ID                                     0
    NAME                                 250
    HOST_ID                                0
    HOST_IDENTITY_VERIFIED               289
    HOST_NAME                            406
    NEIGHBOURHOOD_GROUP                   29
    NEIGHBOURHOOD                         16
    LAT                                    8
    LONG                                   8
    COUNTRY                              532
    COUNTRY CODE                         131
    INSTANT_BOOKABLE                     105
    CANCELLATION_POLICY                   76
    ROOM TYPE                              0
    CONSTRUCTION YEAR                    214
    PRICE                                247
    SERVICE_FEE                          273
    MIN_NIGHTS                           409
    NUMBER_OF_REVIEWS                    183
    LAST_REVIEW                        15893
    REVIEWS_PER_MONTH                  15879
    AVG_REVIEW                           326
    CALCULATED_HOST_LISTINGS_COUNT       319
    AVAILABILITY_365                     448
    HOUSE_RULES                        52131
    LICENSE                           102597
    dtype: int64




```python
df.nunique()
```




    ID                                102058
    NAME                               61281
    HOST_ID                           102057
    HOST_IDENTITY_VERIFIED                 2
    HOST_NAME                          13190
    NEIGHBOURHOOD_GROUP                    5
    NEIGHBOURHOOD                        224
    LAT                                21991
    LONG                               17774
    COUNTRY                                1
    COUNTRY CODE                           1
    INSTANT_BOOKABLE                       2
    CANCELLATION_POLICY                    3
    ROOM TYPE                              4
    CONSTRUCTION YEAR                     20
    PRICE                               1151
    SERVICE_FEE                          231
    MIN_NIGHTS                           153
    NUMBER_OF_REVIEWS                    476
    LAST_REVIEW                         2477
    REVIEWS_PER_MONTH                   1016
    AVG_REVIEW                             5
    CALCULATED_HOST_LISTINGS_COUNT        78
    AVAILABILITY_365                     438
    HOUSE_RULES                         1976
    LICENSE                                1
    dtype: int64




```python
df.LICENSE.unique()
```




    array([nan, '41662/AL'], dtype=object)




```python
df.shape
```




    (102599, 26)




```python
df.drop_duplicates(inplace=True)
```


```python
df.drop(['LICENSE','COUNTRY','COUNTRY CODE'],inplace=True,axis=1)
```


```python
df['duplicado']= df.duplicated(subset=['HOST_ID'],keep=False)
```


```python
df_2= df[df['duplicado']==True]
```


```python
df_2.value_counts('HOST_ID')
```




    HOST_ID
    29531702698    2
    dtype: int64




```python
df[df['HOST_ID']==29531702698]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>NAME</th>
      <th>HOST_ID</th>
      <th>HOST_IDENTITY_VERIFIED</th>
      <th>HOST_NAME</th>
      <th>NEIGHBOURHOOD_GROUP</th>
      <th>NEIGHBOURHOOD</th>
      <th>LAT</th>
      <th>LONG</th>
      <th>INSTANT_BOOKABLE</th>
      <th>...</th>
      <th>SERVICE_FEE</th>
      <th>MIN_NIGHTS</th>
      <th>NUMBER_OF_REVIEWS</th>
      <th>LAST_REVIEW</th>
      <th>REVIEWS_PER_MONTH</th>
      <th>AVG_REVIEW</th>
      <th>CALCULATED_HOST_LISTINGS_COUNT</th>
      <th>AVAILABILITY_365</th>
      <th>HOUSE_RULES</th>
      <th>duplicado</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23397</th>
      <td>13923499</td>
      <td>Magnificent Lakeview Home on Kissena Park in NYC</td>
      <td>29531702698</td>
      <td>verified</td>
      <td>Ann</td>
      <td>Queens</td>
      <td>Flushing</td>
      <td>40.74982</td>
      <td>-73.80610</td>
      <td>True</td>
      <td>...</td>
      <td>$208</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>256.0</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>42960</th>
      <td>24728144</td>
      <td>NaN</td>
      <td>29531702698</td>
      <td>verified</td>
      <td>Seth</td>
      <td>Brooklyn</td>
      <td>Williamsburg</td>
      <td>40.71867</td>
      <td>-73.96163</td>
      <td>True</td>
      <td>...</td>
      <td>$228</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>5/19/2019</td>
      <td>0.59</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>44.0</td>
      <td>This is a non smoking apartment. No l</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 24 columns</p>
</div>




```python
df.dropna(subset=['AVG_REVIEW','NUMBER_OF_REVIEWS','LAT','LONG'],inplace=True)

df.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>NAME</th>
      <th>HOST_ID</th>
      <th>HOST_IDENTITY_VERIFIED</th>
      <th>HOST_NAME</th>
      <th>NEIGHBOURHOOD_GROUP</th>
      <th>NEIGHBOURHOOD</th>
      <th>LAT</th>
      <th>LONG</th>
      <th>INSTANT_BOOKABLE</th>
      <th>...</th>
      <th>SERVICE_FEE</th>
      <th>MIN_NIGHTS</th>
      <th>NUMBER_OF_REVIEWS</th>
      <th>LAST_REVIEW</th>
      <th>REVIEWS_PER_MONTH</th>
      <th>AVG_REVIEW</th>
      <th>CALCULATED_HOST_LISTINGS_COUNT</th>
      <th>AVAILABILITY_365</th>
      <th>HOUSE_RULES</th>
      <th>duplicado</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001254</td>
      <td>Clean &amp; quiet apt home by the park</td>
      <td>80014485718</td>
      <td>unconfirmed</td>
      <td>Madaline</td>
      <td>Brooklyn</td>
      <td>Kensington</td>
      <td>40.64749</td>
      <td>-73.97237</td>
      <td>False</td>
      <td>...</td>
      <td>$193</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>10/19/2021</td>
      <td>0.21</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>286.0</td>
      <td>Clean up and treat the home the way you'd like...</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 24 columns</p>
</div>




```python
df.drop(['duplicado'],inplace=True,axis=1)
```


```python
df.isnull().sum()
```




    ID                                    0
    NAME                                242
    HOST_ID                               0
    HOST_IDENTITY_VERIFIED              275
    HOST_NAME                           397
    NEIGHBOURHOOD_GROUP                  27
    NEIGHBOURHOOD                        16
    LAT                                   0
    LONG                                  0
    INSTANT_BOOKABLE                     91
    CANCELLATION_POLICY                  76
    ROOM TYPE                             0
    CONSTRUCTION YEAR                   204
    PRICE                               247
    SERVICE_FEE                         273
    MIN_NIGHTS                          379
    NUMBER_OF_REVIEWS                     0
    LAST_REVIEW                       15669
    REVIEWS_PER_MONTH                 15656
    AVG_REVIEW                            0
    CALCULATED_HOST_LISTINGS_COUNT      251
    AVAILABILITY_365                    380
    HOUSE_RULES                       51570
    dtype: int64




```python
df.PRICE
```




    0           $966 
    1           $142 
    2           $620 
    3           $368 
    4           $204 
               ...   
    102053      $696 
    102054      $909 
    102055      $387 
    102056      $848 
    102057    $1,128 
    Name: PRICE, Length: 101549, dtype: object




```python
price_review=df.groupby(['PRICE'])['AVG_REVIEW'].agg('mean').reset_index()
price_review['PRICE'] = price_review['PRICE'].str.replace(',', '').str.replace('$', '').astype(int)
price_review.sort_values(by='PRICE',
                         inplace=True,
                         ignore_index=True)
price_review
```

    C:\Users\Zé\AppData\Local\Temp\ipykernel_7120\3265200479.py:2: FutureWarning: The default value of regex will change from True to False in a future version. In addition, single character regular expressions will *not* be treated as literal strings when regex=True.
      price_review['PRICE'] = price_review['PRICE'].str.replace(',', '').str.replace('$', '').astype(int)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PRICE</th>
      <th>AVG_REVIEW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50</td>
      <td>3.204301</td>
    </tr>
    <tr>
      <th>1</th>
      <td>51</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>52</td>
      <td>3.285714</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>3.384615</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54</td>
      <td>3.171875</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1146</th>
      <td>1196</td>
      <td>3.179487</td>
    </tr>
    <tr>
      <th>1147</th>
      <td>1197</td>
      <td>3.253165</td>
    </tr>
    <tr>
      <th>1148</th>
      <td>1198</td>
      <td>3.272727</td>
    </tr>
    <tr>
      <th>1149</th>
      <td>1199</td>
      <td>3.039216</td>
    </tr>
    <tr>
      <th>1150</th>
      <td>1200</td>
      <td>3.014706</td>
    </tr>
  </tbody>
</table>
<p>1151 rows × 2 columns</p>
</div>




```python
plt.figure(figsize=(20,6))
sns.lineplot(data=price_review, 
                x="PRICE",
                y="AVG_REVIEW"
            )
```




    <Axes: xlabel='PRICE', ylabel='AVG_REVIEW'>




    
![png](output_23_1.png)
    



```python
plt.figure(figsize=(20,6))
df['PRICE'] = df['PRICE'].str.replace(',', '').str.replace('$', '').astype(float)
sns.barplot(data=df,
       x='NEIGHBOURHOOD_GROUP',
       y='PRICE', 
       hue='AVG_REVIEW'
       )
plt.show()
```

    C:\Users\Zé\AppData\Local\Temp\ipykernel_7120\1962300077.py:2: FutureWarning: The default value of regex will change from True to False in a future version. In addition, single character regular expressions will *not* be treated as literal strings when regex=True.
      df['PRICE'] = df['PRICE'].str.replace(',', '').str.replace('$', '').astype(float)
    


    
![png](output_24_1.png)
    



```python
plt.figure(figsize=(20,6))
sns.boxplot(data=df,
       x='NEIGHBOURHOOD_GROUP',
       y='AVG_REVIEW', 
       )
plt.show()
```


    
![png](output_25_0.png)
    



```python
df.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>NAME</th>
      <th>HOST_ID</th>
      <th>HOST_IDENTITY_VERIFIED</th>
      <th>HOST_NAME</th>
      <th>NEIGHBOURHOOD_GROUP</th>
      <th>NEIGHBOURHOOD</th>
      <th>LAT</th>
      <th>LONG</th>
      <th>INSTANT_BOOKABLE</th>
      <th>...</th>
      <th>PRICE</th>
      <th>SERVICE_FEE</th>
      <th>MIN_NIGHTS</th>
      <th>NUMBER_OF_REVIEWS</th>
      <th>LAST_REVIEW</th>
      <th>REVIEWS_PER_MONTH</th>
      <th>AVG_REVIEW</th>
      <th>CALCULATED_HOST_LISTINGS_COUNT</th>
      <th>AVAILABILITY_365</th>
      <th>HOUSE_RULES</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001254</td>
      <td>Clean &amp; quiet apt home by the park</td>
      <td>80014485718</td>
      <td>unconfirmed</td>
      <td>Madaline</td>
      <td>Brooklyn</td>
      <td>Kensington</td>
      <td>40.64749</td>
      <td>-73.97237</td>
      <td>False</td>
      <td>...</td>
      <td>966.0</td>
      <td>$193</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>10/19/2021</td>
      <td>0.21</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>286.0</td>
      <td>Clean up and treat the home the way you'd like...</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 23 columns</p>
</div>




```python
#df_Brooklyn=df[df['NEIGHBOURHOOD_GROUP'] == 'Brooklyn']
df_byBrooklyn=df.groupby(['NEIGHBOURHOOD_GROUP','NEIGHBOURHOOD'])['AVG_REVIEW','PRICE'].agg('mean').reset_index()
df_byBrooklyn.head(1)
```

    C:\Users\Zé\AppData\Local\Temp\ipykernel_7120\3643360599.py:2: FutureWarning: Indexing with multiple keys (implicitly converted to a tuple of keys) will be deprecated, use a list instead.
      df_byBrooklyn=df.groupby(['NEIGHBOURHOOD_GROUP','NEIGHBOURHOOD'])['AVG_REVIEW','PRICE'].agg('mean').reset_index()
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NEIGHBOURHOOD_GROUP</th>
      <th>NEIGHBOURHOOD</th>
      <th>AVG_REVIEW</th>
      <th>PRICE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bronx</td>
      <td>Allerton</td>
      <td>3.294737</td>
      <td>641.968085</td>
    </tr>
  </tbody>
</table>
</div>




```python
from IPython.display import display
```


```python
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
```


    interactive(children=(Dropdown(description='City:', options=('Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Stat…





    <function __main__.update_plot(NEIGHBOURHOOD_GROUP, PARAMETERS)>




```python
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>NAME</th>
      <th>HOST_ID</th>
      <th>HOST_IDENTITY_VERIFIED</th>
      <th>HOST_NAME</th>
      <th>NEIGHBOURHOOD_GROUP</th>
      <th>NEIGHBOURHOOD</th>
      <th>LAT</th>
      <th>LONG</th>
      <th>INSTANT_BOOKABLE</th>
      <th>...</th>
      <th>PRICE</th>
      <th>SERVICE_FEE</th>
      <th>MIN_NIGHTS</th>
      <th>NUMBER_OF_REVIEWS</th>
      <th>LAST_REVIEW</th>
      <th>REVIEWS_PER_MONTH</th>
      <th>AVG_REVIEW</th>
      <th>CALCULATED_HOST_LISTINGS_COUNT</th>
      <th>AVAILABILITY_365</th>
      <th>HOUSE_RULES</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001254</td>
      <td>Clean &amp; quiet apt home by the park</td>
      <td>80014485718</td>
      <td>unconfirmed</td>
      <td>Madaline</td>
      <td>Brooklyn</td>
      <td>Kensington</td>
      <td>40.64749</td>
      <td>-73.97237</td>
      <td>False</td>
      <td>...</td>
      <td>966.0</td>
      <td>$193</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>10/19/2021</td>
      <td>0.21</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>286.0</td>
      <td>Clean up and treat the home the way you'd like...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002102</td>
      <td>Skylit Midtown Castle</td>
      <td>52335172823</td>
      <td>verified</td>
      <td>Jenna</td>
      <td>Manhattan</td>
      <td>Midtown</td>
      <td>40.75362</td>
      <td>-73.98377</td>
      <td>False</td>
      <td>...</td>
      <td>142.0</td>
      <td>$28</td>
      <td>30.0</td>
      <td>45.0</td>
      <td>5/21/2022</td>
      <td>0.38</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>228.0</td>
      <td>Pet friendly but please confirm with me if the...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1002403</td>
      <td>THE VILLAGE OF HARLEM....NEW YORK !</td>
      <td>78829239556</td>
      <td>NaN</td>
      <td>Elise</td>
      <td>Manhattan</td>
      <td>Harlem</td>
      <td>40.80902</td>
      <td>-73.94190</td>
      <td>True</td>
      <td>...</td>
      <td>620.0</td>
      <td>$124</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>352.0</td>
      <td>I encourage you to use my kitchen, cooking and...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1002755</td>
      <td>NaN</td>
      <td>85098326012</td>
      <td>unconfirmed</td>
      <td>Garry</td>
      <td>Brooklyn</td>
      <td>Clinton Hill</td>
      <td>40.68514</td>
      <td>-73.95976</td>
      <td>True</td>
      <td>...</td>
      <td>368.0</td>
      <td>$74</td>
      <td>30.0</td>
      <td>270.0</td>
      <td>7/5/2019</td>
      <td>4.64</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>322.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1003689</td>
      <td>Entire Apt: Spacious Studio/Loft by central park</td>
      <td>92037596077</td>
      <td>verified</td>
      <td>Lyndon</td>
      <td>Manhattan</td>
      <td>East Harlem</td>
      <td>40.79851</td>
      <td>-73.94399</td>
      <td>False</td>
      <td>...</td>
      <td>204.0</td>
      <td>$41</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>11/19/2018</td>
      <td>0.10</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>289.0</td>
      <td>Please no smoking in the house, porch or on th...</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
coord_mean = df.groupby('NEIGHBOURHOOD_GROUP')['LAT','LONG','AVG_REVIEW','PRICE'].mean().reset_index()
coord_mean
```

    C:\Users\Zé\AppData\Local\Temp\ipykernel_7120\1527993752.py:1: FutureWarning: Indexing with multiple keys (implicitly converted to a tuple of keys) will be deprecated, use a list instead.
      coord_mean = df.groupby('NEIGHBOURHOOD_GROUP')['LAT','LONG','AVG_REVIEW','PRICE'].mean().reset_index()
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NEIGHBOURHOOD_GROUP</th>
      <th>LAT</th>
      <th>LONG</th>
      <th>AVG_REVIEW</th>
      <th>PRICE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bronx</td>
      <td>40.849226</td>
      <td>-73.883254</td>
      <td>3.331341</td>
      <td>629.961423</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brooklyn</td>
      <td>40.683814</td>
      <td>-73.950489</td>
      <td>3.258506</td>
      <td>626.644103</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Manhattan</td>
      <td>40.765283</td>
      <td>-73.974419</td>
      <td>3.276897</td>
      <td>622.752608</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Queens</td>
      <td>40.728570</td>
      <td>-73.867656</td>
      <td>3.329217</td>
      <td>630.228030</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Staten Island</td>
      <td>40.611614</td>
      <td>-74.105278</td>
      <td>3.404030</td>
      <td>623.167021</td>
    </tr>
  </tbody>
</table>
</div>




```python
#df['Lat_Mean'] = df['NEIGHBOURHOOD_GROUP'].map(coord_mean['LAT'])
#df['Long_Mean'] = df['NEIGHBOURHOOD_GROUP'].map(coord_mean['LONG'])
```


```python
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
```


    
![png](output_33_0.png)
    



```python
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
```


    
![png](output_34_0.png)
    



```python

```
