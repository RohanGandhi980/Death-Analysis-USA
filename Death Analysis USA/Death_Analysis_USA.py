


get_ipython().run_line_magic('pip', 'install --upgrade plotly')


# ## Import Statements

# In[9]:


import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# This might be helpful:
from collections import Counter


# ## Notebook Presentation

# In[12]:


pd.options.display.float_format = '{:,.2f}'.format


# ## Load the Data

# In[15]:


df_hh_income = pd.read_csv('Median_Household_Income_2015.csv', encoding="windows-1252")
df_pct_poverty = pd.read_csv('Pct_People_Below_Poverty_Level.csv', encoding="windows-1252")
df_pct_completed_hs = pd.read_csv('Pct_Over_25_Completed_High_School.csv', encoding="windows-1252")
df_share_race_city = pd.read_csv('Share_of_Race_By_City.csv', encoding="windows-1252")
df_fatalities = pd.read_csv('Deaths_by_Police_US.csv', encoding="windows-1252")


# # Preliminary Data Exploration
# 
# * What is the shape of the DataFrames? 
# * How many rows and columns do they have?
# * What are the column names?
# * Are there any NaN values or duplicates?

# In[17]:


dfs = {
    'df_hh_income': df_hh_income,
    'df_pct_poverty': df_pct_poverty,
    'df_pct_completed_hs': df_pct_completed_hs,
    'df_share_race_city': df_share_race_city,
    'df_fatalities': df_fatalities
}
for df_name, df in dfs.items():
  print(f"Shape of {df_name}: {df.shape}")


# In[27]:


for df_name, df in dfs.items():
    print(f"Columns of {df_name}: {df.columns}")
    print()


# In[29]:


for df_name, df in dfs.items():
  if df.isna().any().any():
    print(f"There are Null values in {df_name}.")
  if df.duplicated().any():
    print(f"There are duplicated values in {df_name}.")


# ## Data Cleaning - Check for Missing Values and Duplicates
# 
# Consider how to deal with the NaN values. Perhaps substituting 0 is appropriate. 

# In[31]:


df_hh_income['Median Income'] = pd.to_numeric(df_hh_income['Median Income'], errors='coerce')
df_hh_income.dropna(inplace=True)


# In[33]:


df_fatalities.isna().sum()


# # Chart the Poverty Rate in each US State
# 
# Create a bar chart that ranks the poverty rate from highest to lowest by US state. Which state has the highest poverty rate? Which state has the lowest poverty rate?  Bar Plot

# In[35]:


df_pct_poverty['poverty_rate'] = pd.to_numeric(df_pct_poverty['poverty_rate'], errors='coerce')
df_pct_poverty.dropna(inplace=True)

poverty_rate = df_pct_poverty.groupby('Geographic Area')['poverty_rate'].mean().sort_values(ascending=False).reset_index()


# In[37]:


plt.figure(figsize=(20, 8),dpi=150)

plt.bar(poverty_rate['Geographic Area'], poverty_rate['poverty_rate'])

plt.xlabel("US State")
plt.ylabel("Poverty Rate")

plt.title("Poverty Rate in each US State")

plt.grid(axis='y', linestyle='--', alpha=0.8)
plt.show()


# # Chart the High School Graduation Rate by US State
# 
# Show the High School Graduation Rate in ascending order of US States. Which state has the lowest high school graduation rate? Which state has the highest?

# In[39]:


df_pct_completed_hs['percent_completed_hs'] = pd.to_numeric(df_pct_completed_hs['percent_completed_hs'], errors='coerce')
df_pct_completed_hs.dropna(inplace=True)
hs_graduation_rate = df_pct_completed_hs.groupby('Geographic Area')['percent_completed_hs'].mean().sort_values().reset_index()


# In[47]:


plt.figure(figsize=(20, 8),dpi=150)

plt.plot(hs_graduation_rate['Geographic Area'], hs_graduation_rate['percent_completed_hs'])

plt.xlabel("US State")
plt.ylabel("High School Graduation Rate")

plt.title("High School Graduation Rate by US State")
plt.grid(axis='y', linestyle='--', alpha=0.8)
plt.show()


# In[51]:


#we can alternatively again use a bar plot to show graduation rate in a percentage form
plt.figure(figsize=(20, 8),dpi=150)

plt.bar(hs_graduation_rate['Geographic Area'], hs_graduation_rate['percent_completed_hs'])

plt.xlabel("US State")
plt.ylabel("High School Graduation Rate")

plt.title("High School Graduation Rate by US State")
plt.grid(axis='y', linestyle='--', alpha=0.8)
plt.show()


# # Visualise the Relationship between Poverty Rates and High School Graduation Rates
# 
# #### Create a line chart with two y-axes to show if the rations of poverty and high school graduation move together.  

# In[53]:


merged_df = poverty_rate.merge(hs_graduation_rate, on='Geographic Area')
merged_df.sort_values('percent_completed_hs', ascending=False, inplace=True)


# In[55]:


plt.figure(figsize=(20,8))
plt.title('Relationship Between Poverty Rates and High School Graduation Rates')

ax = plt.gca()
ax2 = ax.twinx()

ax.plot(merged_df['Geographic Area'], merged_df['poverty_rate'], label='Poverty Rate', linestyle='-', marker='o', markersize=8, linewidth=2)

ax2.plot(merged_df['Geographic Area'], merged_df['percent_completed_hs'], color='crimson', label='High School Graduation Rate', linestyle='--', marker='x', markersize=8, linewidth=2)

ax.set_ylabel('Poverty Rate')
ax.set_xlabel('State')
ax2.set_ylabel('High School Graduation Rate')

ax2.invert_yaxis()
plt.legend(loc='upper center')
ax.yaxis.grid(False)
ax2.grid(False)
plt.show()


# #### Now use a Seaborn .jointplot() with a Kernel Density Estimate (KDE) and/or scatter plot to visualise the same relationship

# In[ ]:





# In[ ]:





# #### Seaborn's `.lmplot()` or `.regplot()` to show a linear regression between the poverty ratio and the high school graduation ratio. 

# In[ ]:





# # Create a Bar Chart with Subsections Showing the Racial Makeup of Each US State
# 
# Visualise the share of the white, black, hispanic, asian and native american population in each US State using a bar chart with sub sections. 

# In[57]:


cols = ['share_white', 'share_black','share_native_american', 'share_asian', 'share_hispanic']
df_share_race_city[cols] = df_share_race_city[cols].apply(pd.to_numeric, errors='coerce')
df_share_race_city.dropna(inplace=True)
racial_makeup = df_share_race_city.groupby('Geographic area')[cols].mean().reset_index()
racial_makeup.rename(columns={'share_white': 'White', 'share_black': 'Black', 'share_native_american': 'Native American', 'share_asian': 'Asian', 'share_hispanic': 'Hispanic'}, inplace=True)


# In[67]:


fig = px.bar(racial_makeup,
    x="Geographic area",
    y=['White', 'Black', 'Native American', 'Asian', 'Hispanic'],
    title="Racial Makeup of Each US State",
    labels={"value": "Racial Makeup (%)", 'variable': 'Race', 'Geographic area': 'State'},
    barmode="stack",
)
fig.update_xaxes(tickangle=0)

fig.show()


# # Create Donut Chart by of People Killed by Race
# 
# Hint: Use `.value_counts()`

# In[69]:


deaths_by_race = df_fatalities[df_fatalities['race'].notna()]
deaths_by_race = deaths_by_race['race'].value_counts()


# In[71]:


fig = px.pie(names=deaths_by_race.index,
             values=deaths_by_race.values,
             title="Deaths by Race",
             hole=0.4,)

fig.update_traces(textfont_size=15, labels=['White', 'Black', 'Native American', 'Asian', 'Hispanic'])

fig.show()


# # Create a Chart Comparing the Total Number of Deaths of Men and Women
# 
# Use `df_fatalities` to illustrate how many more men are killed compared to women. 

# In[73]:


deaths_by_gender = df_fatalities['gender'].value_counts()


# In[75]:


fig = px.bar(deaths_by_gender, x=deaths_by_gender.index,
             y=deaths_by_gender.values,
             color=deaths_by_gender.index,
             title="Deaths by Gender")
fig.update_xaxes(title_text='Gender', tickvals=[0, 1], ticktext=['MEN', 'WOMEN'])
fig.update_yaxes(title_text='Death Count')

fig.show()


# # Create a Box Plot Showing the Age and Manner of Death
# 
# Break out the data by gender using `df_fatalities`. Is there a difference between men and women in the manner of death? 

# In[77]:


age_x_manner_of_death = df_fatalities[df_fatalities['age'].notna()]
age_x_manner_of_death = age_x_manner_of_death.groupby('gender')[['age', 'manner_of_death']].value_counts().reset_index(name='count')
age_x_manner_of_death['gender'].replace({'M': 'Man', 'F': 'Woman'}, inplace=True)


# In[79]:


fig = px.box(age_x_manner_of_death, x='manner_of_death', y='age', color='gender', color_discrete_map={'Woman': 'red', 'Man': 'blue'})
fig.update_layout(
    title="Manner of Death by Gender and Age",
    xaxis_title="Manner of Death",
    yaxis_title="Age",
    legend_title="Gender",
)
fig.show()


# In[81]:


people_armed = df_fatalities[df_fatalities['armed'].notna()]
unarmed_percentage = ((people_armed['armed'] == 'unarmed').sum() / people_armed['armed'].value_counts().sum()) * 100
people_armed = people_armed['armed'].value_counts()
print(f"Armed people killed: {round(100 - unarmed_percentage)}%")


# # Were People Armed? 
# 
# In what percentage of police killings were people armed? Create chart that show what kind of weapon (if any) the deceased was carrying. How many of the people killed by police were armed with guns versus unarmed? 

# In[83]:


fig = px.bar(people_armed, x=people_armed.index, y=people_armed.values, color=people_armed.values)

fig.update_layout(
    title="Type of Weapon Carried",
    xaxis_title="Weapon",
    yaxis_title="Count",
)

fig.show()


# In[85]:


gun = people_armed['gun']
unarmed = people_armed['unarmed']

print(f'People armed with guns killed by the police: {gun}')
print(f'People unarmed killed by the police: {unarmed}')


# In[87]:


df_fatalities['race'].replace({'W': 'White', 'B': 'Black', 'N': 'Native American', 'A': 'Asian', 'H': 'Hispanic', 'O': 'Other'}, inplace=True)


# # How Old Were the People Killed?

# Work out what percentage of people killed were under 25 years old.  

# In[89]:


killed_by_age = df_fatalities[df_fatalities['age'].notna()]
under_25 = killed_by_age[killed_by_age['age'] < 25].value_counts().sum()
total = killed_by_age.value_counts().sum()
percentage = (under_25/total) * 100
print(f'People killed under 25 years old: {round(percentage)}%')


# Create a histogram and KDE plot that shows the distribution of ages of the people killed by police. 

# In[93]:


plt.figure(figsize=(10, 6), dpi = 150)

sns.histplot(data=killed_by_age, x='age', kde=True, color='red', alpha=0.1)

plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Kernel Density Estimation of Age')
plt.xticks(range(0, 101, 10))

plt.show()


# Create a seperate KDE plot for each race. Is there a difference between the distributions? 

# In[95]:


g = sns.FacetGrid(killed_by_age, col="race")
g.map(sns.histplot, 'age', kde=True, color='red', alpha=0.1)

g.set_axis_labels('Age', 'Count')
g.set_titles('KDE of Age - Race: {col_name}')

plt.show()


# # Race of People Killed
# 
# Create a chart that shows the total number of people killed by race. 

# In[97]:


killed_by_race = df_fatalities[df_fatalities['race'].notna()]
killed_by_race = killed_by_race['race'].value_counts()


# In[101]:


plt.figure(figsize=(20, 8),dpi=150)
plt.plot(killed_by_race, marker='o', markersize=8, linewidth=2)

plt.xlabel("Race")
plt.ylabel("Death Count")

plt.title("Total Number of People Killed by Race")
plt.grid(True)
plt.yticks(range(0, killed_by_race.max(), 100))
plt.show()


# # Mental Illness and Police Killings
# 
# What percentage of people killed by police have been diagnosed with a mental illness?

# In[ ]:





# In[ ]:





# # In Which Cities Do the Most Police Killings Take Place?
# 
# Create a chart ranking the top 10 cities with the most police killings. Which cities are the most dangerous?  

# In[107]:


top15_cities = df_fatalities[['state', 'city']].value_counts().head(15).reset_index(name='count')


# In[109]:


fig = px.bar(top15_cities, x='city', y='count')

fig.update_layout(
    title="Top 15 Cities by Death Count",
    xaxis_title="City",
    yaxis_title="Death Count",
    showlegend=False
)

fig.show()


# # Rate of Death by Race
# 
# Find the share of each race in the top 10 cities. Contrast this with the top 10 cities of police killings to work out the rate at which people are killed by race for each city. 

# In[111]:


merged_df = top10_cities.merge(df_fatalities, on=['state', 'city'])
merged_df = merged_df.groupby(['state', 'city', 'count'])['race'].value_counts(dropna=False).reset_index(name='death_race')
merged_df['death_race'] = round((merged_df['death_race'] / merged_df['count']) * 100)
merged_df = merged_df[merged_df['race'].isin(["Asian", "Black", "Hispanic", "Native American", "White"])]


# In[113]:


cities = '|'.join(top10_cities['city'].tolist())
top10_cities_race = df_share_race_city[df_share_race_city['City'].str.contains(cities, case=False)]
for city in top10_cities['city']:
    city_variation = top10_cities_race['City'].str.contains(city, case=False)
    top10_cities_race.loc[city_variation, 'City'] = city
top10_cities_race = top10_cities_race.groupby(['Geographic area', 'City']).mean().reset_index()
top10_cities_race = top10_cities_race.merge(top10_cities, left_on=['Geographic area', 'City'], right_on=['state', 'city'])
top10_cities_race.drop(['Geographic area', 'City', 'count'], axis=1, inplace=True)
melted_df = top10_cities_race.melt(id_vars=["state", "city"], var_name="race", value_name="race_share")
race_mapping = {
    "share_white": "White",
    "share_black": "Black",
    "share_native_american": "Native American",
    "share_asian": "Asian",
    "share_hispanic": "Hispanic"
}
melted_df['race'] = melted_df['race'].replace(race_mapping)


# In[115]:


for city in top10_cities['city']:

    fig, ax = plt.subplots()

    sns.barplot(data=melted_df[melted_df['city'] == city], x='race', y='race_share', ax=ax, color='red', alpha=0.5)
    ax2 = ax.twinx()
    sns.lineplot(data=merged_df[merged_df['city'] == city], x='race', y='death_race', ax=ax2, color='black', alpha=0.7, marker='o', linewidth=3, label='Death Race')
    ax2.grid(None)

    ax.set_xlabel("Race")
    ax.set_ylabel("Race Share (%)")
    ax2.set_ylabel("Death by Race (%)")

    plt.legend(loc='upper center')
    plt.title(f'City: {city}')
    plt.show()


# # Create a Choropleth Map of Police Killings by US State
# 
# Which states are the most dangerous? Compare your map with your previous chart. Are these the same states with high degrees of poverty? 

# In[ ]:





# In[ ]:





# # Number of Police Killings Over Time
# 
# Analyse the Number of Police Killings over Time. Is there a trend in the data? 

# In[117]:


df_fatalities['date'] = pd.to_datetime(df_fatalities['date'])


# In[121]:


deaths_over_time = df_fatalities.groupby('date').size().reset_index(name='count').sort_values('date')
deaths_over_time['year'] = deaths_over_time['date'].dt.year
deaths_over_time['month'] = deaths_over_time['date'].dt.month
deaths_over_time = deaths_over_time.groupby(['year', 'month'])['count'].sum().reset_index()


# In[125]:


plt.figure(figsize=(20, 6), dpi=200)

plt.plot(deaths_over_time.index, deaths_over_time['count'], marker='o', linestyle='-')
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Number of Police Killings Over Time')

plt.xticks(deaths_over_time.index, deaths_over_time.year, rotation=45)

plt.show()


# In[ ]:





# # Epilogue
# 
# Now that you have analysed the data yourself, read [The Washington Post's analysis here](https://www.washingtonpost.com/graphics/investigations/police-shootings-database/).

# In[ ]:




