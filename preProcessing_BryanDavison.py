#!/usr/bin/env python
# coding: utf-8

# In[19]:


# Bryan Davison #

import pandas as pd
import numpy as np
from scipy.stats import normaltest
from sklearn import preprocessing


# # Cleaning up World Development Indicators Data

# In[3]:


WDIData = pd.read_csv('WDIData.csv', na_values=["nan"])

def preprocessWDI(dataframe):
    
    # removing years 1960-1989 from the WDI Dataset so both datasets start at 1990 so the 
    # timeline of this dataset matches the timeline of the Human Dev. Index dataset
    newDf = dataframe.drop(columns=["{0}".format(i) for i in range(1960,1990)]).copy()
    
    # Set index by Country Code for easier use
    newDf.set_index(['Country Code'], inplace=True)
    
    # Selecting rows from Afghanistan to the end of dataframe
    newDf = newDf.loc["AFG":]
    
    # Temporarily resetting index 
    newDf.reset_index(level=0,inplace=True)
    
    # Fixing Country Names by removing unecessary strings in Country Names
    newDf["Country Name"] = newDf["Country Name"].str.replace(r"\(.*\)|,.*", "")
    
    newDf.set_index(['Country Code'], inplace=True)
    return newDf


# Fucntion to created a dataframe filtered by the specified indicator
def minimizeData(dataframe,indicator_code):
    newDf = dataframe.loc[dataframe["Indicator Code"].isin([indicator_code])].copy()
    newDf.drop(["Indicator Name"],axis=1,inplace=True)
    return newDf

def fillNanZero(dataframe):
    dataframe.fillna(0,inplace=True)

# Function to prepare more complex dataframes for plotting by reucing the dataframe 
# to countries as columns and indicator observations as rows
def prepareForAnalysis(dataframe, indicator_code):
    newDf = dataframe.copy()
    newDf = newDf[newDf.index.get_level_values("Indicator Code")==indicator_code]
    newDf.reset_index(["Country Code","Indicator Code"], drop=True, inplace=True)
    transposed = newDf.transpose()[:-2]  
    return transposed

# Function to fix dataframes where the correct header is actually the first row of the dataframe
def fixHeader(dataframe):
    newDf = dataframe.copy()
    if(dataframe.columns.isin(["Burundi"]).any()):
        # Special condition for the bottom countries. This fixes the column names of their dataframes
        dataframe.set_axis(["Burundi", "Burkina Faso","Central African Republic","Liberia","Mali","Mozambique",
                            "Niger","Sierra Leone","South Sudan","Chad"], axis=1, inplace=True)
    else:
        newDf.rename(columns=dataframe.iloc[0], inplace=True)
        newDf = newDf.iloc[1:].astype('float64')
    return newDf


WDIData = preprocessWDI(WDIData)



# Temporarily resetting index 
WDIData.reset_index(level=0,inplace=True)


# Creating a dataframe of all country codes and their names for later use
country_codes = WDIData[["Country Code","Country Name"]].drop_duplicates(keep='first').astype(str)


# Fixing Hong Kong row
country_codes.loc[country_codes["Country Code"] == "HKG", "Country Name"] = "Hong Kong"


# Setting the index back to Country Code
WDIData.set_index(['Country Code'], inplace=True)

WDIData.head()


# # Extracting data from WDI based on one indicator

# #### Indicator: Domestic general government health expenditure per capita (current USD)

# In[4]:


govHealthCode = "SH.XPD.GHED.PC.CD"

# Creates copy of the WDI that only includes country data 
# on 'Domestic general government health expenditure per capita'
govHealthExp = minimizeData(WDIData, govHealthCode)

# Checking the data within this new dataframe
govHealthExp.info()

# since columns 1990 through 1999 and 2017-2018 are all NaNs I will drop them
govHealthExp.drop(columns=["{0}".format(i) for i in range(1990,2000)], inplace=True)
govHealthExp.drop(columns=["2017","2018"], inplace=True)


# Filling nan values with 0
fillNanZero(govHealthExp)


# #### Indicator: Domestic private health expenditure per capita (current USD)

# In[5]:


privHealthCode = "SH.XPD.PVTD.PC.CD"

# Creates copy of the WDI that only includes country data on ' Domestic private health expenditure per capita'
privHealthExp = minimizeData(WDIData, privHealthCode)


# Checking the data within this new dataframe
privHealthExp.info()


# Since columns 1990 through 1999 and 2017-2018 are all NaNs I will drop them
privHealthExp.drop(columns=["{0}".format(i) for i in range(1990,2000)], inplace=True)
privHealthExp.drop(columns=["2017","2018"], inplace=True)


# Filling nan with 0
fillNanZero(privHealthExp)


# # Cleaning up Human Devopment Index Data

# In[6]:


HDIndex = pd.read_csv('Human development index (HDI).csv', na_values=[".."], skiprows=1, encoding='latin-1')


# Saving all year columns to a list
cols = list(HDIndex.columns[range(2,30)])

# Replace nans with the median of the column in which they reside
#for i in cols:
#    HDIndex[i].replace(np.NaN,0,inplace=True)#HDIndex[i].median(), inplace=True)


# Converting 'HDIndex Rank (2017)' from a string to a number
HDIndex['HDI Rank (2017)'] = pd.to_numeric(HDIndex['HDI Rank (2017)'], errors='coerce')


# Sorting by ascending HDIndex for 2017
HDIndex.sort_values(by='Country', inplace=True)
HDIndex.rename(columns={'Country': 'Country Name'}, inplace=True)


# Adding Indicator Code column
c = {"Indicator Code" : "HDI"}
HDIndex = HDIndex.assign(**c)


# Merging this Human Dev. Index dataframe with the country codes dataframe to assign country codes to the correct countries
HDIndex = pd.merge(HDIndex, country_codes,  how='left', left_on=['Country Name'], right_on = ['Country Name'])


# Removing years 1990-1999 from dataframe
HDIndex.drop(columns = ["{0}".format(i) for i in range(1990,2000)], inplace=True)
HDIndex.drop(columns = ["2017"], inplace=True)


# Removing any unecessary country name strings 
HDIndex["Country Name"] = HDIndex["Country Name"].str.replace(r"\(.*\)|,.*", "")

# Fixing Sweden row
HDIndex.loc[(HDIndex["Country Name"]=="Sweden"),"HDI Rank (2017)"] = 8.000

# Only include non-null values
HDIndex = HDIndex[pd.notnull(HDIndex["HDI Rank (2017)"])]

# Fix Country code for Hong Kong
HDIndex.loc[(HDIndex["Country Name"]=="Hong Kong"),"Country Code"] = "HKG"

# Making Country Code the index
HDIndex.set_index("Country Code", inplace=True)


# ## Merging all Data needed for Question 8

# In[7]:


# merging all 3 dataframes into one
num8Data = pd.concat([privHealthExp, govHealthExp, HDIndex], sort=False)


# set index to Indicator code
num8Data.set_index("Indicator Code", append=True, inplace=True)


# sorting by index for format purposes
num8Data.sort_index(0, level=[0,1], inplace=True)



# filling nan values for each country with it's 2017 HDI Rank
num8Data["HDI Rank (2017)"].ffill(inplace=True, limit=2)



# Fixing the Hong Kong row
num8Data.loc["HKG","HDI Rank (2017)"] = 7.000



# Sort dataframe by HDI Rank value
num8Data.sort_values("HDI Rank (2017)", inplace=True)



# Rearranging columns
tempCol = num8Data.pop("HDI Rank (2017)")
num8Data.insert(1, tempCol.name, tempCol)



# Selecting the last 10 country entries out of 189 countries. 
#The range is [0,30] because each country has 3 key indicators
# These are the dataframes I will analyze for Question 8
bottom_10_question_8 = num8Data.loc[num8Data["HDI Rank (2017)"] > 179].copy()[0:30]
top_10_question_8 = num8Data.copy()[0:30]


# Collecting lists of country names of the top and bottom 10 HDI Rankings for 2017 
HDI_top10_countries = top_10_question_8.index.get_level_values(level=0).unique().to_list()
HDI_bottom10_countries = bottom_10_question_8.index.get_level_values(level=0).unique().to_list()


# ##### Preparing Data for Question 10

# In[8]:


#GDP per capita growth (annual %)
GDP_PC_Code = "NY.GDP.PCAP.KD.ZG" 
GDP_perCapita = minimizeData(WDIData,GDP_PC_Code)
fillNanZero(GDP_perCapita)



#Inflation, consumer prices (annual %)
inflation_code = "FP.CPI.TOTL.ZG"
consumer_inflation = minimizeData(WDIData,inflation_code)
fillNanZero(consumer_inflation)



#Exports of goods and services (annual % growth)
exports_code = "NE.EXP.GNFS.KD.ZG"
goodService_exports = minimizeData(WDIData,exports_code)
fillNanZero(goodService_exports)



#Adjusted net national income (annual % growth)"
net_income_code = "NY.ADJ.NNTY.KD.ZG"
net_national_income = minimizeData(WDIData,net_income_code)
fillNanZero(net_national_income)


# In[21]:


# merging all 4 indicator dataframes into one
num10Data = pd.concat([GDP_perCapita, consumer_inflation, goodService_exports,net_national_income],copy=True, sort=False).sort_index(level=0)


# Add Indicator Code to the  index to Indicator code
num10Data.set_index("Indicator Code", append=True, inplace=True)


# Creating dataframe that only contains the 10 highest HDI indicies and their filtered indicator data
top_10_question_10 = num10Data.loc[num10Data.index.get_level_values("Country Code").isin(HDI_top10_countries)].copy()

# Creating dataframe that only contains the 10 lowest HDI indicies and their filtered indicator data
bottom_10_question_10 = num10Data.loc[num10Data.index.get_level_values("Country Code").isin(HDI_bottom10_countries)]

# List of  bottom 10 country names
bottom_list = ["Burundi", "Burkina Faso","Central African Republic","Liberia","Mali","Mozambique","Niger","Sierra Leone","South Sudan","Chad"]

# Setting index to Country Name to prepare for later transposition
bottom_10_question_10.set_index("Country Name", append=True, inplace=True)




# Preparing all Indicators Data for Analysis

GDP_top = prepareForAnalysis(top_10_question_10,GDP_PC_Code)
GDP_bottom = prepareForAnalysis(bottom_10_question_10,GDP_PC_Code)

inflation_top = prepareForAnalysis(top_10_question_10,inflation_code)
inflation_bottom = prepareForAnalysis(bottom_10_question_10,inflation_code)

exports_top = prepareForAnalysis(top_10_question_10,exports_code)
exports_bottom = prepareForAnalysis(bottom_10_question_10,exports_code)

income_top = prepareForAnalysis(top_10_question_10,net_income_code)
income_bottom = prepareForAnalysis(bottom_10_question_10,net_income_code)



#***************************************************************#



# Fixing the headers for each DataFrame
GDP_top = fixHeader(GDP_top)
GDP_bottom = fixHeader(GDP_bottom)

inflation_top = fixHeader(inflation_top)
inflation_bottom = fixHeader(inflation_bottom)

exports_top = fixHeader(exports_top)
exports_bottom = fixHeader(exports_bottom)

income_top = fixHeader(income_top)
income_bottom = fixHeader(income_bottom)



#***************************************************************#



# Listing info on all Top/Bottom Indicator Dataframes

print("GDP Per Capita of Top 10 HDI DataFrame info\n")
GDP_top.info()
print("\n\nGDP Per Capita of Bottom 10 HDI DataFrame info\n")
GDP_bottom.info()
print("\n\nConsumer Inflation of Top 10 HDI DataFrame info\n")
inflation_top.info()
print("\n\nConsumer Inflation of Bottom 10 HDI DataFrame info\n")
inflation_bottom.info()
print("\n\nExports of Goods/Services of Top 10 HDI DataFrame info\n")
exports_top.info()
print("\n\nExports of Goods/Services of Bottom 10 HDI DataFrame info\n")
exports_bottom.info()
print("\n\nAdj Net National Income of Top 10 HDI DataFrame info\n")
income_top.info()
print("\n\nAdj Net National Income of Bottom 10 HDI DataFrame info\n")
income_bottom.info()



# ### Checking All Distrubtions to check for skewness

# In[18]:


GDP_top.hist(figsize=(30,30))


# In[11]:


GDP_bottom.hist(figsize=(30,30))


# In[12]:


inflation_top.hist(figsize=(30,30))


# In[13]:


inflation_bottom.hist(figsize=(30,30))


# In[14]:


exports_top.hist(figsize=(30,30))


# In[15]:


exports_bottom.hist(figsize=(30,30))


# In[16]:


income_top.hist(figsize=(30,30))


# In[17]:


income_bottom.hist(figsize=(30,30))


# In[ ]:





# In[ ]:




