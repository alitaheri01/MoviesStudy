# First let's import the packages we will use in this project
# You can do this all now or as you need them
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib

plt.style.use('ggplot')
from matplotlib.pyplot import figure

%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12,8) # Adjust the configurations of the plots we will create




# Read in the data

df = pd.read_csv(r' movies.csv)
         
                 
                 
                 
# Let's look at the data

df.head()                 


                 
# Let's see if there is any missing data

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col,pct_missing)) 
    

                 
                 
# Data types for our columns

df.dtypes
                 
 
                 
                 
# Create correct year column

df['monthcorrect'] = df['released'].astype(str).str[:3]

df.head()                 
                 
                 
                 
                           
df.sort_values(by=['gross'], inplace=False, ascending=False)

pd.set_option('display.max_rows', None)
                 
                                  

                 
# Drop any duplication

df['company'].drop_duplicates().sort_values(ascending=False)
                 
                 
df['company'].sort_values(ascending=False)
                 
df['company'] = df['company'].drop_duplicates().sort_values(ascending=False)
                 

df.drop_duplicates()
                 
  
                 
                 
# Budget high correlation
# Company high correlation

# Scatter plot with budget and gross
import matplotlib.pyplot as plt
plt.scatter(x=df['budget'], y=df['gross'])

plt.show()               
                 
                 
    
df = df.sort_values(by=['gross'], inplace=False, ascending=False)
df.head()
  
    
                 
                 
                 
# Scatter plot with budget and gross

plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget vs Gross Earning')
plt.xlabel('Gross Earning')
plt.ylabel('Budget for Film')
plt.show()
                 
                 
  
                 
                 
                 
# Plot the budget vs gross using seaborn

sns.regplot(x='budget', y='gross', data=df, scatter_kws={"color":"red"}, line_kws={"color":"blue"})
                 
                 
                 
                
# Let's start looking at correlation
# peason, kendall, spearman

df.corr()
                 
                 
df.corr(method='pearson')
                                
df.corr(method='kendall')            
                 
df.corr(method='spearman')
                 
                 
                 
# High correlation between budget and gross
# I was right

correlation_matrix = df.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)

plt.title('Correlation Matric for Numeric Feature')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')

plt.show()
                 
                 
                
                 
correlation_matrix = df.corr(method='kendall')
sns.heatmap(correlation_matrix, annot=True)

plt.title('Correlation Matric for Numeric Feature')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')

plt.show()
                 
                 
                 
                 
                 
                 
correlation_matrix = df.corr(method='spearman')
sns.heatmap(correlation_matrix, annot=True)

plt.title('Correlation Matric for Numeric Feature')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')

plt.show()
                 
                 
                 
                 
# Look at company

df.head()
                 
                 
                 
                 
df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
        
df_numerized
                 
                 
                 
# compare with df

df.head()
                 
                 
                 
                 
# replace df with df_numerized

correlation_matrix = df_numerized.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)

plt.title('Correlation Matric for Numeric Feature')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')

plt.show()
                 
                 
df_numerized.corr()
                 
                 
correlation_mat = df_numerized.corr()
corr_pairs = correlation_mat.unstack()
corr_pairs
                 
                 
sorted_pairs = corr_pairs.sort_values()
sorted_pairs
                 
                 
high_corr = sorted_pairs[(sorted_pairs)>0.4]
high_corr
                 
                 

# Votes and budget have the highest correlation to gross earning
# Company has low correlation
# I was wrong
