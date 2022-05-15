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
                 
                 
                 
                 
                 
