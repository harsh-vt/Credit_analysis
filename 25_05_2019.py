import pandas as pd
import numpy as np
s = pd.Series(np.random.randn(4))
print("The axes are: ")
print (s.axes) # returns dimentions overall, 
print("Is object empty?")
print (s.empty) # returns if empty
print("The the dimention of the object: ")
print (s.ndim) # returns dimention of series
print("The size of vector: ")
print (s.size) # return size of series
print("The actual data series is: ")
print (s.values) # returns values in series as list
print("The first two rows of the data is: ")
print (s.head(2)) # reutrns first two elements of series
d = { "Name" : pd.Series(["Tom","James", "Ricky","Vin", "Steve", "Smith", "Jack"]), 
     "Age" :  pd.Series([25, 26, 25, 23, 30, 29, 23]), 
     "Rating" :  pd.Series([4.23, 3.24, 3.98, 2.56, 3.20, 4.6, 3.8])}
df1 = pd.DataFrame(d)
print ("Our data series is")
print (df1)
print (df1.T) # transpose the data series
print ("Row axis and column axes are")
print (df1.axes) # #L5
print ("Size of data is")
print (df1.size)
N = 20
df2 =  pd.DataFrame({
        'A' : pd.date_range(start = "2016-01-01", periods = N, freq = 'D'), 
        'x' : np.linspace(0, stop = N-1, num = N), 
        'y' : np.random.randn(N), 
        'C' : np.random.choice(["Low", "Medium", "High"], N).tolist(), 
        'D' : np.random.normal(100, 10, size=(N)).tolist()})
for col in df2:
    print (col)
df3 = pd.DataFrame(np.random.randn(4,3), columns = ["col1", "col2", "col3"])
for key,value in df3.iteritems():
     print (key, value)
u_df = pd.DataFrame(np.random.randn(10,2), index = [1, 4, 6, 2, 3, 5, 9, 8, 9, 7], columns = ["col2", "col1"])
print (u_df)
s_df = u_df.sort_index(ascending = True, axis = 0, kind = "mergesort") # sorts data series with arguements
print (s_df)
s_df = u_df.sort_values(ascending = True, by = ["col1","col2"]) # sorts data series with arguements
print (s_df)
df4 = pd.DataFrame(np.random.randn(8, 4), index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], columns = ['A', 'B', 'C', 'D'])
print (df4.loc[['a', 'b', 'c'],['A','C']]) # selects rows and columns from data series
print (df4.loc[['a', 'b', 'c'],['A','C']]>0)
df5 = pd.DataFrame(np.random.randn(8, 4),  columns = ['A', 'B', 'C', 'D'])
print (df5.iloc[:4]) # selects rows and columns from data series using integers
df6 = pd.DataFrame(np.random.randn(8, 4),  columns = ['A', 'B', 'C', 'D'])
#print (df6.ix[:,'A']) # selects rows and columns from data series using integers and labels
s = pd.Series([1,2,3,4,5,4])
print (s.pct_change())
df7 = pd.DataFrame(np.random.randn(5,2))
print (df7.pct_change())
s1 = pd.Series(np.random.randn(10))
s2 = pd.Series(np.random.randn(10))
print (s1.cov(s2)) # gets covariance b/w two column
df8 = pd.DataFrame(np.random.randn(10, 5),  columns = ['a', 'b', 'c', 'd', 'e'])
print (df8['a'].corr(df8['b']))
print (df8.corr) # gets correlation b/w two columns
