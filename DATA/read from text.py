import pandas as pd

df =pd.read_table('dist.txt', 
         delim_whitespace=True,
                  names=('roll', 'B', 'C'))
print(df)

print("#################")
data = pd.read_csv('dist.txt', delimiter= '\s+',
                   header=None, index_col=False)
data.columns = ["a", "b", "c"]
print(data)


