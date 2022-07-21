from turtle import color
import matplotlib.pyplot as plt
goal_types = ['penalties','field goals','free kicks']
goals =[12,38,7]
mycolors = ['r','g','b']

plt.pie(goals,labels = goal_types, colors=mycolors,
         shadow = True, explode = (0.0, 0.1, 0.2),
         autopct='%.0f%%')

plt.show()

import pandas as pd
df=pd.read_csv("Machine Learning\DATA\player_data.csv")
print(df)
print('######################')
df1=df.set_index("Names")
print(df1)
print('######################')
goaltypes=df1.columns
players=df.iloc[:,0]
print(players)
print('######################')
r_goals=df.iloc[0,1:]
print(r_goals)

plt.pie(r_goals, labels=goaltypes,colors=mycolors,shadow=True,explode=(0.0,0.0,0.0))

plt.title("Ronaldo")
plt.show()
