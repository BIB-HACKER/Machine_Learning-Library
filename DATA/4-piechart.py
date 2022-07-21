import matplotlib.pyplot as plt
import pandas as pd
goal_types =['Penalties','Field Goals','Free Kicks']
goals = [12,38,7]  
mycolors = ['y','r','b']

plt.pie(goals, labels = goal_types, colors=mycolors ,
       shadow =True, explode = (0.0, 0.0, 0.0),
         autopct = '%.2f%%'
      ) 
plt.show() 

df=pd.read_csv("player_data.csv")
df1=df.set_index("Names")
goaltypes=df1.columns
players=df.iloc[:,0]

r_goals=df.iloc[0,1:]
plt.pie(r_goals,labels=goaltypes,colors=mycolors,
        shadow=True,explode=(0.0,0.0,0.0))
plt.title("Ronaldo")
plt.show()