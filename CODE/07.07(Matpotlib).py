import matplotlib.pyplot as plt
x=['1st','2nd','3rd','4th']
y=[9.82,9.27,9.16,9.58]
plt.plot(x,y,'rp')
plt.plot(x,y)
plt.xlabel("1st to 4th semister")
plt.ylabel("Grade Point")
plt.title("My Semister result graph")
plt.show()