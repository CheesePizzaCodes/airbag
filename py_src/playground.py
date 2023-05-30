import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('./datasets/opendataset/sensor_data/SA06/S06T34R01.csv')

print(data)

fig, ax = plt.subplots()

for key in ['AccX', 'AccY', 'AccZ']:
    ax.plot(data[key])

ax.axvline(x=378, color = 'r')
ax.axvline(x=443, color = 'r')

plt.show()

data2 = pd.read_excel('./datasets/opendataset/label_data/SA06_label.xlsx')

print(data2)
print('hello')
