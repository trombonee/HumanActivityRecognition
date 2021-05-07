import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
## 0 = Jumping Jacks, 1 = Walking, 2 = Baseball, 3 = Dribbling, 4 = Shooting
with open('Motion-sessions_2021-02-19_15-19-56.csv', 'r') as f:
    i = 0
    sessionID = []
    activityName = []
    gyroX = []
    gyroY = []
    gyroZ = []
    accX = []
    accY = []
    accZ = []
    for line in f:
        entries = line.replace(' ', '').split(',')
        if i > 0:
            if entries[18] != '':
                if entries[0] == '1':
                    sessionID.append(0)
                elif entries[0] == '2':
                    sessionID.append(0)
                elif entries[0] == '3':
                    sessionID.append(0)
                elif entries[0] == '0':
                    sessionID.append(0)
                elif entries[0] == '5':
                    sessionID.append(1)
                elif entries[0] == '9':
                    sessionID.append(2)
                gyroX.append(float(entries[18]))
                gyroY.append(float(entries[19]))
                gyroZ.append(float(entries[20]))
                accX.append(float(entries[21]))
                accY.append(float(entries[22]))
                accZ.append(float(entries[23]))
        i+=1
finalData = [sessionID, gyroX, gyroY, gyroZ, accX, accY, accZ]
finalData = np.asarray(finalData).transpose().tolist()

with open('Motion-sessions_2021-02-26_16-31-13.csv', 'r') as f:
    i = 0
    sessionID = []
    gyroX = []
    gyroY = []
    gyroZ = []
    accX = []
    accY = []
    accZ = []
    for line in f:
        entries = line.replace(' ', '').split(',')
        if i > 0:
            if entries[18] != '':
                if entries[0] == '0':
                    sessionID.append(3)
                elif entries[0] == '1':
                    sessionID.append(4)
                gyroX.append(float(entries[18]))
                gyroY.append(float(entries[19]))
                gyroZ.append(float(entries[20]))
                accX.append(float(entries[21]))
                accY.append(float(entries[22]))
                accZ.append(float(entries[23]))
        i+=1

finalData2 = [sessionID, gyroX, gyroY, gyroZ, accX, accY, accZ]
finalData2 = np.asarray(finalData2).transpose().tolist()
finalData = finalData + finalData2

df = pd.DataFrame(data=finalData, columns=['sessionID', 'gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ'])
print(df)

s = df["sessionID"].squeeze()
label_name = s.map({0: "Jumping Jacks", 1:"Walking", 2:"Baseball", 3:"Dribbling", 4:"Shooting"})

df["activity_name"] = label_name

fig = plt.figure(figsize = (25, 25))
plt.tick_params(labelsize = 25)
a = sns.countplot(x = "sessionID", data = df)
plt.xlabel("Activity Type", fontsize = 35)
plt.ylabel("Count", fontsize = 35)
plt.title("Number of Data Items by Category", fontsize=50)
a.set_xticklabels(['Jumping Jacks','Walking','Baseball','Dribbling','Shooting'])
for i in a.patches:
    a.text(x = i.get_x() + 0.3, y = i.get_height()+20, s = str(i.get_height()), fontsize = 30, color = "grey")
plt.savefig('Histo.png')
plt.show()



df_jumping_jacks = df[df["sessionID"] == 0]
df_jumping_jacks = df_jumping_jacks[:2500]

df_walking = df[df["sessionID"] == 1]
df_walking = df_walking[:2500]

df_baseball = df[df["sessionID"] == 2]
df_baseball = df_baseball[:2500]

df_dribbling = df[df["sessionID"] == 3]
df_dribbling = df_dribbling[:2500]

df_shooting = df[df["sessionID"] == 4]
df_shooting = df_shooting[:2500]

sns.set(font_scale = 1)
fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (21, 14))

def plotDist(name, signal, row, col):
    axes[row][col].set_title("Distribution of" + name)
    sns.distplot(df_jumping_jacks[signal], hist=False, label="Jumping Jacks", ax=axes[row][col])
    sns.distplot(df_walking[signal], hist=False, label="Walking", ax=axes[row][col])
    sns.distplot(df_baseball[signal], hist=False, label="Baseball", ax=axes[row][col])
    sns.distplot(df_dribbling[signal], hist=False, label="Dribbling", ax=axes[row][col])
    sns.distplot(df_shooting[signal], hist=False, label="Shooting", ax=axes[row][col])
    axes[row][col].legend(fontsize=15)

#all plots
plotDist("X Gyroscope", "gyroX", 0, 0)
plotDist("Y Gyroscope", "gyroY", 0, 1)
plotDist("Z Gyroscope", "gyroZ", 0, 2)
plotDist("X Acceleration", "accX", 1, 0)
plotDist("Y Acceleration", "accY", 1, 1)
plotDist("Z Acceleration", "accZ", 1, 2)
plt.tight_layout()
plt.savefig('DistPlot.png')
plt.show()


sns.set(font_scale = 1)
fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (21, 14))

def plotBox(name, signal, row, col):
    sns.boxplot(x="activity_name", y=signal, showfliers=False, data=df, ax=axes[row][col])
    axes[row][col].set_title("Box plot of " + name, fontsize = 15)
    axes[row][col].set_ylabel(name, fontsize=15)
    axes[row][col].set_xlabel("")
    axes[row][col].legend(fontsize=15)


plotBox("X Gyroscope", "gyroX", 0, 0)
plotBox("Y Gyroscope", "gyroY", 0, 1)
plotBox("Z Gyroscope", "gyroZ", 0, 2)
plotBox("X Acceleration", "accX", 1, 0)
plotBox("Y Acceleration", "accY", 1, 1)
plotBox("Z Acceleration", "accZ", 1, 2)
plt.tight_layout()
plt.savefig('BoxPlot.png')
plt.show()



