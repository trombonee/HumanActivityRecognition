import pandas as pd
import numpy as np

stdData = np.load('x_test.npy')
trainingData = np.load('x_train.npy')

featureData = []
for i in range(len(trainingData)):
   sample = trainingData[i].transpose()
   gyroXMean = sample[0].mean()
   gyroYMean = sample[1].mean()
   gyroZMean = sample[2].mean()
   accXMean = sample[3].mean()
   accYMean = sample[4].mean()
   accZMean = sample[5].mean()

   gyroXstd = sample[0].std()
   gyroYstd = sample[1].std()
   gyroZstd = sample[2].std()
   accXstd = sample[3].std()
   accYstd = sample[4].std()
   accZstd = sample[5].std()

   gyroXmax = sample[0].max()
   gyroYmax = sample[1].max()
   gyroZmax = sample[2].max()
   accXmax = sample[3].max()
   accYmax = sample[4].max()
   accZmax = sample[5].max()

   gyroXmin = sample[0].min()
   gyroYmin = sample[1].min()
   gyroZmin = sample[2].min()
   accXmin = sample[3].min()
   accYmin = sample[4].min()
   accZmin = sample[5].min()

   magGyro = np.linalg.norm(np.array([gyroXMean, gyroYMean, gyroZMean]))
   magAcc = np.linalg.norm(np.array([accXMean, accYMean, accZMean]))

   overallMag = np.linalg.norm(np.array([magGyro, magAcc]))

   newData = [gyroXMean, gyroYMean, gyroZMean, accXMean, accYMean, accZMean, gyroXstd, gyroYstd, gyroZstd, accXstd, accYstd, accZstd, gyroXmax, gyroYmax, gyroZmax, accXmax, accYmax, accZmax, gyroXmin, gyroYmin, gyroZmin, accXmin, accYmin, accZmin, magGyro, magAcc, overallMag]
   featureData.append(newData)

featureData = np.asarray(featureData)
np.save('trainingFeatureData', featureData)
