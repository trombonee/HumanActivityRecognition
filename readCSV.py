import numpy as np

def readFile(filename):
    with open(filename, 'r') as f:
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
                if int(entries[0]) > 0 and entries[18] is not '':
                    sessionID.append(entries[0])
                    gyroX.append(entries[18])
                    gyroY.append(entries[19])
                    gyroZ.append(entries[20])
                    accX.append(entries[21])
                    accY.append(entries[22])
                    accZ.append(entries[23])
            i+=1
        
        gyroX = window(np.asarray(gyroX).astype(float))
        gyroY = window(np.asarray(gyroY).astype(float))
        gyroZ = window(np.asarray(gyroZ).astype(float))
        accX = window(np.asarray(accX).astype(float))
        accY = window(np.asarray(accY).astype(float))
        accZ = window(np.asarray(accZ).astype(float))
        
        finalData = [gyroX, gyroY, gyroZ, accX, accY, accZ]      
        finalData = np.transpose(np.asarray(finalData), (1,2,0))

        print(finalData.shape)
        

def window(a, w = 120, o = 60, copy = False):
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view




readFile('sampleData.csv')