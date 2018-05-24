"""
Extract 3d coordinates fte that the format of text should be something like this:
====================
20 (point index)
DeltaX (x coordinate)
DeltaY (y coordinate)
DeltaZ (z coordinate)rom .txt file and save as a numpy array.
No

"""
import numpy as np
file = open('C:\\Users\\Xu Yang\\Desktop\\3Dcoordinates.txt')
contents = file.readlines()
numofFeatures = 103
coordinates = np.zeros((numofFeatures, 3))
for i in range(len(contents)):
    if contents[i][0] == '=':
        ptIndex = int(contents[i+1])
        x = float(contents[i+2][9:-4])
        y = float(contents[i+3][9:-4])
        z = float(contents[i+4][9:-4])
        coordinates[ptIndex, :] = x, y, z
file.close()

