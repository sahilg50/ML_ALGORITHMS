from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from ClassifyNB import Classify

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

clf = Classify(features_train, labels_train)

prettyPicture(clf, features_test, labels_test)


