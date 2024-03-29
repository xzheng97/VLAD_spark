# Jorge Guevara
# jorged@br.ibm.com
# retrieve the most similar k images for an image query
# USAGE :
# python query.py  --query image --retrieve retrieve --descriptor descriptor --visualDictonary visualDictonary --index indexTree 
# python query.py  -q image -r retrieve -d descriptor -dV visualDictionary -i indexTree 

# example :
# python query.py  -q queries/1409.1047-img-3-06.jpg -r 7 -d SURF -dV visualDictionary/visualDictionary16SURF.pickle -i ballTreeIndexes/index_SURF_W16.pickle


from VLADlib.VLAD import *
from VLADlib.Descriptors import *
import itertools
import argparse
import glob
import cv2

# xianjie parallel
from pyspark.ml.clustering import KMeans, KMeansModel
##

#parser
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required = True,
	help = "Path of a query image")

ap.add_argument("-r", "--retrieve", required = True,
	help = "number of images to retrieve")
ap.add_argument("-d", "--descriptor", required = True,
	help = "descriptors: SURF, SIFT or ORB")
ap.add_argument("-dV", "--visualDictionary", required = True,
	help = "Path to the visual dictionary")
ap.add_argument("-i", "--index", required = True,
	help = "Path of the Ball tree")

args = vars(ap.parse_args())

## Vicky's modification
# #args
# path = args["query"]
# #path = "/Users/vickyli/Desktop/Rokid/RemoteRokid/docker_volume/vlad/Train/FountainB0.jpg"
# #k=int(args["retrieve"])
# k = 4 #number of images to retrieve
# #descriptorName=args["descriptor"]
# descriptorName = "SIFT"
# #pathVD = args["visualDictionary"]
# pathVD = "/Users/vickyli/Desktop/Rokid/RemoteRokid/docker_volume/vlad/VisualDictionary/trainVisualDictionary4SIFT.pickle"
# #treeIndex=args["index"]
# treeIndex = "/Users/vickyli/Desktop/Rokid/RemoteRokid/docker_volume/vlad/ballTreeIndexes/index_train_SIFT_VW4.pickle"

path = args["query"]
k=int(args["retrieve"])
descriptorName=args["descriptor"]
pathVD = args["visualDictionary"]
treeIndex=args["index"]

#load the index
with open(treeIndex, 'rb') as f:
    indexStructure=pickle.load(f)

# #load the visual dictionary
# with open(pathVD, 'rb') as f:
#     visualDictionary=pickle.load(f)    
    
# xianjie parallel
spark = SparkSession\
        .builder\
        .appName("PythonKMeans")\
        .getOrCreate()
visualDictionary = KMeansModel.load(pathVD)
##

imageID=indexStructure[0]
tree = indexStructure[1]
pathImageData = indexStructure[2]

#print("bellow is pathImageData:")
#print(pathImageData)

#computing descriptors
dist,ind = query(path, k,descriptorName, visualDictionary,tree)

#print("bellow is dist value")
#print(dist)
#print("bellow is ind value")
#print(ind)

ind=list(itertools.chain.from_iterable(ind))

#print("image path is "+path)

# display the query
imageQuery=cv2.imread(path)
#cv2.imshow("Query", imageQuery)
#cv2.waitKey(0);

# loop over the results
for i in ind:
	# load the result image and display it
	imagePath = imageID[i]
	print(imagePath[23:-4])
	result = cv2.imread(imageID[i])
#	cv2.imshow("Result", result)
	cv2.imwrite("/home/rokid/result"+str(i)+".jpg", result)
	#cv2.waitKey(0)

