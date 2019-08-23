# implementation of VLAD ballTree algorithms for CBIR
# Jorge Guevara
# jorged@br.ibm.com

import numpy as np 
import itertools
# from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
import pickle
import glob
import cv2
from VLADlib.Descriptors import *

# xianjie parallel
#from mpi4py import MPI
from pyspark import SparkConf, SparkContext
from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors


# inputs
# getDescriptors for whole dataset
# Path = path to the image dataset
# functionHandleDescriptor={describeSURF, describeSIFT, describeORB}
def getDescriptors(path,functionHandleDescriptor):
#     xianjie parallel: utilize MPI doing parallel computing; split 359 files into 4 tasks and let 4 processors to process them
 
#     comm = MPI.COMM_WORLD
#     my_rank = comm.Get_rank()   # rank of this process
#     p = comm.Get_size()         # number of processes
#     image_paths = glob.glob(path+"/*.jpg")
#     n = len(image_paths)        # number of images
#     print(n,p)
#     h = np.ceil(n/p)            # number of images that each process should handle
#     dest = 0                    # destination process
#     local_start = int(0+my_rank*h)   # local images' start
#     print("local starting at: ", local_start)
#     local_end = int(local_start + h) # local images' end
#     print("local starting at: ", local_end)
#     portion = image_paths[local_start:local_end]
#     temp = list()               # temporary list of descriptors for each process
#     for path in portion:        # go thru all the images assigned in each process
#         im = cv2.imread(path)
#         kp,des = functionHandleDescriptor(im)
#         if not (des is None):
#             temp.append(des)
##

    descriptors=list()

#     xianjie parallel
#     if my_rank == 0:            # the process that reduce all the work
#         for source in range(1,p):
#             temp = comm.recv(source=source)
#             print("Descriptors length of kp ",my_rank,"<-",source,",",len(kp),"\n")
#             descriptors = descriptors + temp
#     else:
#         print("Descriptors length of kp ",my_rank,"->",dest,",",len(kp),"\n")
#         comm.send(temp, dest=0)

#     if my_rank == 0:
#         print("The process is complete!")
#         #flatten list
#         descriptors = list(itertools.chain.from_iterable(descriptors))
#         #list to array
#         descriptors = np.asarray(descriptors)
#     MPI.Finalize
#     return descriptors
##
    
   

    for imagePath in glob.glob(path+"/*.jpg"):
        print(imagePath)
        im=cv2.imread(imagePath)
        kp,des = functionHandleDescriptor(im)
        if not (des is None):
            descriptors.append(des)
            print(len(kp))

   #flatten list
    descriptors = list(itertools.chain.from_iterable(descriptors))
   #list to array
    descriptors = np.asarray(descriptors)

    return descriptors


# input
# training = a set of descriptors
def  kMeansDictionary(training, k):

#   xianjie parallel
    spark = SparkSession\
        .builder\
        .appName("PythonKMeans")\
        .getOrCreate()
    dff = map(lambda x: (int(x[0]), Vectors.dense(x[:])), training)
    mydf = spark.createDataFrame(dff,schema=["label", "features"])
    kmeans =KMeans().setK(k).setSeed(1)
    model = kmeans.fit(mydf.select("features"))
    return model
##    
    
    #K-means algorithm
#     est = KMeans(n_clusters=k,init='k-means++',tol=0.0001,verbose=1).fit(training)
    #centers = est.cluster_centers_
    #labels = est.labels_
    #est.predict(X)
#     return est
    #clf2 = pickle.loads(s)

# compute vlad descriptors for te whole dataset
# input: path = path of the dataset
#        functionHandleDescriptor={describeSURF, describeSIFT, describeORB}
#        visualDictionary = a visual dictionary from k-means algorithm


def getVLADDescriptors(path,functionHandleDescriptor,visualDictionary):
    descriptors=list()
    idImage =list()
    for imagePath in glob.glob(path+"/*.jpg"):
        print(imagePath)
        im=cv2.imread(imagePath)
        kp,des = functionHandleDescriptor(im)
        if des is not None:
            v=VLAD(des,visualDictionary)
            descriptors.append(v)
            idImage.append(imagePath)
                    
    #list to array    
    descriptors = np.asarray(descriptors)
    return descriptors, idImage


# fget a VLAD descriptor for a particular image
# input: X = descriptors of an image (M x D matrix)
# visualDictionary = precomputed visual dictionary

# compute vlad descriptors per PDF for te whole dataset, f
# input: path = dataset path
#        functionHandleDescriptor={describeSURF, describeSIFT, describeORB}
#        visualDictionary = a visual dictionary from k-means algorithm


def getVLADDescriptorsPerPDF(path,functionHandleDescriptor,visualDictionary):
    descriptors=list()
    idPDF =list()
    desPDF= list()

    #####
    #sorting the data
    data=list()
    for e in glob.glob(path+"/*.jpg"):
        #print("e: {}".format(e))
        s=e.split('/')
        #print("s: {}".format(s))
        s=s[1].split('-')
        #print("s: {}".format(s))
        s=s[0].split('.')
        #print("s: {}".format(s))
        s=int(s[0]+s[1])
        #print("s: {}".format(s))

        data.append([s,e])

    data=sorted(data, key=lambda atr: atr[0])
    #####

    #sFirst=glob.glob(path+"/*.jpg")[0].split('-')[0]
    sFirst=data[0][0]
    docCont=0
    docProcessed=0
    #for imagePath in glob.glob(path+"/*.jpg"):
    for s, imagePath in data:
        #print(imagePath)
        #s=imagePath.split('-')[0]
        #print("s : {}".format(s))
        #print("sFirst : {}".format(sFirst))

        #accumulate all pdf's image descriptors in a list
        if (s==sFirst):
            
            im=cv2.imread(imagePath)
            kp,des = functionHandleDescriptor(im)
            if des!=None:
                desPDF.append(des)   
            
        else:
            docCont=docCont+1
            #compute VLAD for all the descriptors whithin a PDF
            #------------------
            if len(desPDF)!=0: 
                docProcessed=docProcessed+1
                #print("len desPDF: {}".format(len(desPDF)))
                #flatten list       
                desPDF = list(itertools.chain.from_iterable(desPDF))
                #list to array
                desPDF = np.asarray(desPDF)
                #VLAD per PDF
                v=VLAD(desPDF,visualDictionary)     
                descriptors.append(v)
                idPDF.append(sFirst)
            #------------------
            #update vars
            desPDF= list()
            sFirst=s
            im=cv2.imread(imagePath)
            kp,des = functionHandleDescriptor(im)
            if des!=None:
                desPDF.append(des)

    #Last element
    docCont=docCont+1
    if len(desPDF)!=0: 
        docProcessed=docProcessed+1
        desPDF = list(itertools.chain.from_iterable(desPDF))
        desPDF = np.asarray(desPDF)
        v=VLAD(desPDF,visualDictionary)     
        descriptors.append(v)
        idPDF.append(sFirst)
                    
    #list to array    
    descriptors = np.asarray(descriptors)
    print("descriptors: {}".format(descriptors))
    print("idPDF: {}".format(idPDF))
    print("len descriptors : {}".format(descriptors.shape))
    print("len idpDF: {}".format(len(idPDF)))
    print("total number of PDF's: {}".format(docCont))
    print("processed number of PDF's: {}".format(docProcessed))

    return descriptors, idPDF


# fget a VLAD descriptor for a particular image
# input: X = descriptors of an image (M x D matrix)
# visualDictionary = precomputed visual dictionary

def VLAD(X,visualDictionary):

    # xianjie parallel
    spark = SparkSession\
        .builder\
        .appName("PythonKMeans")\
        .getOrCreate()
    dff = map(lambda x: (int(x[0]), Vectors.dense(x[0:])), X) # data type conversion
    mydf = spark.createDataFrame(dff,schema=["label", "features"]) # data type conversion
    
    transformed = visualDictionary.transform(mydf) # transformed dataframe
    prediction = transformed.select('prediction').collect()
    realpredictedLabels = [p.prediction for p in prediction] # prediction of each point
    predictedLabels = np.asarray(realpredictedLabels, dtype=np.int32)
    centers = visualDictionary.clusterCenters() # cluster centers
#     print(centers.shape)
#    print(centers)
#     print(predictedLabels)
#     print(realpredictedLabels)
#     print(centers)
#     labelsrow = visualDictionary.summary.predictions.collect()
#     labels = [l.prediction for l in labelsrow ]
    k =len(centers)
    ##
    
#     predictedLabels = visualDictionary.predict(X)
#     centers = visualDictionary.cluster_centers_
#     labels=visualDictionary.labels_
#     k=visualDictionary.n_clusters
   
    m,d = X.shape
    V=np.zeros([k,d])
    #computing the differences


    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels==i)>0:
            # add the diferences
            V[i]=np.sum(X[predictedLabels==i,:]-centers[i],axis=0)

    V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = np.sign(V)*np.sqrt(np.abs(V))

    # L2 normalization

    V = V/np.sqrt(np.dot(V,V))
    return V



#Implementation of a improved version of VLAD
#reference: Revisiting the VLAD image representation
def improvedVLAD(X,visualDictionary):

    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    labels=visualDictionary.labels_
    k=visualDictionary.n_clusters
   
    m,d = X.shape
    V=np.zeros([k,d])
    #computing the differences

    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels==i)>0:
            # add the diferences
            V[i]=np.sum(X[predictedLabels==i,:]-centers[i],axis=0)
    

    V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = np.sign(V)*np.sqrt(np.abs(V))

    # L2 normalization

    V = V/np.sqrt(np.dot(V,V))
    return V

def indexBallTree(X,leafSize):
    tree = BallTree(X, leaf_size=leafSize)              
    return tree

"""bellow is the code from Github Issues Thread
"""

#typeDescriptors =SURF, SIFT, OEB
#k = number of images to be retrieved
def query(image, k,descriptorName, visualDictionary,tree):
    #read image
    im=cv2.imread(image,0)
    #compute descriptors
    dict={"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}
    funDescriptor=dict[descriptorName]
    kp, descriptor=funDescriptor(im)

    #compute VLAD
    v=VLAD(descriptor,visualDictionary)

    #print(v.reshape(1, -1))
    #print(k)

    #find the k most relevant images
    dist, ind = tree.query(v.reshape(1, -1), k)
    return dist, ind

"""
bellow is the original code


#typeDescriptors =SURF, SIFT, OEB
#k = number of images to be retrieved
def query(image, k,descriptorName, visualDictionary,tree):
    #read image
    im=cv2.imread(image)
    #compute descriptors
    dict={"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}
    funDescriptor=dict[descriptorName]
    kp, descriptor=funDescriptor(im)

    #compute VLAD
    v=VLAD(descriptor,visualDictionary)

    #find the k most relevant images
    dist, ind = tree.query(v, k)    

    return dist, ind

"""





	




