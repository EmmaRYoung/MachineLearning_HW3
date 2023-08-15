import numpy as np
import pandas as pd
from scipy import stats as st
import numpy.linalg as LA
import scipy.io
mat = scipy.io.loadmat('mnist_all.mat')

#read in and prepare training data
keep0 = 1000 #Training data has same number of samples for each class (easier)
train0 = mat["train0"]
train0 = train0[0:keep0,:]
train0 = train0/255  

train1 = mat["train1"]
train1 = train1[0:keep0,:]
train1 = train1/255  

train2 = mat["train2"]
train2 = train2[0:keep0,:]
train2 = train2/255  

train3 = mat["train3"]
train3 = train3[0:keep0,:]
train3 = train3/255  

train4 = mat["train4"]
train4 = train4[0:keep0,:]
train4 = train4/255  

train5 = mat["train5"]
train5 = train5[0:keep0,:]
train5 = train5/255  

train6 = mat["train6"]
train6 = train6[0:keep0,:]
train6 = train6/255  

train7 = mat["train7"]
train7 = train7[0:keep0,:]
train7 = train7/255  

train8 = mat["train8"]
train8 = train8[0:keep0,:]
train8 = train8/255  

train9 = mat["train9"]
train9 = train9[0:keep0,:]
train9 = train9/255  

#read in and prepare testing data
keep = 500
test0 = mat["test0"]
test0 = test0[0:keep,:]
test0 = test0/255  

test1 = mat["test1"]
test1 = test1[0:keep,:]
test1 = test1/255  

test2 = mat["test2"]
test2 = test2[0:keep,:]
test2 = test2/255  

test3 = mat["test3"]
test3 = test3[0:keep,:]
test3 = test3/255  

test4 = mat["test4"]
test4 = test4[0:keep,:]
test4 = test4/255  

test5 = mat["test5"]
test5 = test5[0:keep,:]
test5 = test5/255  

test6 = mat["test6"]
test6 = test6[0:keep,:]
test6 = test6/255  

test7 = mat["test7"]
test7 = test7[0:keep,:]
test7 = test7/255  

test8 = mat["test8"]
test8 = test8[0:keep,:]
test8 = test8/255  

test9 = mat["test9"]
test9 = test9[0:keep,:]
test9 = test9/255  


#define PCA function
def PCA(X):
    #step1: Calculate mean and std of the dataset (vector)
    [NumSample, NumFeature] = np.shape(X)
    x_mu = np.reshape(np.mean(X, axis=0), (NumFeature,1))
    
    #step2: subtract the mean from the dataset, construct "A" matrix
    A = np.transpose(X) - x_mu #(784 x 60,000)
    
    #Then construct AA' matrix "C": characterizes the scatter of the data
    C = np.zeros((NumFeature, NumFeature)) 
    for i in range(NumSample):
        vect = np.reshape(A[:,i], (NumFeature,1)) #reshape step makes matrix multiplication work
        C = C + vect@np.transpose(vect)
        
    C = C/NumSample #(784 x 784)
    
    #Compute the eigenvalues and eigenvectors of C
    [W, V] = LA.eig(C) #W: eigenvalues V: eigenvectors
    W = np.real(W)
    V = np.real(V)
    #The column v[:,i] is the eigenvector corresponding to 
    #the eigenvector corresponsing to the eigenvalue w[i]
    
    #find cutoff to maintain variance of 90% and 95% of the data
    thresh1 = 0.90
    thresh2 = 0.95
    
    total_eig = np.sum(W)
    eig_store = 0
    found = 0
    for i in range(len(W)):
        eig_store = eig_store + W[i]
        fraction = np.sum(eig_store)/total_eig
        if fraction > thresh1 and found!=1:
            keep90 = i-1
            found = 1 #the found boolean is so we can look for both keep90 and keep95 in one loop
            
        if fraction > thresh2:
            keep95 = i-1

    #Use for dimensionality reduction, basis (eigen vectors) are trunkated at the indices that represents
    #the desired ammount of variation
    
    #Dimensionality reduction is done in another function
    b90 = np.transpose(V[:,0:keep90])
    b95 = np.transpose(V[:,0:keep95])

    return b90, b95

#define KNN function
def KNN(classify, trainD, keep, K):
    #classify: the testing data we want to classify
    #trainD : training data from each class
    #keep: the number of training samples kept from original 5,000+ samples
    
    
    #the dimensionality reduction for both needs to match for matrix math to work
    #(aka both must use the same number of reduced classes)
    
    [NumSample, NumFeature] = np.shape(classify)
    
    distances = []
    #row: sample
    #column: class
    PredClass = np.zeros(NumSample)
    keepOrig = keep #not really necessary, keep is never modified
    for i in range(NumSample): #go through each sample in the training data
        X = classify[i,:]
        X = np.reshape(X, (NumFeature,1))
        begin = 0 #begin and end variables just help us pull out each class of training data (trainD is ALL traing data)
        end = keep 
        for j in range(10): #Compare current sample to data in each class, store distances
            trainCurr = trainD[begin:end,:] #pull out jth class of training data
            begin = end
            end = begin + keep
            distInd = j*np.ones((1,keepOrig)) #Class of the current training data
            classDist = np.zeros((1,keepOrig)) #initialize to modify in the loop below
            for k in range(keepOrig):
                vector = np.reshape(trainCurr[k,:], (NumFeature,1)) #reshape for matrix math to work
                
                classDist[0, k] = LA.norm(X-vector)
                #rows: sample number
                #columns: class
            
            #vertically stack row of distances from each sample ontop of row of their class
            KNNclassinfo = np.vstack((classDist, distInd))
            
            #horizontally stack the KNNclassinfo with each iteration. 
            #top row is the distances from test to every sample
            #bottom row is the class associated with that distance. 
            #[distance, distance, ... distance]
            #[0 0 ... 0 0 1 1... 1 1 2 2... 2 2 3 3... 3 3 ........ 9 9...9 9]
            #same length vectors
            if j == 0:
                distances = KNNclassinfo
            else:
                distances = np.hstack((distances, KNNclassinfo))
            #hope this makes sense
                
        #Convert to a pandas dataframe to sort more easily
        #data is transposed to make the dataframe sorting work 
        distances = pd.DataFrame(np.transpose(distances), columns = ['dist','class'])
        #sort the first column in ascending order, the class label in the column next to it gets shuffled around,
        #but is still tied to the sample. The pandas dataframe does this easily, that's why I switched it
        #temporarily
        distances_sort = distances.sort_values(by = "dist")
        
        #convert back to a numpyarray because they're cooler
        distances_sort = np.asarray(distances_sort)
        
        #find K smallest values of the sorted distances, but we're only interested in the class (second column)
        KNNClass = distances_sort[0:K,1]
        
        PredClass[i] = st.mode(KNNClass[:])[0]  #the most frequent class in the first K samples gets stored as the predicted class for the testing sample
        
    
    return PredClass

def DimRed(X,basis):
    #reduce the dimensionality of the samples with the basis obtained from PCA
    
    [NumSample, NumFeature] = np.shape(X)
    x_mu = np.reshape(np.mean(X, axis=0), (NumFeature,1))
    A = np.transpose(X) - x_mu 
    
    X_red = basis@A 
    return X_red

def ReportAccuracy(Confusion):
    #obtain useful information from the confusion matrix
    #right now reports accuracy, but can report anything as TP, TN, FN, FP are calculated first
    dim = len(Confusion)
    TPstore = np.zeros((dim,1))
    FNstore = np.zeros((dim,1))
    FPstore = np.zeros((dim,1))
    TNstore = np.zeros((dim,1))
    AccurStore = np.zeros((dim,1))

    for i in range(dim):
        TPstore[i] = Confusion[i,i]
        
        beforei_R = Confusion[i,0:i]
        afteri_R = Confusion[i,i+1:dim]
        beforei_C = Confusion[0:i,i]
        afteri_C = Confusion[i+1:dim,i]
        temp = np.delete(Confusion, obj = i, axis=0)
        AllXcept = np.delete(temp, obj = i , axis=1)
        
        FNstore[i] = np.sum(beforei_R) + np.sum(afteri_R)
        FPstore[i] = np.sum(beforei_C) + np.sum(afteri_C)
        TNstore[i] = np.sum(AllXcept)
        
        AccurStore[i] = (TPstore[i] + TNstore[i])/(TPstore[i] + TNstore[i] + FPstore[i] + FNstore[i])
        
    return AccurStore

#perform PCA, to keep 90% and 95% of the variance of the data
#data is combined so PCA can be done all all the training data, and dimensionality of training and testing 
#can be done all at once

#also, my loops to do KNN work with all the training combined. The samples for the current class of interest
#are extracted in the loop
trainALL = np.vstack((train0,train1,train2,train3,train4,train5,train6,train7,train8,train9))
Ltr_all = len(trainALL)

testALL = np.vstack((test0,test1,test2,test3,test4,test5,test6,test7,test8,test9))
Lte_all = len(testALL)

#gives us the basis for dimensionality reduction
[b90, b95] = PCA(trainALL)
#keep basis(eigen vectors) from training data, need to project testing to lower dimensions with the basis

#dimensionality reduction (training and testing)
trainALL_90 = np.transpose(DimRed(trainALL, b90))
trainALL_95 = np.transpose(DimRed(trainALL, b95))

testALL_90 = np.transpose(DimRed(testALL, b90))
testALL_95 = np.transpose(DimRed(testALL, b95))


begin1 = 0 #for sorting through testing data
end1 = keep
#four confusion matrices!!! initialize here
Confusion90_3 = np.zeros((10,10))
Confusion90_5 = np.zeros((10,10))

Confusion95_3 = np.zeros((10,10))
Confusion95_5 = np.zeros((10,10))
for i in range(10): 
    #go through the testing data (ACTUAL: rows of the confusion matrix)
    testCurr_90 = testALL_90[begin1:end1,:] 
    testCurr_95 = testALL_95[begin1:end1,:] 
    begin1 = end1
    end1 = begin1 + keep
       
    PredClass = KNN(testCurr_90, trainALL_90, keep0, 3) #k=3 
    for j in range(len(PredClass)):
        Confusion90_3[i, int(PredClass[j])] = Confusion90_3[i, int(PredClass[j])] + 1
    
    PredClass = KNN(testCurr_90, trainALL_90, keep0, 5) #k=5
    for j in range(len(PredClass)):
        Confusion90_5[i, int(PredClass[j])] = Confusion90_5[i, int(PredClass[j])] + 1
    
    PredClass = KNN(testCurr_95, trainALL_95, keep0, 3)
    for j in range(len(PredClass)):
        Confusion95_3[i, int(PredClass[j])] = Confusion95_3[i, int(PredClass[j])] + 1
        
    PredClass = KNN(testCurr_95, trainALL_95, keep0, 5)
    for j in range(len(PredClass)):
        Confusion95_5[i, int(PredClass[j])] = Confusion95_5[i, int(PredClass[j])] + 1
        
        
#calculate confusion matrix
AccurStore90_3 = ReportAccuracy(Confusion90_3)
AccurStore90_5 = ReportAccuracy(Confusion90_5)     

AccurStore95_3 = ReportAccuracy(Confusion95_3)
AccurStore95_5 = ReportAccuracy(Confusion95_5)  
        
   
    
