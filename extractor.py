import gzip
import cv2
import numpy as np
import time

s = time.time()

def predict(x,prior,m,S,inv,d):
    class_prob = np.zeros((10))
    x.reshape(1,784)
    for i in range(10):        
        log_posterior =  ((-784/2)*np.log(2*np.pi)) - (0.5*np.log(d[i]))  - np.dot(0.5*np.dot((x-m[i]),inv[i]),np.transpose(x-m[i]))        
        
        class_prob[i] = (prior[i] + log_posterior)
    predicted = np.argmax(class_prob)
    return predicted
        
size = 28
images = 60000

f = gzip.open('train-images-idx3-ubyte.gz','r')


f.read(16)
buf = f.read(size * size * images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

data = data.reshape(images, size, size, 1)

data = data.reshape(60000,size*size)
f = gzip.open('train-labels-idx1-ubyte.gz','r')
f.read(8)
buf = f.read(1*60000)

train_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
priors = [0 for i in range(10)]
log_priors = []
a = np.zeros(784)
b = np.zeros(784)

mean = np.zeros((10,28,28))
sd = np.zeros((10,28,28))
for numbers in range(10):
    a *= 0
    b *= 0
    class_values = data[train_labels == numbers]
    count = 0
    for i in range(60000):
        if(train_labels[i] == numbers):
            a += data[i] 
            count+=1
    a/=count
    
    for i in range(60000):
        if(train_labels[i] == numbers):
            b += np.square(data[i]-a)
    b= np.sqrt(b/count)    
    
    mean[numbers] = np.asarray(a.reshape(size,size))        
    sd[numbers] = np.asarray(b.reshape(size,size))
    cv2.imwrite("output_images\mean_"+str(numbers)+".png",mean[numbers])
    cv2.imwrite("output_images\sd_"+str(numbers)+".png",sd[numbers])
    #Creating priors for each class
    priors[numbers] = len(class_values)/60000
print('Images have been saved.\n')
print("Training and testing with MNIST Digit Data.....")
#Initializing covariance matrix
afsd = np.zeros((10,784,784))
dets = np.zeros((10))
inverses = np.zeros((10,784,784))
log_priors = np.log(priors)

#For Naive Bayes Decision Rule
cov2 = np.zeros((10,784,784))
#for i in range(10):
    

print("Creating 784x784 Covariance Matrix for each class.\n")
for c in range(10):    
    for j in data[train_labels == c]:
        afsd[c] += np.dot(np.transpose(np.asarray(j) - mean.reshape(10,784)[c]),(np.asarray(j)- mean.reshape(10,784)[c]))
    afsd[c]=(afsd[c] * (1/len(data[train_labels == c])))
    #Adding a small number to the diagonal to prevent singular matrix error
    afsd[c] += np.eye(784,784)*0.9
    #cov2[i] = np.eye(784)*np.transposed(np.square(sd[i]))
    #dets[c] = np.linalg.det(afsd[c])
    temp = np.linalg.slogdet(afsd[c])
    
    dets[c] = temp[0] * np.exp(temp[1])
    inverses[c] = np.linalg.inv(afsd[c])
print("Covariance matrix created...\n")
#print(dets)
f = gzip.open('t10k-images-idx3-ubyte.gz','r')
im = 10000

f.read(16)
test_buf = f.read(size * size * im)
test_data = np.frombuffer(test_buf, dtype=np.uint8).astype(np.float32)
test_data = test_data.reshape(im, size,size)

test_data = test_data.reshape(10000,size*size)
f = gzip.open('t10k-labels-idx1-ubyte.gz','r')
f.read(8)
test_buf = f.read(1*10000)
test_labels = np.frombuffer(test_buf, dtype=np.uint8).astype(np.int64)
print("Testing the trained model with the MNIST Training Data.\n This might take a while...\n")
count = 0
#0-1 loss function, loss is 0 if predicted label matches the test label, 0 otherwise
loss = []
for i in range(10000):
    if(predict(test_data[i],log_priors,np.array(mean).reshape(10,784),afsd,inverses,dets) == test_labels[i]):
        #count+=1
        loss.append(0)
    else:
        loss.append(1)
        
print("Bayes Classification Complete.\n")
#print("Accuracy using count :" + str(100*count/10000) + "%")
print("Loss/Error: " + str((100*sum(loss)/10000)) + "%")
print("Accuracy using 0-1 loss function :" + str(100 - (100*sum(loss)/10000)) + "%")
print("Run Time:" + str(time.time() - s))

