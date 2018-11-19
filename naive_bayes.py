

# this is a naive bayes implemtation following the basic multinomial event model


import numpy as np 
import pandas as pd 


df = pd.read_csv('trainq2.csv' )
print(df.shape)

t0=0          # number of words occurence in class 0
t1=0          # number of words occurence in class 1
k=0           # word index in the overhall dictionary
dic1 = {}                    # dictionary of class 1
dic0 ={}                     # dictionary of class 0
dic = {}                    # overhall dictionary 
for i in range(5000):
    
    line = (df['question_text'][i]).strip(' ,?/".()/').split(' ')
    for word in line:
        if(word not in dic):
            dic[word]=k
            k+=1
            
        if(df['target'][i]==0):
            if(word in dic0):
                dic0[word]+=1
                t0+=1
            else:
                dic0[word]=1
                t0+=1
        elif (df['target'][i]==1):
            if(word in dic1):
                dic1[word]+=1
                t1+=1
            else:
                dic1[word]=1
                t1+=1


print(t0,t1)



print(len(dic),len(dic0),len(dic1))
n_k = len(dic)


p = np.ones([len(dic),2])    # probability distributions over words for each class
j=0
for xj in dic:

    if xj in dic1:
        p[j][1]=(1+dic1[xj])/(t1+n_k)  # column 1 in p(n_k by 2 array) for spam ( class 1)
        
    else:
        p[j][1] = 1/(t1+n_k)
       
        
    if xj in dic0:
        p[j][0]=(1+dic0[xj])/(t0+n_k)  #  column 0 in p(n_k by 2 array) for not spam ( class 0)
        
    else:
        p[j][0] = 1/(t0+n_k)
    
        
            
    j+=1     
        

y1 = t1/(t1+t0)
y2 = 1-y1

np.save('probxjy_1.npy',p)

np.save('dic_1.npy',dic)
np.save('dic0_1.npy',dic0)
np.save('dic1_1.npy',dic1)                        # save the dictionary and count , to avoid  recomputing from stratch
np.save('t0_1.npy',t0)
np.save('t1_1.npy',t1)                






def prob(question):
    question = question.strip(' ,?/".()/').split(' ')
    lik1=1  
    lik0 = 1
    i=0
    for word in question:
        j = dic[word]
        lik1+=np.log(p[j][1])                # use the log likelihood to avoid floating point underflow
        lik0+=np.log(p[j][0])
    lik1=np.exp(lik1)
    lik0=np.exp(lik0)
    
    res = (lik1*y1)/(lik1*y1+lik0*y2) 
    return res
    
for i in range(20,35):                      # test on a subset of the training data set, the accuaracy should be particularly high since the model was 
    ques  = df['question_text'][i]          # trained on it
    print('i:',i,ques)
    print(100*prob(ques))
    



