import pandas as pd 
import numpy as np 
df=pd.read_csv("c:\\Users\Omojire\Downloads\\insurance_data (1).csv")
y_true=df['bought_insurance']
x=df[['age','affordibility']]
x['age']=x['age']/x['age'].max()
x['affordibility']=x['affordibility']/x['affordibility'].max()

class myNeuralnetwork :
        
    def __init__(self):
        self.m1=0
        self.m2=0
        self.b=0
                              

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def forward_propagation(self,x,m1,m2,b):
        z=x['age']*m1+x['affordibility']*m2+b
        
        return self.sigmoid(z)

    
    def log_loss(self,y_true,y_predicted,eps=1e-15): #log loss function
        y_predicted= np.clip(y_predicted,eps,1-eps)# to avoid log(0)
    
        log_loss=-(y_true*np.log(y_predicted)+(1-y_true)*np.log(1-y_predicted))
        return np.mean(log_loss)


    def backward_propagation(self,x,y_true,epochs):
        m1=m2=b=0
        learning_rate=0.01
        n=len(y_true)
        for _ in range(1,epochs):
            y_predicted=self.forward_propagation(x,m1,m2,b)
            md1=x['age'].dot(y_predicted-y_true)/n
            md2=x['affordibility'].dot(y_predicted-y_true)/n
            bd=np.mean(y_predicted-y_true)

            loss=self.log_loss(y_true,y_predicted)
            m1=m1-learning_rate*(md1)
            m2=m2-learning_rate*(md2)
            b=b-learning_rate*(bd)
        return m1,m2,b
    
    def fit(self,x,y_true,epochs):
        self.m1,self.m2,self.b=self.backward_propagation(x,y_true,epochs=5000)
        return self.m1,self.m2,self.b     
    
    def predict(self,x):
        prediction=self.forward_propagation(x,self.m1,self.m2,self.b)
        return prediction


myNeuron=myNeuralnetwork()
myNeuron.fit(x,y_true,epochs=5000)
print(myNeuron.predict(x))

    
    
    
