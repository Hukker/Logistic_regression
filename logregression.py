import pandas as pd
import numpy as np



import numpy as np
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
)

path = './dataset/diabetes2.csv'
diabetes_df = pd.read_csv(path)

X = diabetes_df.drop(['Outcome'],axis =1 )
y = diabetes_df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=True
                                                    )





class LogRegression():
    def __init__(self, X: pd.DataFrame, y:list):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.w = np.random.randn(self.n_features)       
        self.b = np.random.rand(self.n_samples)
        
        print(f'размер датасета: {self.n_samples}')
        print(f'размер фичей: {self.n_features}')
    
        
    #X - vector
    #w - vector
    #b - number
    #a is sigmoid function(model predict)
    def a(self, X:np.ndarray, w:np.ndarray, b: float): 
        return 1/(1+np.exp(-np.dot(X,w)+b))
            
    def L_i(self, X : np.ndarray, y:float ,w : np.ndarray,b:float):
        return -y * np.log(self.a(X,w,b)) - (1 - y)*np.log(1 - self.a(X,w,b))
    
    def Loss(self, X: np.ndarray,y:np.ndarray, w:np.ndarray, b:np.ndarray):
        L = 0
        for i in range(self.n_samples):
            L += self.L_i(X[i],y[i],w,b[i])
        Loss = L/self.n_samples
        
        return Loss
    
    def gradient_descent_w(self, X: np.ndarray, y: np.ndarray,b: np.ndarray, alpha = 0.0005):
        w = np.array(self.w)
        y_pred = []
        X1 = np.array(X)
        for i in range(self.n_samples):
            y_pred.append(self.a(X1[i],self.w,b[i]))
        
        y_pred1 = np.array(y_pred)
        
        # print(y_pred1.shape)
        counter = 0
        w1 = w - alpha * np.dot(X.T, (y_pred1 - y))
        while self.Loss(X,y,w1,b) < self.Loss(X,y,w,b):
            counter += 1
            if counter == 5000:
                break
            w = w1
            w1 -= alpha*np.dot(X , (y_pred1 - y))
        counter = 0
        self.w = w1        
        return w1
            
    def gradient_descent_b(self, X:np.ndarray, y: np.ndarray, alpha = 0.0005):
        b = self.b
        b1 = list(b)
        counter = 0
        for i in range(self.n_samples):
            b1[i] = b[i] - alpha*( self.a(X[i],self.w,b[i]) - y[i])
            while b1[i] < b[i]:
                counter += 1
                if counter == 5000:
                    break
                
                b[i] = b1[i]
                b1[i] -= alpha*( self.a(X[i],self.w,b[i]) - y[i] )
        
        self.b = np.array(b1)
        return b1
        
    def predict(self, X: np.ndarray):
        return self.a(X,self.w,self.b)
    # def gradient_descent_b(self, X: np.ndarray, y:float,b:float, beta = 0.001):
                
        
    # def train(self,X,y, alpha = 0.001):
        
        
    
def main():
    model = LogRegression(X_train, y_train)
    X = np.array(X_train)
    y = np.array(y_train)
    # print(X[0])
    w = model.gradient_descent_w(X, y, model.b, alpha = 0.0005)
    b = model.gradient_descent_b(X, y, alpha = 0.0005)
    print(b.size)
    print(w)
    print(b)
    
    
if __name__ == '__main__':
    main()