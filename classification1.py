import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# ============================================================
# 1. LOAD DATA
# ============================================================

data = pd.read_csv("student-lifestyle-and-stress-dataset.csv")
print(data.head())


feature_names = ["Sleep_Hours",
"Study_Hours",
"Social_Media_Hours",
"Attendance",
"Exam_Pressure",
"Family_Support",
"Month"]

# Remove rows with missing labels
data = data.dropna(subset=["Stress_Level"])

data[feature_names] = data[feature_names].fillna(
    data[feature_names].mean()
)


x_all = data[feature_names].values.astype(float)
y_all = data["Stress_Level"].values.astype(float)




# ============================================================
# 2. TRAIN-TEST SPLIT
# ============================================================

np.random.seed(42)

indices = np.random.permutation(len(x_all))
split_index = int(0.8 * len(x_all))

train_indices = indices[:split_index]
test_indices = indices[split_index:]

x_train = x_all[train_indices]
y_train = y_all[train_indices]

x_test = x_all[test_indices]
y_test = y_all[test_indices
               ]
print("Feature shape:", x_all.shape)
print("Target shape :", y_all.shape)
print("Target classes:", np.unique(y_all))
#print["x_train", x_train]
# ============================================================
# 3. FEATURE SCALING
# ============================================================

def feature_scalling(x):
    mu=np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    sigma[sigma == 0 ]=1
    x_norm = (x - mu) / sigma
    return x_norm, mu, sigma 

x_train_norm, mu, sigma = feature_scalling(x_train)
x_test_norm =  (x_test - mu) / sigma

# ============================================================
# 4. LOGISTIC MODEL
# ============================================================

def logistic_regression_model(x,w,b):
    z = np.dot(x,w)+ b
    return z

############# sigmoid function #####

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


# ============================================================
# 5. COST FUNCTION
# ============================================================

def compute_cost_logistic(x,y,w,b):
    m,n = x.shape
    cost = 0.0

    for i in range(m):
        z_wb = 0
        for j in range(n):
            
            z_wb_ij = w[j] * x[i][j]
            z_wb += z_wb_ij
        z_wb += b
        f_wb = sigmoid(z_wb)
        cost += - y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb) 
    cost = cost / m
    return cost 

# ============================================================
# 6. GRADIENT
# ============================================================
# Compute and display gradient with w and b initialized to zeros

def compute_gradient(x,y,w,b):
    m,n = x.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    ### START CODE HERE ### 

    for i in range(m):
        z_wb = 0
        for j in range(n):
            
            z_wb_ij = w[j] * x[i][j]
            z_wb += z_wb_ij
        z_wb += b
        f_wb = sigmoid(z_wb)  
        error =  f_wb - y[i] 
        
        dj_db += error 
        for j in range(n):

            dj_dw_ij = error * x[i][j]
            dj_dw[j] += dj_dw_ij
        
    dj_dw = dj_dw / m
    dj_db = dj_db /m   
    ### END CODE HERE ###
    
    return dj_dw, dj_db




# ============================================================
# 7. REGULARIZED COST
# ============================================================

def cost_regularization(x,y,w,b,lambda_):
    m, n = x.shape
    cost_without_reg = compute_cost_logistic(x,y,w,b)
    
    reg_cost=0
    for j in range(n):
        reg_cost_j = w[j]**2
        reg_cost = reg_cost + reg_cost_j
    reg_cost=( lambda_ /(2 * m)) * reg_cost    
    total_cost = cost_without_reg + reg_cost
    return total_cost



# ============================================================
# 8. REGULARIZED GRADIENT
# ============================================================

def regularized_gradient(x, y, w, b, lambda_):
    m, n = x.shape

    dj_dw, dj_db = compute_gradient(x, y, w, b)

    for j in range(n):
        dj_dw[j] += (lambda_ / m) * w[j]

    return dj_dw, dj_db


# ============================================================
# 9. GRADIENT DESCENT
# ============================================================



def Gradient_descent(x,y,w_in,b_in,alpha,num_iter,lambda_):
    m,n=x.shape
    j_history =[]
    w_history =[]
    for i in range(num_iter):
    # Calculate the gradient and update the parameters
        dj_dw ,dj_db =regularized_gradient(x,y,w_in,b_in,lambda_) 


        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db
        if i<10000:
            cost= compute_cost_logistic(x,y,w_in,b_in)
            j_history.append(cost)

        if i% math.ceil(num_iter/10) == 0 or i == (num_iter-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(j_history[-1]):8.2f}   ")

    return w_in,b_in,j_history    

# ============================================================
# 10. PREDICTION
# ============================================================

def predict_probability(x, w, b):
    z = np.dot(x, w) + b
    return sigmoid(z)

def predict(x,w,b,threshold=0.5):
    probability = predict_probability(x,w,b)
    predictions = (probability >= threshold).astype(int)
    return predictions


#print("predictions: ", predict(x_train, w_init, b_init))

def compute_accuracy(y_true,y_predict):
    return np.mean (y_true == y_predict) * 100
# ============================================================
# 11. TEST INITIAL FUNCTIONS
# ============================================================

n = x_train_norm.shape[1]
w_init = np.zeros(n)
b_init = 0.0

initial_cost = compute_cost_logistic(x_train_norm,y_train,w_init,b_init)
initial_dj_dw , initial_dj_db = compute_gradient(x_train_norm,y_train,w_init,b_init)

print("\nInitial tests")
print("initial cost:", initial_cost)
print("initial dw_db:", initial_dj_dw)
print("initial dj_db:", initial_dj_db)


# ============================================================
# 12. TRAIN MODEL
# ============================================================

alpha = 0.01
num_iteration = 300
lambda_= 0.1
w_final, b_final,J_history =Gradient_descent(x_train_norm,
                                              y_train,
                                            w_init,
                                              b_init,
                                              alpha,
                                              num_iteration,
                                              0.1)

print("\nFinal parameters")
print("w_final:", w_final)
print("b_final:", b_final)

# ============================================================
# 13. EVALUATE MODEL
# ============================================================

train_predictions = predict(x_train_norm,
                            w_final,
                            b_final)
test_predictions = predict(x_test_norm,
                           w_final,
                           b_final)
train_accuracy = compute_accuracy(y_train,
                                  train_predictions)
test_accuracy = compute_accuracy(y_test,test_predictions)

print("Train accuracy: ", train_accuracy)
print("Test accuracy: ",test_accuracy)

# ============================================================
# 14. PLOT COST
# ============================================================
plt.figure(figsize=(8,5))
plt.plot(J_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Logistic Regression Cost During Training")
plt.grid(True)
plt.show()