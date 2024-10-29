import numpy as np
import pickle  
import time

def softmax(Z):
    Z_exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    softmax_output = Z_exp / np.sum(Z_exp, axis=1, keepdims=True)
    return softmax_output

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0

def cross_entropy_loss(predictions, labels):
    return -np.mean(np.sum(labels * np.log(predictions + 1e-8), axis=1))

def compute_accuracy(predictions, labels):
    return np.mean(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))

def dropout(X, drop_prob): 
    mask = np.random.rand(*X.shape) < drop_prob  
    return X * mask / drop_prob  


start_time = time.time()

with open('mnist_data.pkl', 'rb') as f:
    train_X, train_Y, test_X, test_Y = pickle.load(f)
    
    train_X = np.array([array / 255.0 for array in train_X])  
    test_X = np.array([array / 255.0 for array in test_X])  
    
    train_Y = np.eye(10)[train_Y]
    test_Y = np.eye(10)[test_Y]

    input_size = 784  
    hidden_size = 100
    output_size = 10  
    learning_rate = 0.01
    epochs = 100
    batch_size = 100

    drop_prob = 0.95

    plateau_patience = 5  
    decay_factor = 0.1  
    min_lr = 1e-6  
    best_loss = float('inf')
    epochs_since_improvement = 0
    
    rng = np.random.default_rng(21)   
    W1 = rng.random((input_size, hidden_size)) * 0.01   
    b1 = np.zeros((hidden_size,))
    W2 = rng.random((hidden_size, output_size)) * 0.01   
    b2 = np.zeros((output_size,))
 
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(train_X), batch_size):
            X_batch = train_X[i:i + batch_size]   
            y_batch = train_Y[i:i + batch_size]
            
            
            Z1 = np.dot(X_batch, W1) + b1  #(batch_size, hidden_size)
            A1 = relu(Z1)  
            A1 = dropout(A1, drop_prob)  #dropout
            Z2 = np.dot(A1, W2) + b2  # (batch_size, output_size)
            A2 = softmax(Z2)  
                                    
            loss = cross_entropy_loss(A2, y_batch) 
            epoch_loss += loss
            
            
            dZ2 = A2 - y_batch # (batch_size, output_size)

            # need (hidden_size, output_size); 
            # ! (batch_size, hidden_size) * (batch_size, output_size)
            # transpose: (hidden_size, batch_size) * (batch_size, output_size) = (hidden_size, output_size)
            dW2 = np.dot(A1.T, dZ2) / batch_size

            db2 = np.sum(dZ2, axis=0) / batch_size
            
            # (batch_size, output_size) * (hidden_size, output_size) != (batch_size, hidden_size)
            # (batch_size, output_size) * (output_size, hidden_size) = (batch_size, hidden_size)
            # element wise (batch_size, hidden_size) * (batch_size, hidden_size) 
            dZ1 = np.dot(dZ2, W2.T) * relu_derivative(Z1) * (A1 > 0)  #dropout 


            # (batch_size, input_size) * (batch_size, hidden_size) != (input_size, hidden_size)
            # (input_size, batch_size) * (batch_size, hidden_size) = (input_size, hidden_size)
            dW1 = np.dot(X_batch.T, dZ1) / batch_size  
            db1 = np.sum(dZ1, axis=0) / batch_size  

            
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
        
        epoch_loss /= (len(train_X) // batch_size)
                        
        # scheduler
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= plateau_patience:
            learning_rate = max(learning_rate * decay_factor, min_lr) 
            print(f'Learning rate decayed to: {learning_rate}')

        if epoch % 10 == 9 or epoch == epochs - 1:            
            Z1_train = np.dot(train_X, W1) + b1
            A1_train = relu(Z1_train)
            train_predictions = softmax(np.dot(A1_train, W2) + b2)
            train_accuracy = compute_accuracy(train_predictions, train_Y)
            
            Z1_test = np.dot(test_X, W1) + b1
            A1_test = relu(Z1_test)
            test_predictions = softmax(np.dot(A1_test, W2) + b2)
            test_accuracy = compute_accuracy(test_predictions, test_Y)
            loss = cross_entropy_loss(test_predictions,test_Y)
            
            print(f'Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Train Accuracy: {train_accuracy*100:.2f}% - Test Accuracy: {test_accuracy*100:.2f}%')
            print(f"{time.time() - start_time} time elapsed")
    
    final_train_accuracy = compute_accuracy(train_predictions, train_Y)
    final_test_accuracy = compute_accuracy(test_predictions, test_Y)
    print(f'Final Train Accuracy: {final_train_accuracy*100:.2f}%')
    print(f'Final Test Accuracy: {final_test_accuracy*100:.2f}%')
    print(f"{time.time() - start_time} time elapsed")
