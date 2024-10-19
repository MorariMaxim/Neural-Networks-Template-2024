import numpy as np
import pickle  

def softmax(Z):
    Z_exp = np.exp(Z)
    softmax_output = Z_exp / np.sum(Z_exp, axis=1, keepdims=True)
    return softmax_output

def cross_entropy_loss(predictions, labels):
    return -np.mean(np.sum(labels * np.log(predictions + 1e-8), axis=1))

def compute_accuracy(predictions, labels):
    return np.mean(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))


with open('mnist_data.pkl', 'rb') as f:
    train_X, train_Y, test_X, test_Y = pickle.load(f)
    
    train_X = np.array([array / 255.0 for array in train_X])  
    test_X = np.array([array / 255.0 for array in test_X])  
    
    train_Y = np.eye(10)[train_Y]
    test_Y = np.eye(10)[test_Y]

    input_size = 784  
    output_size = 10  
    learning_rate = 0.01
    epochs = 100
    batch_size = 100
    
    rng = np.random.default_rng(42)   
    W = rng.random((input_size, output_size)) * 0.01   
    b = np.zeros((output_size,))
 
    for epoch in range(epochs):
        
        for i in range(0, len(train_X), batch_size):
            X_batch = train_X[i:i + batch_size]   
            y_batch = train_Y[i:i + batch_size]
            
            # Forward propagation
            Z = np.dot(X_batch, W) + b # (batch_size, 784) * (784, 10) = (100,10) 
            predictions = softmax(Z)
                                    
            loss = cross_entropy_loss(predictions, y_batch) 
            
            # - (Target - y); gradient loss
            dZ = predictions - y_batch # 100, 10
            
            #X^T * -(Target - y); weight gradient
            dW = np.dot(X_batch.T, dZ) / batch_size 
            
            # bias gradient
            db = np.sum(dZ, axis=0) / batch_size
            
            # Update weights and biases using gradient descent
            W -= learning_rate * dW
            b -= learning_rate * db
        
        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            train_accuracy = compute_accuracy(softmax(np.dot(train_X, W) + b), train_Y)
            test_accuracy = compute_accuracy(softmax(np.dot(test_X, W) + b), test_Y)
            print(f'Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Train Accuracy: {train_accuracy*100:.2f}% - Test Accuracy: {test_accuracy*100:.2f}%')

    # Final evaluation
    final_train_accuracy = compute_accuracy(softmax(np.dot(train_X, W) + b), train_Y)
    final_test_accuracy = compute_accuracy(softmax(np.dot(test_X, W) + b), test_Y)
    print(f'Final Train Accuracy: {final_train_accuracy*100:.2f}%')
    print(f'Final Test Accuracy: {final_test_accuracy*100:.2f}%')