import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
class NeuralNetwork:
    def __init__(self, layers, learning_rate, use_bias, activation_fn):
        self.layers = layers
        self.learning_rate = learning_rate
        self.use_bias = use_bias
        self.activation_fn = activation_fn
        
        self.weights = []
        self.biases = []
        
        for i in range(len(layers) - 1):
            limit = np.sqrt(6 / (layers[i] + layers[i+1]))
            w = np.random.uniform(-limit, limit, (layers[i], layers[i+1]))
            self.weights.append(w)
            
            b = np.zeros((1, layers[i+1]))

            self.biases.append(b)
    
    def _activation(self, x):
        if self.activation_fn == 'sigmoid':
            return np.where(x >= 0, 
                          1 / (1 + np.exp(-x)),
                          np.exp(x) / (1 + np.exp(x)))
        elif self.activation_fn == 'tanh':
            return np.tanh(x)
    
    def _activation_derivative(self, x):
        if self.activation_fn == 'sigmoid':
            return x * (1 - x)
        elif self.activation_fn == 'tanh':
            return 1.0 - x**2
    
    def _softmax(self, z):
        exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def forward(self, x):
        activations = [x.reshape(1, -1)]
        current_input = x.reshape(1, -1)
        
        for i in range(len(self.weights) - 1):
            net_value = np.dot(current_input, self.weights[i]) + self.biases[i]
            output = self._activation(net_value)
            activations.append(output)
            current_input = output
        
        net_value = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        output = self._softmax(net_value)
        activations.append(output)
        
        return activations
    
    def _cross_entropy_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(y_pred))
    
    def train_sgd(self, X, y, epochs, print_every=100):
        print(f"\nTraining for {epochs} epochs (SGD)...")
        
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(len(X)):
                input_data = X[i]
                target = y[i].reshape(1, -1)
                
                activations = self.forward(input_data)
                final_output = activations[-1]
                
                total_loss += self._cross_entropy_loss(target, final_output)
                
                output_delta = final_output - target
                deltas = [output_delta]
                
                for j in range(len(self.weights) - 1, 0, -1):
                    hidden_error = deltas[-1].dot(self.weights[j].T)
                    delta = hidden_error * self._activation_derivative(activations[j])
                    deltas.append(delta)
                
                deltas.reverse()
                
                for j in range(len(self.weights)):
                    layer_input = activations[j]
                    delta = deltas[j]
                    self.weights[j] -= self.learning_rate * layer_input.T.dot(delta)
                    if self.use_bias:
                        self.biases[j] -= self.learning_rate * delta
            
            if (epoch + 1) % print_every == 0:
                avg_loss = total_loss / len(X)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    def train_bgd(self, X, y, epochs, print_every=100):

        print(f"\nTraining for {epochs} epochs (BGD)...")
        
        num_layers = len(self.weights)
        
        for epoch in range(epochs):
            total_loss = 0
            grad_W = [np.zeros_like(w) for w in self.weights]
            grad_B = [np.zeros_like(b) for b in self.biases]
            
            for i in range(len(X)):
                input_data = X[i]
                target = y[i].reshape(1, -1)
                
                activations = self.forward(input_data)
                final_output = activations[-1]
                
                total_loss += self._cross_entropy_loss(target, final_output)
                
                output_delta = final_output - target
                deltas = [output_delta]
                
                for j in range(num_layers - 1, 0, -1):
                    hidden_error = deltas[-1].dot(self.weights[j].T)
                    delta = hidden_error * self._activation_derivative(activations[j])
                    deltas.append(delta)
                
                deltas.reverse()
                
                for j in range(num_layers):
                    layer_input = activations[j]
                    delta = deltas[j]
                    grad_W[j] += layer_input.T.dot(delta)
                    if self.use_bias:
                        grad_B[j] += delta
            
            scale_factor = self.learning_rate / len(X)
            
            for j in range(num_layers):
                self.weights[j] -= scale_factor * grad_W[j]
                if self.use_bias:
                    self.biases[j] -= scale_factor * grad_B[j]
            
            if (epoch + 1) % print_every == 0:
                avg_loss = total_loss / len(X)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    def train_adam(self, X, y, epochs, beta1=0.9, beta2=0.999, epsilon=1e-8, print_every=100):

        print(f"\nTraining for {epochs} epochs using Adam...")
        
        num_layers = len(self.weights)
        
        mW = [np.zeros_like(w) for w in self.weights]
        vW = [np.zeros_like(w) for w in self.weights]
        mb = [np.zeros_like(b) for b in self.biases]
        vb = [np.zeros_like(b) for b in self.biases]
        
        t = 0
        
        for epoch in range(epochs):
            total_loss = 0
            grad_W = [np.zeros_like(w) for w in self.weights]
            grad_B = [np.zeros_like(b) for b in self.biases]
            
            for i in range(len(X)):
                input_data = X[i]
                target = y[i].reshape(1, -1)
                
                activations = self.forward(input_data)
                final_output = activations[-1]
                
                total_loss += self._cross_entropy_loss(target, final_output)
                
                output_delta = final_output - target
                deltas = [output_delta]
                
                for j in range(num_layers - 1, 0, -1):
                    hidden_error = deltas[-1].dot(self.weights[j].T)
                    delta = hidden_error * self._activation_derivative(activations[j])
                    deltas.append(delta)
                
                deltas.reverse()
                
                for j in range(num_layers):
                    grad_W[j] += activations[j].T.dot(deltas[j])
                    if self.use_bias:
                        grad_B[j] += deltas[j]
            
            t += 1
            
            for j in range(num_layers):
                g_W = grad_W[j] / len(X)
                g_B = grad_B[j] / len(X)
                
                mW[j] = beta1 * mW[j] + (1 - beta1) * g_W
                vW[j] = beta2 * vW[j] + (1 - beta2) * (g_W ** 2)
                
                m_hat_W = mW[j] / (1 - beta1 ** t)
                v_hat_W = vW[j] / (1 - beta2 ** t)
                
                self.weights[j] -= self.learning_rate * m_hat_W / (np.sqrt(v_hat_W) + epsilon)
                
                if self.use_bias:
                    mb[j] = beta1 * mb[j] + (1 - beta1) * g_B
                    vb[j] = beta2 * vb[j] + (1 - beta2) * (g_B ** 2)
                    
                    m_hat_B = mb[j] / (1 - beta1 ** t)
                    v_hat_B = vb[j] / (1 - beta2 ** t)
                    
                    self.biases[j] -= self.learning_rate * m_hat_B / (np.sqrt(v_hat_B) + epsilon)
            
            if (epoch + 1) % print_every == 0:
                avg_loss = total_loss / len(X)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            activations = self.forward(X[i])
            predictions.append(np.argmax(activations[-1]))
        return np.array(predictions)
    
    def predict_proba(self, X):
        probabilities = []
        for i in range(len(X)):
            activations = self.forward(X[i])
            probabilities.append(activations[-1].flatten())
        return np.array(probabilities)
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        return accuracy


# Example usage:
if __name__ == "__main__":
    # Example: Create sample data
    df = pd.read_csv(r'penguins.csv')
    categorical_cols = ['Species', 'OriginLocation']
    numerical_cols = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']
    
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

    encoded_df = df.copy()

    for col in categorical_cols:
        unique_values = df[col].dropna().unique()
        
        for val in unique_values:
            new_col_name = f"{col}_{val}"
            encoded_df[new_col_name] = (df[col] == val).astype(int)
        
        encoded_df.drop(columns=[col], inplace=True)


    for col in numerical_cols:
        mu = encoded_df[col].mean()
        
        sigma = encoded_df[col].std()
        
        if sigma != 0:
            encoded_df[col] = (encoded_df[col] - mu) / sigma
        else:
            encoded_df[col] = 0.0

    df = encoded_df
    del encoded_df

    target_cols = df.filter(like='Species_').columns
    X = df.drop(target_cols, axis=1).values
    y = df[target_cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    input_size = X.shape[1]
    hidden_size = 8
    output_size = y.shape[1]
    
    nn = NeuralNetwork(
        layers=[input_size, hidden_size, output_size],
        learning_rate=0.01,
        use_bias=True,
        activation_fn='sigmoid'
    )
    
    # Train the network
    nn.train_adam(X_train, y_train, epochs=1000)
    
    # Evaluate
    accuracy = nn.evaluate(X_test, y_test)
    print(f"\nTraining Accuracy: {accuracy:.4f}")
    
    # Make predictions
    # predictions = nn.predict(y_train.iloc[30])
    probabilities = nn.predict_proba(X_test)   
    print(probabilities) 
    # print("Neural Network class ready to use!")