import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk, messagebox

class NeuralNetwork:
    def __init__(self, layers, learning_rate, use_bias, activation_fn):
        self.layers = layers
        self.learning_rate = learning_rate
        self.use_bias = use_bias
        self.activation_fn = activation_fn
        
        self.weights = []
        self.biases = []
        
        #Xavier (or Glorot) initialization
        for i in range(len(layers) - 1):
            limit = np.sqrt(6 / (layers[i] + layers[i+1]))
            w = np.random.uniform(-limit, limit, (layers[i], layers[i+1]))
            self.weights.append(w)
            
            b = np.zeros((1, layers[i+1]))

            self.biases.append(b)

    #forward formula:
    def _activation(self, x):
        if self.activation_fn == 'sigmoid':
            return np.where(x >= 0, 
                          1 / (1 + np.exp(-x)),
                          np.exp(x) / (1 + np.exp(x))) #an equivalent formula that handles large negative numbers safely
        #as sigmoid fails with very negative numbers, causing overflow
        elif self.activation_fn == 'tanh':
            return np.tanh(x) #i guess this will be removed, built in, but is it a must?
        
    #backward derivative:
    def _activation_derivative(self, x): #x here is the acivated output (e.g. x=tanh(x))
        if self.activation_fn == 'sigmoid':
            return x * (1 - x)
        elif self.activation_fn == 'tanh':
            return 1.0 - x**2 #avoids calculating expensive exponentials during backprop
    
    #multi-class classification (3 classes) -> softmax, unlike sigmoid (binary)
    #forces the sum of all outputs to equal 1.0 (probability)
    #It pushes the values apart, suppressing the weak classes and highlighting the strong one
    def _softmax(self, z):
        exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True)) #shift invariant property
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
    
    #stochastic gradient descent (anxious)
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
    #batch (patient)
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

    #stochastic adaptive moment estimation(smart)
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

#preprocessing
def load_and_preprocess_data():
    df = pd.read_csv(r'penguins.csv')
    #null filling
    numerical_cols = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']
    categorical_cols = ['Species', 'OriginLocation']
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

    #One-Hot encoding
    encoded_df = df.copy()
    # --- DICTIONARY TO SAVE THE EXACT ORDER ---
    cat_mappings = {} 
    for col in categorical_cols:
        unique_values = df[col].dropna().unique()
        #this specific order saved to send to the GUI later
        cat_mappings[col] = unique_values
        
        for val in unique_values:
            new_col_name = f"{col}_{val}"
            encoded_df[new_col_name] = (df[col] == val).astype(int)
        
        encoded_df.drop(columns=[col], inplace=True)
    target_cols = encoded_df.filter(like='Species_').columns
    X = encoded_df.drop(columns=target_cols).values
    y = encoded_df[target_cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=np.argmax(y, axis=1)
    )
    return X_train, X_test, y_train, y_test, cat_mappings


#normalization (standardization/Z-score normalization)
#for neural network to converge
def scale_data(X_train, X_test):
    #identify and scale numerical columns only (indices where values are not just 0 and 1)
    numerical_indices = []
    for i in range(X_train.shape[1]):
        col_values = np.unique(X_train[:, i])
        if not np.all(np.isin(col_values, [0, 1])):
            numerical_indices.append(i)

    mean = np.mean(X_train[:, numerical_indices], axis=0)
    std = np.std(X_train[:, numerical_indices], axis=0) + 1e-8 #a small epsilon to avoid div by zero

    X_train[:, numerical_indices] = (X_train[:, numerical_indices] - mean) / std
    #transforming X_test using x_mean & x_std from X_train to prevent data leakage
    X_test[:, numerical_indices] = (X_test[:, numerical_indices] - mean) / std
    return X_train, X_test, numerical_indices, mean, std

# if __name__ == "__main__":

#     #processed X train and test
#     X_train, X_test = scale_numerical_only(X_train, X_test)

#     input_size = X.shape[1]
#     hidden_size = 8
#     output_size = y.shape[1]
    
#     nn = NeuralNetwork(
#         layers=[input_size, hidden_size, output_size],
#         learning_rate=0.01,
#         use_bias=True,
#         activation_fn='sigmoid'
#     )

#     nn.train_adam(X_train, y_train, epochs=1000)
#     accuracy = nn.evaluate(X_test, y_test)
#     print(f"\nTraining Accuracy: {accuracy:.4f}")
    
#     # Make predictions
#     # predictions = nn.predict(y_train.iloc[30])
#     probabilities = nn.predict_proba(X_test)   
#     print(probabilities) 
#     # print("Neural Network class ready to use!")

















# #interface:
# features = df.drop(columns='Species', axis=1)
# label = df['Species']

# root = tk.Tk()
# root.title("Backpropagation Algorithm")
# root.geometry("950x600")

# root.columnconfigure(0, weight=1)
# root.columnconfigure(1, weight=1)
# root.rowconfigure(0, weight=1)

# hyper_parameters_frame = ttk.LabelFrame(root, text="1. User Input", padding=(10, 5))
# hyper_parameters_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# ttk.Label(hyper_parameters_frame, text="Number of hidden layers:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
# hidden_layers_entry = ttk.Entry(hyper_parameters_frame)
# hidden_layers_entry.insert(0, "2") #default value
# hidden_layers_entry.grid(row=0, column=1, padx=10, pady=5)

# #note: Since there are multiple layers, we usually enter this as a comma-separated list
# ttk.Label(hyper_parameters_frame, text="Neurons in each layer (e.g. 5,4):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
# neurons_entry = ttk.Entry(hyper_parameters_frame)
# neurons_entry.insert(0, "5, 4") #default values for 2 layers
# neurons_entry.grid(row=1, column=1, padx=10, pady=5)

# ttk.Label(hyper_parameters_frame, text="Learning Rate (eta):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
# lr_entry = ttk.Entry(hyper_parameters_frame)
# lr_entry.insert(0, "0.01")
# lr_entry.grid(row=2, column=1, padx=10, pady=5)

# ttk.Label(hyper_parameters_frame, text="Epochs (m):").grid(row=3, column=0, padx=10, pady=5, sticky="w")
# epochs_entry = ttk.Entry(hyper_parameters_frame)
# epochs_entry.insert(0, "100")
# epochs_entry.grid(row=3, column=1, padx=10, pady=5)

# #checkbox
# bias_var = tk.BooleanVar(value=True)
# ttk.Checkbutton(hyper_parameters_frame, text="Include Bias", variable=bias_var).grid(row=4, column=0, columnspan=2, pady=5)

# #smart variable
# ttk.Label(hyper_parameters_frame, text="Activation Function:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
# activation_var = tk.StringVar(value="Sigmoid")

# #the dropdown
# activation_menu = ttk.OptionMenu(
#     hyper_parameters_frame, 
#     activation_var, 
#     "Sigmoid",            #default value to show
#     "Sigmoid",            #option 1
#     "Hyperbolic Tangent"  #option 2
# )
# activation_menu.grid(row=5, column=1, padx=10, pady=5, sticky="ew")

# #eval frame
# model_evaluation = ttk.LabelFrame(root, text="4. Model Evaluation", padding=(10, 5))
# model_evaluation.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

# Training_result = tk.StringVar(value="N/A")
# testing_result  = tk.StringVar(value="N/A")

# ttk.Label(model_evaluation, text="Training Accuracy:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
# ttk.Label(model_evaluation, textvariable=Training_result, foreground="blue").grid(row=0, column=1, padx=10, pady=5, sticky="w")

# ttk.Label(model_evaluation, text="Testing Accuracy:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
# ttk.Label(model_evaluation, textvariable=testing_result, foreground="blue").grid(row=1, column=1, padx=10, pady=5, sticky="w")

# # Confusion Matrix Button (New!)
# # This button will trigger the plotting function you imported earlier - to be added
# cm_button = ttk.Button(model_evaluation, text="Show Confusion Matrix", command=lambda: print("Link this to your plot_cm function")) 
# cm_button.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")

# #new sample frame
# classify_frame = ttk.LabelFrame(root, text="5. Classify New Sample", padding=(10, 5))
# classify_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

# #list to store the entry widgets so we can read them later
# feature_entry_widgets = []

# #5 input fields
# for i in range(5):
#     ttk.Label(classify_frame, text=f"Feature {i+1}:").grid(row=i, column=0, padx=10, pady=5, sticky="w")
    
#     entry = ttk.Entry(classify_frame)
#     entry.insert(0, "0.0") #default value
#     entry.grid(row=i, column=1, padx=10, pady=5)
#     feature_entry_widgets.append(entry)

# classification_result = tk.StringVar(value="Not classified yet")

# # --- Prediction Logic Helper ---
# def on_classify_click():
#     try:
#         # 1. Loop through the 5 widgets and get the numbers
#         user_inputs = []
#         for entry in feature_entry_widgets:
#             val = float(entry.get())
#             user_inputs.append(val)
        
#         # 2. Convert to NumPy array (Shape: 1 row, 5 cols)
#         sample_vector = np.array(user_inputs).reshape(1, -1)
        
#         # 3. Run prediction (Assume 'nn' is your trained NeuralNetwork instance)
#         #preprocess this sample YA MARIAM
#         activations = nn.forward(sample_vector)
#         final_probs = activations[-1] # This will be an array like [[0.1, 0.8, 0.1]]
        
#         # 4. Get the winning class (0, 1, or 2)
#         predicted_index = np.argmax(final_probs)
        
#         # 5. Update the GUI
#         classification_result.set(f"Class {predicted_index}")
        
#     except ValueError:
#         messagebox.showerror("Input Error", "Please ensure all 5 feature inputs are valid numbers.")
#     except NameError:
#         messagebox.showerror("Model Error", "Please train the model first.")

# # Button to trigger the logic
# classify_button = ttk.Button(classify_frame, text="Predict Class", command=on_classify_click)
# # Place button at row 5 (since rows 0-4 are taken by inputs)
# classify_button.grid(row=5, column=0, columnspan=2, pady=10)

# # Result Labels
# ttk.Label(classify_frame, text="Prediction:").grid(row=6, column=0, padx=10, pady=5, sticky="w")
# ttk.Label(classify_frame, textvariable=classification_result, foreground="blue", font=("Arial", 10, "bold")).grid(row=6, column=1, padx=10, pady=5, sticky="w")
