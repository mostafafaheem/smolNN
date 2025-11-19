import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import task_2

trained_model = None
scaler_stats = {} #to hold mean, std, and indices for scaling new samples
saved_cat_mappings = {}

root = tk.Tk()
root.title("Backpropagation Algorithm")
root.geometry("950x650")

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.rowconfigure(0, weight=1)

#hyperparameters Frame
hyper_parameters_frame = ttk.LabelFrame(root, text="1. User Input", padding=(10, 5))
hyper_parameters_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

ttk.Label(hyper_parameters_frame, text="Number of hidden layers:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
hidden_layers_entry = ttk.Entry(hyper_parameters_frame)
hidden_layers_entry.insert(0, "2")
hidden_layers_entry.grid(row=0, column=1, padx=10, pady=5)

ttk.Label(hyper_parameters_frame, text="Neurons (e.g. 5,4):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
neurons_entry = ttk.Entry(hyper_parameters_frame)
neurons_entry.insert(0, "5, 4")
neurons_entry.grid(row=1, column=1, padx=10, pady=5)

ttk.Label(hyper_parameters_frame, text="Learning Rate (eta):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
lr_entry = ttk.Entry(hyper_parameters_frame)
lr_entry.insert(0, "0.01")
lr_entry.grid(row=2, column=1, padx=10, pady=5)

ttk.Label(hyper_parameters_frame, text="Epochs (m):").grid(row=3, column=0, padx=10, pady=5, sticky="w")
epochs_entry = ttk.Entry(hyper_parameters_frame)
epochs_entry.insert(0, "1000")
epochs_entry.grid(row=3, column=1, padx=10, pady=5)

bias_var = tk.BooleanVar(value=True)
ttk.Checkbutton(hyper_parameters_frame, text="Include Bias", variable=bias_var).grid(row=4, column=0, columnspan=2, pady=5)

ttk.Label(hyper_parameters_frame, text="Activation Function:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
activation_var = tk.StringVar(value="sigmoid")
activation_menu = ttk.OptionMenu(hyper_parameters_frame, activation_var, "sigmoid", "sigmoid", "tanh")
activation_menu.grid(row=5, column=1, padx=10, pady=5, sticky="ew")

def on_train_click():
    global trained_model, scaler_stats
    
    try:
        #parsing inputs
        n_layers = int(hidden_layers_entry.get())
        neurons = [int(x.strip()) for x in neurons_entry.get().split(',')]
        lr = float(lr_entry.get())
        epochs = int(epochs_entry.get())
        use_bias = bias_var.get()
        act_fn = activation_var.get()

        if len(neurons) != n_layers:
            raise ValueError(f"Layer count ({n_layers}) matches neuron count ({len(neurons)}).")

        #processing data
        X_train, X_test, y_train, y_test, mappings = task_2.load_and_preprocess_data()
        saved_cat_mappings = mappings
        #scaling
        X_train, X_test, num_indices, mean, std = task_2.scale_data(X_train, X_test)
        scaler_stats = {'indices': num_indices, 'mean': mean, 'std': std}

        #model initialization
        input_size = X_train.shape[1]
        output_size = y_train.shape[1]
        #construct layer list
        layer_architecture = [input_size] + neurons + [output_size]
        
        trained_model = task_2.NeuralNetwork(
            layers=layer_architecture,
            learning_rate=lr,
            use_bias=use_bias,
            activation_fn=act_fn
        )

        #train
        trained_model.train_adam(X_train, y_train, epochs=epochs)
        #evaluate
        train_acc = trained_model.evaluate(X_train, y_train)
        test_acc = trained_model.evaluate(X_test, y_test)
        
        Training_result.set(f"{train_acc * 100:.2f}%")
        testing_result.set(f"{test_acc * 100:.2f}%")
        messagebox.showinfo("Success", "Training Completed Successfully!")

    except Exception as e:
        messagebox.showerror("Error", str(e))

run_button = ttk.Button(hyper_parameters_frame, text="Start Training", command=on_train_click)
run_button.grid(row=6, column=0, columnspan=2, pady=20, sticky="ew")


#model eval frame
model_evaluation = ttk.LabelFrame(root, text="4. Model Evaluation", padding=(10, 5))
model_evaluation.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

Training_result = tk.StringVar(value="N/A")
testing_result  = tk.StringVar(value="N/A")

ttk.Label(model_evaluation, text="Training Accuracy:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
ttk.Label(model_evaluation, textvariable=Training_result, foreground="blue").grid(row=0, column=1, padx=10, pady=5, sticky="w")
ttk.Label(model_evaluation, text="Testing Accuracy:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
ttk.Label(model_evaluation, textvariable=testing_result, foreground="blue").grid(row=1, column=1, padx=10, pady=5, sticky="w")
cm_button = ttk.Button(model_evaluation, text="Show Confusion Matrix", command=lambda: print("Add plot code here")) 
cm_button.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")


#classification frame
classify_frame = ttk.LabelFrame(root, text="5. Classify New Sample", padding=(10, 5))
classify_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
feature_entry_widgets = []
feature_names = ["Culmen Length", "Culmen Depth", "Flipper Length", "Origin_Location", "Body Mass"]

for i in range(5):
    # Try to name them nicely if possible, else Feature i
    label_text = feature_names[i] if i < len(feature_names) else f"Feature {i+1}"
    ttk.Label(classify_frame, text=f"{label_text}:").grid(row=i, column=0, padx=10, pady=5, sticky="w")
    entry = ttk.Entry(classify_frame)
    entry.insert(0, "0.0")
    entry.grid(row=i, column=1, padx=10, pady=5)
    feature_entry_widgets.append(entry)

classification_result = tk.StringVar(value="Not classified yet")

def on_classify_click():
    if trained_model is None:
        messagebox.showerror("Error", "Please train the model first!")
        return

    try:
        # 1. READ RAW INPUTS
        # [Length, Depth, Flipper, Origin, Mass]
        raw_L = float(feature_entry_widgets[0].get())
        raw_D = float(feature_entry_widgets[1].get())
        raw_F = float(feature_entry_widgets[2].get())
        raw_Origin = feature_entry_widgets[3].get() # e.g., "Dream"
        raw_M = float(feature_entry_widgets[4].get())

        # 2. SCALE NUMERICALS (Same as before)
        mean = scaler_stats['mean'] 
        std = scaler_stats['std']
        
        # Assume backend order: [L, D, F, M] -> Indices 0, 1, 2, 3
        scaled_L = (raw_L - mean[0]) / std[0]
        scaled_D = (raw_D - mean[1]) / std[1]
        scaled_F = (raw_F - mean[2]) / std[2]
        scaled_M = (raw_M - mean[3]) / std[3]

        # 3. ENCODE CATEGORICAL (DYNAMICALLY)
        # Retrieve the order found during training (e.g. ['Torgersen', 'Dream', 'Biscoe'])
        origin_order_list = saved_cat_mappings['OriginLocation']
        # Create a zero vector size of how many origins we have
        origin_one_hot = [0] * len(origin_order_list)
        
        # Find where the user's choice lives in that list
        if raw_Origin in origin_order_list:
            # NumPy/Pandas unique() returns an ndarray, so we use np.where or convert to list
            idx = list(origin_order_list).index(raw_Origin)
            origin_one_hot[idx] = 1
        else:
            # Fallback if user selected something weird or list is empty
            #pass
            messagebox.showwarning("Warning", f"Unknown location: {raw_Origin}") 

        # 4. ASSEMBLE
        # [Numericals] + [One-Hots]
        # Note: Ensure Mass is inserted in correct spot if needed, or append at end
        # Based on your backend loop, numericals usually come first in X unless columns were reordered.
        # If you used `df[numerical_cols] = ...` and then `encoded_df = df.copy()`, 
        # the numerical columns are usually on the LEFT and new encoded columns added to the RIGHT.
        
        final_input_list = [scaled_L, scaled_D, scaled_F, scaled_M] + origin_one_hot
        
        sample_vector = np.array(final_input_list).reshape(1, -1)

        # 5. PREDICT
        activations = trained_model.forward(sample_vector)
        final_probs = activations[-1]
        predicted_index = np.argmax(final_probs)
        
        # Map index to class name
        # We can also use the Saved Species Mapping to be 100% safe!
        species_order = saved_cat_mappings['Species']
        result_text = species_order[predicted_index]
            
        classification_result.set(result_text)

    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))

classify_button = ttk.Button(classify_frame, text="Predict Class", command=on_classify_click)
classify_button.grid(row=5, column=0, columnspan=2, pady=10)

ttk.Label(classify_frame, text="Prediction:").grid(row=6, column=0, padx=10, pady=5, sticky="w")
ttk.Label(classify_frame, textvariable=classification_result, foreground="blue", font=("Arial", 10, "bold")).grid(row=6, column=1, padx=10, pady=5, sticky="w")

root.mainloop()