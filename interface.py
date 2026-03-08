import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import nn

# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# globals
trained_model = None
scaler_stats = {}
saved_cat_mappings = {}
global_X_test = None
global_y_test = None

root = tk.Tk()
root.title("smolNN GUI")
root.geometry("800x500")

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)

# hyperparameters Frame
hyper_parameters_frame = ttk.LabelFrame(root, text="Hyperparameters", padding=(10, 5))
hyper_parameters_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")

ttk.Label(hyper_parameters_frame, text="Number of hidden layers:").grid(
    row=0, column=0, padx=10, pady=5, sticky="w"
)
hidden_layers_entry = ttk.Entry(hyper_parameters_frame)
hidden_layers_entry.insert(0, "2")
hidden_layers_entry.grid(row=0, column=1, padx=10, pady=5)

ttk.Label(hyper_parameters_frame, text="Neurons (e.g. 5,4):").grid(
    row=1, column=0, padx=10, pady=5, sticky="w"
)
neurons_entry = ttk.Entry(hyper_parameters_frame)
neurons_entry.insert(0, "5, 4")
neurons_entry.grid(row=1, column=1, padx=10, pady=5)

ttk.Label(hyper_parameters_frame, text="Learning Rate (eta):").grid(
    row=2, column=0, padx=10, pady=5, sticky="w"
)
lr_entry = ttk.Entry(hyper_parameters_frame)
lr_entry.insert(0, "0.01")
lr_entry.grid(row=2, column=1, padx=10, pady=5)

ttk.Label(hyper_parameters_frame, text="Epochs (m):").grid(
    row=3, column=0, padx=10, pady=5, sticky="w"
)
epochs_entry = ttk.Entry(hyper_parameters_frame)
epochs_entry.insert(0, "1000")
epochs_entry.grid(row=3, column=1, padx=10, pady=5)

bias_var = tk.BooleanVar(value=True)
ttk.Checkbutton(hyper_parameters_frame, text="Include Bias", variable=bias_var).grid(
    row=4, column=0, columnspan=2, pady=5
)

ttk.Label(hyper_parameters_frame, text="Activation Function:").grid(
    row=5, column=0, padx=10, pady=5, sticky="w"
)
activation_var = tk.StringVar(value="sigmoid")
activation_menu = ttk.OptionMenu(
    hyper_parameters_frame, activation_var, "sigmoid", "sigmoid", "tanh"
)
activation_menu.grid(row=5, column=1, padx=10, pady=5, sticky="ew")

ttk.Label(hyper_parameters_frame, text="Optimizer:").grid(
    row=6, column=0, padx=10, pady=5, sticky="w"
)
optimizer_var = tk.StringVar(value="Adam")
optimizer_menu = ttk.OptionMenu(
    hyper_parameters_frame, optimizer_var, "Adam", "Adam", "SGD", "BGD"
)
optimizer_menu.grid(row=6, column=1, padx=10, pady=5, sticky="ew")

def on_train_click():
    global trained_model, scaler_stats, saved_cat_mappings, global_X_test, global_y_test

    try:
        # parsing inputs
        n_layers = int(hidden_layers_entry.get())
        neurons = [int(x.strip()) for x in neurons_entry.get().split(",")]
        lr = float(lr_entry.get())
        epochs = int(epochs_entry.get())
        use_bias = bias_var.get()
        act_fn = activation_var.get()
        optimizer = optimizer_var.get()

        if len(neurons) != n_layers:
            raise ValueError(
                f"Layer count ({n_layers}) matches neuron count ({len(neurons)})."
            )

        # processing data
        X_train, X_test, y_train, y_test, mappings = nn.load_and_preprocess_data()
        saved_cat_mappings = mappings
        # scaling
        X_train, X_test, num_indices, mean, std = nn.scale_data(X_train, X_test)
        scaler_stats = {"indices": num_indices, "mean": mean, "std": std}
        global_X_test = X_test
        global_y_test = y_test
        # model initialization
        input_size = X_train.shape[1]
        output_size = y_train.shape[1]
        # layer list construction
        layer_architecture = [input_size] + neurons + [output_size]

        trained_model = nn.NeuralNetwork(
            layers=layer_architecture,
            learning_rate=lr,
            use_bias=use_bias,
            activation_fn=act_fn,
        )

        # train
        if optimizer == "Adam":
            trained_model.train_adam(X_train, y_train, epochs=epochs)
        elif optimizer == "SGD":
            trained_model.train_sgd(X_train, y_train, epochs=epochs)
        elif optimizer == "BGD":
            trained_model.train_bgd(X_train, y_train, epochs=epochs)
            
        # evaluate
        train_acc = trained_model.evaluate(X_train, y_train)
        test_acc = trained_model.evaluate(X_test, y_test)

        Training_result.set(f"{train_acc * 100:.2f}%")
        testing_result.set(f"{test_acc * 100:.2f}%")
        messagebox.showinfo("Success", "Training Completed Successfully!")

    except Exception as e:
        messagebox.showerror("Error", str(e))


# confusion matrix
def on_show_cm_click():
    if trained_model is None:
        messagebox.showerror("Error", "Please train the model first.")
        return
    # checker
    if global_X_test is None or global_y_test is None:
        messagebox.showerror("Error", "Test data not found. Train the model again.")
        return

    try:
        y_pred = trained_model.predict(global_X_test)
        # to convert true labels (if One-Hot)
        if global_y_test.ndim > 1 and global_y_test.shape[1] > 1:
            y_true = np.argmax(global_y_test, axis=1)
        else:
            y_true = global_y_test

        # from scratch confusion matrix
        cm, classes_found = nn.confusion_matrix_manual(y_true, y_pred)

        if "Species" in saved_cat_mappings:
            species_names = saved_cat_mappings["Species"].tolist()
        else:
            species_names = ["Class 0", "Class 1", "Class 2"]

        nn.plot_manual_cm_graphic(
            cm,
            classes_found,
            "Confusion Matrix (Manual)",
            class_names_map=species_names,
        )

    except Exception as e:
        messagebox.showerror("Plot Error", str(e))


run_button = ttk.Button(
    hyper_parameters_frame, text="Start Training", command=on_train_click
)
run_button.grid(row=7, column=0, columnspan=2, pady=20, sticky="ew")


# model eval frame
model_evaluation = ttk.LabelFrame(root, text="Model Evaluation", padding=(10, 5))
model_evaluation.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

Training_result = tk.StringVar(value="N/A")
testing_result = tk.StringVar(value="N/A")

ttk.Label(model_evaluation, text="Training Accuracy:").grid(
    row=0, column=0, padx=10, pady=5, sticky="w"
)
ttk.Label(model_evaluation, textvariable=Training_result, foreground="blue").grid(
    row=0, column=1, padx=10, pady=5, sticky="w"
)
ttk.Label(model_evaluation, text="Testing Accuracy:").grid(
    row=1, column=0, padx=10, pady=5, sticky="w"
)
ttk.Label(model_evaluation, textvariable=testing_result, foreground="blue").grid(
    row=1, column=1, padx=10, pady=5, sticky="w"
)
# cm_button = ttk.Button(model_evaluation, text="Show Confusion Matrix", command=lambda: print("Add plot code here"))
# cm_button.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")
cm_button = ttk.Button(
    model_evaluation, text="Show Confusion Matrix", command=on_show_cm_click
)
cm_button.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")


# classification frame
classify_frame = ttk.LabelFrame(root, text="Classify New Sample", padding=(10, 5))
classify_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
feature_names = [
    "Culmen Length",
    "Culmen Depth",
    "Flipper Length",
    "Body Mass",
    "Origin_Location",
]
origin_options = ["Biscoe", "Dream", "Torgersen"]
feature_entry_widgets = []

for i, name in enumerate(feature_names):
    ttk.Label(classify_frame, text=f"{name}:").grid(
        row=i, column=0, padx=10, pady=5, sticky="w"
    )
    if name == "Origin_Location":
        entry = ttk.Combobox(classify_frame, values=origin_options, state="readonly")
        entry.current(0)  # default "Biscoe" (the first item)
    else:
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
        raw_L = float(feature_entry_widgets[0].get())  # CulmenLength
        raw_D = float(feature_entry_widgets[1].get())  # CulmenDepth
        raw_F = float(feature_entry_widgets[2].get())  # FlipperLength
        raw_M = float(feature_entry_widgets[3].get())  # BodyMass

        # text (string)
        raw_Origin = feature_entry_widgets[4].get()  # OriginLocation
        mean = scaler_stats["mean"]
        std = scaler_stats["std"]

        # (x - u) / s
        # for the indices to match the order of columns in backend X_train
        scaled_L = (raw_L - mean[0]) / std[0]
        scaled_D = (raw_D - mean[1]) / std[1]
        scaled_F = (raw_F - mean[2]) / std[2]
        scaled_M = (raw_M - mean[3]) / std[3]

        # match the exact One-Hot order the model learned
        if "OriginLocation" not in saved_cat_mappings:
            raise ValueError("Mapping missing. Did you restart without training?")

        origin_order = saved_cat_mappings["OriginLocation"]
        origin_one_hot = [0] * len(origin_order)

        if raw_Origin in origin_order:
            idx = list(origin_order).index(raw_Origin)
            origin_one_hot[idx] = 1
        else:
            # if for some reason the dropdown has a value not in training data
            messagebox.showwarning(
                "Warning", f"Location '{raw_Origin}' was never seen during training."
            )

        # combining [Scaled Numbers] + [One Hot Vector]
        final_input_list = [scaled_L, scaled_D, scaled_F, scaled_M] + origin_one_hot
        sample_vector = np.array(final_input_list).reshape(1, -1)
        activations = trained_model.forward(sample_vector)
        final_probs = activations[-1]
        predicted_index = np.argmax(final_probs)

        if "Species" in saved_cat_mappings:
            species_list = saved_cat_mappings["Species"]
            result_text = species_list[predicted_index]
        else:
            result_text = f"Class {predicted_index}"

        classification_result.set(result_text)

    except ValueError as e:
        messagebox.showerror("Input Error", f"Please check your inputs.\nDetails: {e}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

classify_button = ttk.Button(
    classify_frame, text="Predict Class", command=on_classify_click
)
classify_button.grid(row=5, column=0, columnspan=2, pady=10)

ttk.Label(classify_frame, text="Prediction:").grid(
    row=6, column=0, padx=10, pady=5, sticky="w"
)
ttk.Label(
    classify_frame,
    textvariable=classification_result,
    foreground="blue",
    font=("Arial", 10, "bold"),
).grid(row=6, column=1, padx=10, pady=5, sticky="w")

root.mainloop()
