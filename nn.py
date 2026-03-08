import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import itertools
import matplotlib.pyplot as plt
import seaborn as sns


class NeuralNetwork:
    def __init__(self, layers, learning_rate, use_bias, activation_fn):
        self.layers = layers
        self.learning_rate = learning_rate
        self.use_bias = use_bias
        self.activation_fn = activation_fn

        self.weights = []
        self.biases = []

        # Xavier (or Glorot) initialization
        for i in range(len(layers) - 1):
            limit = np.sqrt(6 / (layers[i] + layers[i + 1]))
            w = np.random.uniform(-limit, limit, (layers[i], layers[i + 1]))
            self.weights.append(w)

            b = np.zeros((1, layers[i + 1]))

            self.biases.append(b)

    # forward formula:
    def _activation(self, x):
        if self.activation_fn == "sigmoid":
            return np.where(
                x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x))
            )  # an equivalent formula that handles large negative numbers safely
        # as sigmoid fails with very negative numbers, causing overflow
        elif self.activation_fn == "tanh":
            return np.tanh(
                x
            )  # i guess this will be removed, built in, but is it a must?

    # backward derivative:
    def _activation_derivative(
        self, x
    ):  # x here is the acivated output (e.g. x=tanh(x))
        if self.activation_fn == "sigmoid":
            return x * (1 - x)
        elif self.activation_fn == "tanh":
            return (
                1.0 - x**2
            )  # avoids calculating expensive exponentials during backprop

    # multi-class classification (3 classes) -> softmax, unlike sigmoid (binary)
    # forces the sum of all outputs to equal 1.0 (probability)
    # It pushes the values apart, suppressing the weak classes and highlighting the strong one
    def _softmax(self, z):
        exp_scores = np.exp(
            z - np.max(z, axis=1, keepdims=True)
        )  # shift invariant property
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

    # stochastic gradient descent (anxious)
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

    # batch (patient)
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

    # stochastic adaptive moment estimation(smart)
    def train_adam(
        self, X, y, epochs, beta1=0.9, beta2=0.999, epsilon=1e-8, print_every=100
    ):

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
                vW[j] = beta2 * vW[j] + (1 - beta2) * (g_W**2)

                m_hat_W = mW[j] / (1 - beta1**t)
                v_hat_W = vW[j] / (1 - beta2**t)

                self.weights[j] -= (
                    self.learning_rate * m_hat_W / (np.sqrt(v_hat_W) + epsilon)
                )

                if self.use_bias:
                    mb[j] = beta1 * mb[j] + (1 - beta1) * g_B
                    vb[j] = beta2 * vb[j] + (1 - beta2) * (g_B**2)

                    m_hat_B = mb[j] / (1 - beta1**t)
                    v_hat_B = vb[j] / (1 - beta2**t)

                    self.biases[j] -= (
                        self.learning_rate * m_hat_B / (np.sqrt(v_hat_B) + epsilon)
                    )

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


# preprocessing
def load_and_preprocess_data():
    df = pd.read_csv(r"penguins.csv")
    # null filling
    numerical_cols = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]
    categorical_cols = ["Species", "OriginLocation"]
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

    # One-Hot encoding
    encoded_df = df.copy()
    cat_mappings = {}
    for col in categorical_cols:
        unique_values = df[col].dropna().unique()
        # this specific order saved to send to the GUI later
        cat_mappings[col] = unique_values

        for val in unique_values:
            new_col_name = f"{col}_{val}"
            encoded_df[new_col_name] = (df[col] == val).astype(int)

        encoded_df.drop(columns=[col], inplace=True)
    target_cols = encoded_df.filter(like="Species_").columns
    X = encoded_df.drop(columns=target_cols).values
    y = encoded_df[target_cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=np.argmax(y, axis=1)
    )
    return X_train, X_test, y_train, y_test, cat_mappings


# normalization (standardization/Z-score normalization)
# for neural network to converge
def scale_data(X_train, X_test):
    # identify and scale numerical columns only (indices where values are not just 0 and 1)
    numerical_indices = []
    for i in range(X_train.shape[1]):
        col_values = np.unique(X_train[:, i])
        if not np.all(np.isin(col_values, [0, 1])):
            numerical_indices.append(i)

    mean = np.mean(X_train[:, numerical_indices], axis=0)
    std = (
        np.std(X_train[:, numerical_indices], axis=0) + 1e-8
    )  # a small epsilon to avoid div by zero

    X_train[:, numerical_indices] = (X_train[:, numerical_indices] - mean) / std
    # transforming X_test using x_mean & x_std from X_train to prevent data leakage
    X_test[:, numerical_indices] = (X_test[:, numerical_indices] - mean) / std
    return X_train, X_test, numerical_indices, mean, std


# from scratch confusion matrix
def confusion_matrix_manual(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_true)):
        actual = np.where(classes == y_true[i])[0][0]
        pred = np.where(classes == y_pred[i])[0][0]
        cm[actual, pred] += 1

    return cm, classes


def plot_manual_cm_graphic(cm, classes, title, class_names_map=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    if class_names_map is not None:
        tick_labels = [
            class_names_map[int(c)] if int(c) < len(class_names_map) else str(c)
            for c in classes
        ]
    else:
        tick_labels = classes

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.show()


# gridsearch
def run_grid_search_and_plot():
    print("Preparing Data for Grid Search...")
    X_train, X_test, y_train, y_test, mappings = load_and_preprocess_data()
    species_names = mappings["Species"].tolist()
    X_train, X_test, _, _, _ = scale_data(X_train, X_test)
    species_names = mappings.get("Species", ["0", "1", "2"])
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    activations = ["sigmoid", "tanh"]
    learning_rates = [0.01, 0.001]
    epochs_list = [500, 1000]
    hidden_layers_configs = [[5], [8], [5, 4], [8, 8]]
    best_results = []

    print(f"{'='*60}")
    print(f"STARTING GRID SEARCH")
    print(f"{'='*60}")

    for act in activations:
        print(f"\n--- Searching best parameters for: {act.upper()} ---")

        best_acc_for_act = -1
        best_params_for_act = {}
        best_model_for_act = None

        for lr in learning_rates:
            for epochs in epochs_list:
                for hidden_conf in hidden_layers_configs:

                    full_layers = [input_size] + hidden_conf + [output_size]

                    nn = NeuralNetwork(
                        layers=full_layers,
                        learning_rate=lr,
                        use_bias=True,
                        activation_fn=act,
                    )

                    nn.train_adam(
                        X_train, y_train, epochs=epochs, print_every=epochs + 1
                    )
                    test_acc = nn.evaluate(X_test, y_test)
                    train_acc = nn.evaluate(X_train, y_train)
                    if test_acc > best_acc_for_act:
                        best_acc_for_act = test_acc
                        best_model_for_act = nn
                        best_params_for_act = {
                            "Activation Function": act,
                            "Train Accuracy": f"{train_acc*100:.2f}%",
                            "Test Accuracy": f"{test_acc*100:.2f}%",
                            "LR": lr,
                            "Epochs": epochs,
                            "#Layers": len(hidden_conf),
                            "#HiddenNodes": str(hidden_conf),
                        }

        print(
            f"Best {act} model found. Accuracy: {best_params_for_act['Test Accuracy']}"
        )
        best_results.append(best_params_for_act)

        # from scratch confusion matrix
        y_pred = best_model_for_act.predict(X_test)
        if y_test.ndim > 1 and y_test.shape[1] > 1:
            y_true = np.argmax(y_test, axis=1)
        else:
            y_true = y_test

        cm, classes_found = confusion_matrix_manual(y_true, y_pred)
        plot_manual_cm_graphic(
            cm,
            classes_found,
            f"Best Confusion Matrix ({act})",
            class_names_map=species_names,
        )

    print(f"\n{'='*60}")
    print("GRID SEARCH COMPLETE. BEST RESULTS:")
    print(f"{'='*60}")
    for res in best_results:
        print(res)

    plot_results_table(best_results)


def plot_results_table(data):
    df = pd.DataFrame(data)
    cols = [
        "Activation Function",
        "Train Accuracy",
        "Test Accuracy",
        "LR",
        "Epochs",
        "#Layers",
        "#HiddenNodes",
    ]
    df = df[cols]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colColours=["#f2f2f2"] * len(df.columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    plt.title(
        "Best Model Parameters per Activation Function", fontsize=14, weight="bold"
    )
    plt.show()


# def visualize_separability():
#     print("Loading data for visualization...")
#     try:
#         df = pd.read_csv('penguins.csv')
#     except FileNotFoundError:
#         print("Error: penguins.csv not found.")
#         return

#     plot_cols = ['Species', 'CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']
#     df = df[plot_cols].dropna()
#     print("Generating Pair Plot...")

#     sns.set_style("whitegrid")
#     g = sns.pairplot(df, hue='Species', markers=["o", "s", "D"], palette="bright")
#     g.fig.suptitle("Feature Pairwise Relationships (Look for Gaps)", y=1.02)
#     plt.show()

#     print("Generating Focused Plot...")
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(
#         data=df,
#         x='CulmenLength',
#         y='FlipperLength',
#         hue='Species',
#         style='Species',
#         s=100,
#         palette="deep"
#     )
#     plt.title("Can you draw straight lines between the colors?", fontsize=14)
#     plt.xlabel("Culmen Length (mm)")
#     plt.ylabel("Flipper Length (mm)")
#     plt.grid(True, alpha=0.3)
#     plt.show()

if __name__ == "__main__":
    # visualize_separability()
    run_grid_search_and_plot()
