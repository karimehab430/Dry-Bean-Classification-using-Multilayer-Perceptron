import tkinter as ttk
from tkinter import messagebox
import main


def submit_values():
    features = []
    classes = []

    if MajorAxisLength_var.get():
        features.append("MajorAxisLength")
    if MinorAxisLength_var.get():
        features.append("MinorAxisLength")
    if Roundnes_var.get():
        features.append("roundnes")
    if Perimeter_var.get():
        features.append("Perimeter")
    if Area_var.get():
        features.append("Area")

    if bombay_var.get():
        classes.append("BOMBAY")
    if cali_var.get():
        classes.append("CALI")
    if sira_var.get():
        classes.append("SIRA")

    global learning_rate, num_of_epochs, mse_threshold, bias, num_hidden_layers, num_neurons, activation_function
    learning_rate = float(learning_rate_entry.get())
    num_of_epochs = int(num_of_epochs_entry.get())
    bias = bias_var.get()
    layers = [len(features)]

    for entry in num_neurons_entries:
        num_neurons = int(entry.get())
        layers.append(num_neurons)
    activation_function = activation_function_var.get()

    layers.append(len(classes))
    print("Layers:", layers, '\n')
    if len(features) < 2:
        messagebox.showinfo("showerror", "Select at least two features")
        features.clear()
        classes.clear()

    if len(classes) < 2:
        messagebox.showinfo("showerror", "Select at least two classes")
        features.clear()
        classes.clear()

    main.run(features, classes, learning_rate, num_of_epochs, bias, layers, activation_function)


def select_all_features():
    MajorAxisLength_var.set(1)
    MinorAxisLength_var.set(1)
    Roundnes_var.set(1)
    Perimeter_var.set(1)
    Area_var.set(1)


def select_all_classes():
    bombay_var.set(1)
    cali_var.set(1)
    sira_var.set(1)


def add_hidden_layer_entry(window):
    global num_neurons_entries

    label_text = f"Layer {len(num_neurons_entries) + 1}:"

    label = ttk.Label(window, text=label_text)
    label.grid(row=3, column=len(num_neurons_entries), padx=(50, 0))

    entry = ttk.Entry(window, width=5)
    entry.grid(row=3, column=len(num_neurons_entries) + 1, padx=(0, 50))

    num_neurons_entries.append(entry)


def main_screen(window):
    window.geometry("1000x500")

    global MajorAxisLength_var, MinorAxisLength, Roundnes_var, Perimeter_var, MinorAxisLength_var, Area_var
    MajorAxisLength_var = ttk.BooleanVar()
    MinorAxisLength_var = ttk.BooleanVar()
    Roundnes_var = ttk.BooleanVar()
    Perimeter_var = ttk.BooleanVar()
    Area_var = ttk.BooleanVar()

    global bombay_var, cali_var, sira_var
    bombay_var = ttk.BooleanVar()
    cali_var = ttk.BooleanVar()
    sira_var = ttk.BooleanVar()

    choose_feature = ttk.Label(window, text="Select Features")
    choose_feature.grid(row=0, column=0, pady=10, sticky="w")

    Area_checkbox = ttk.Checkbutton(window, text="Area", variable=Area_var)
    Area_checkbox.grid(row=0, column=1, padx=(0, 10))

    Perimeter_checkbox = ttk.Checkbutton(window, text="Perimeter", variable=Perimeter_var)
    Perimeter_checkbox.grid(row=0, column=2, padx=(0, 10))

    MajorAxisLength_checkbox = ttk.Checkbutton(window, text="MajorAxisLength", variable=MajorAxisLength_var)
    MajorAxisLength_checkbox.grid(row=0, column=3, padx=(0, 10))

    MinorAxisLength_checkbox = ttk.Checkbutton(window, text="MinorAxisLength", variable=MinorAxisLength_var)
    MinorAxisLength_checkbox.grid(row=0, column=4, padx=(0, 10))

    Roundness_checkbox = ttk.Checkbutton(window, text="roundnes", variable=Roundnes_var)
    Roundness_checkbox.grid(row=0, column=5, padx=(0, 10))

    select_all_features_button = ttk.Button(window, text="Select All Features", command=select_all_features)
    select_all_features_button.grid(row=0, column=6, pady=10, sticky="w")

    choose_class = ttk.Label(window, text="Select Classes")
    choose_class.grid(row=1, column=0, pady=10, sticky="w")

    BOMBAY_checkbox = ttk.Checkbutton(window, text="BOMBAY", variable=bombay_var)
    BOMBAY_checkbox.grid(row=1, column=1, padx=(0, 10))

    CALI_checkbox = ttk.Checkbutton(window, text="CALI", variable=cali_var)
    CALI_checkbox.grid(row=1, column=2, padx=(0, 10))

    SIRA_checkbox = ttk.Checkbutton(window, text="SIRA", variable=sira_var)
    SIRA_checkbox.grid(row=1, column=3, padx=(0, 10))

    select_all_classes_button = ttk.Button(window, text="Select All Classes", command=select_all_classes)
    select_all_classes_button.grid(row=1, column=4, pady=10, sticky="w")

    global num_neurons_entries

    ttk.Label(window, text="Number of Neurons in Layer 1:").grid(row=3, column=0, pady=10, sticky="w")
    num_neurons_entries = [ttk.Entry(window, width=5)]
    num_neurons_entries[0].grid(row=3, column=1, padx=5, pady=10, sticky="w")

    ttk.Button(window, text="Add Hidden Layer", command=lambda: add_hidden_layer_entry(window)).grid(row=4, column=0,
                                                                                                     pady=10,
                                                                                                     sticky="w",padx=5)

    global activation_function_var
    activation_function_var = ttk.StringVar(value="Sigmoid")

    activation_label = ttk.Label(window, text="Choose Activation Function")
    activation_label.grid(row=5, column=0, pady=10, sticky="w")

    sigmoid_radio = ttk.Radiobutton(window, text="Sigmoid", variable=activation_function_var, value="Sigmoid")
    sigmoid_radio.grid(row=5, column=1, padx=(0, 10), sticky="w")

    tanh_radio = ttk.Radiobutton(window, text="Hyperbolic Tangent", variable=activation_function_var, value="Tanh")
    tanh_radio.grid(row=5, column=2, padx=(0, 10), sticky="w")

    epochs_label = ttk.Label(window, text="Number of Epochs (m)")
    epochs_label.grid(row=6, column=0, pady=10, sticky="w")

    global num_of_epochs_entry, learning_rate_entry, mse_entry, bias_var
    num_of_epochs_entry = ttk.Entry(window)
    num_of_epochs_entry.grid(row=6, column=1, pady=10, sticky="w")

    learning_rate_label = ttk.Label(window, text="Learning Rate (eta)")
    learning_rate_label.grid(row=6, column=2, pady=10, sticky="w")
    learning_rate_entry = ttk.Entry(window)
    learning_rate_entry.grid(row=6, column=3, pady=10, sticky="w")

    bias_var = ttk.BooleanVar()
    bias_checkbox = ttk.Checkbutton(window, text="Add Bias", variable=bias_var)
    bias_checkbox.grid(row=6, column=4, pady=10, sticky="w")

    submit_button = ttk.Button(window, text="Submit", command=submit_values)
    submit_button.grid(row=7, column=0, pady=10, sticky="w")

    num_of_epochs_entry.insert(0, "1000")
    learning_rate_entry.insert(0, "0.01")

    window.mainloop()


if __name__ == "__main__":
    main_screen(ttk.Tk())
