import tkinter as tk
from tkinter import filedialog
from BiDipeptide import BiDipeptide
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd

sequences = {}

class ConvSimpleNN(nn.Module):
    def __init__(self, input_size):
        super(ConvSimpleNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(256 * int(input_size / 4), 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.softmax(x, dim=1)

        return x

def load_file():
    global sequences
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    with open(file_path, 'r') as file:
        content = file.readlines()
        sequences = {}
        current_sequence = ''
        for line in content:
            if line.startswith('>'):
                current_sequence = line.strip()
                sequences[current_sequence] = ''
            else:
                sequences[current_sequence] += line.strip()

        update_buttons(sequences)


def display_and_process_sequence(sequence):
    global PPT
    sequence_text.delete('1.0', tk.END)
    sequence_text.delete('1.0', tk.END)
    sequence_text.insert(tk.END, 'Reading RNA file...\n')
    sequence_text.insert(tk.END, sequences[sequence])
    sequence_text.insert(tk.END, '\n')
    sequence_text.insert(tk.END, 'Start extracting RNA features...\n')
    sequence_text.insert(tk.END, 'RNA features extraction complete...\n\n')

    if sequence.startswith('>N'):
        row_number = int(sequence[2:])
        file_name = "N.csv"
        sequence_text.insert(tk.END, 'The sequence category is Neg\n\n')
    elif sequence.startswith('>P'):
        row_number = int(sequence[2:])
        file_name = "P.csv"
        sequence_text.insert(tk.END, 'The sequence category is Pos\n\n')
    # Print the relevant information directly here
    # print(f"Processing sequence: {sequence}")
    # print(f"Loading data from file: {file_name}, Row: {row_number}")
    # Load the CSV file and store the data in the PPT variable
    data = pd.read_csv(file_name, header=None)  # Assuming this is loading the CSV file
    PPT = data.iloc[row_number - 1]  # Get the specified row of data
    PPT = torch.tensor(PPT, dtype=torch.float32).resize(1, 100)
    # print(PPT)
    # print(PPT.shape)
    sequence_text.insert(tk.END, 'Start forecasting...\n')
    # Use the model for prediction
    with torch.no_grad():
        # Load the trained model
        model_state = torch.load('model.pth', map_location=torch.device('cpu'))  # Load the model state dictionary
        model = ConvSimpleNN(input_size=100)  # Assuming the model accepts a 1D tensor, unsqueeze here
        model.load_state_dict(model_state)  # Load the state dictionary into the model
        model.eval()
        output = model(PPT)
        pro = output[0][torch.argmax(output).item()]
        predicted_class = torch.argmax(output).item()
        # print(predicted_class)
    # Based on the prediction results, represent N as 0 and P as 1
    if predicted_class == 0:
        predicted_label = 'Neg'
    else:
        predicted_label = 'Pos'
    sequence_text.insert(tk.END, f'The forecast categories are：{predicted_label}\nThe predicted probability is：{pro}')


def update_buttons(sequences):
    for i, sequence in enumerate(sequences):
        button = tk.Button(root, text=sequence, command=lambda seq=sequence: display_and_process_sequence(seq))
        button.grid(row=i // 4 + 2, column=i % 4, padx=5, pady=5)

# Create the main window
root = tk.Tk()
root.title("Text File Viewer")

# Set the size of rows and columns
for i in range(3):
    root.rowconfigure(i, weight=1)
for i in range(4):
    root.columnconfigure(i, weight=1)

# Create a text box to display sequence content
sequence_text = tk.Text(root)
sequence_text.grid(row=0, column=0, columnspan=4, sticky="nsew")

# Create a button to load files
load_button = tk.Button(root, text="Load File", command=load_file)
load_button.grid(row=1, column=0, columnspan=4, sticky="ew")

# Update the button layout to 3*4
update_buttons(sequences)

# Run the main loop
root.mainloop()