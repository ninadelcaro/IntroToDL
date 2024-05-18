#################################################################################################### Imports
import pickle
import random
import math
import os
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torchsummary import summary

from tqdm import tqdm
import torch.optim as optim
import torchvision.transforms.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#################################################################################################### Load Pickle Files
def trim_array(arr):
    if np.all(arr == 0):  # Check if the array is all zeros
        return arr
    first_non_zero = np.argmax(arr != 0)
    last_non_zero = len(arr) - np.argmax(arr[::-1] != 0)
    return arr[first_non_zero:last_non_zero]

allaudios = [] # Creates an empty list
for root, dirs, files in os.walk("train"):
    i=0 
    for file in files:
        if file.endswith(".pkl"):
           audio = file
           openaudios = open(os.getcwd() + "/train/" + audio, 'rb')
           loadedaudios = pickle.load(openaudios)
           
           allaudios.append(loadedaudios)
           i+=1

audio_data = []
valence = []
audio_lengths = []
for audio in allaudios:
    # Get Rid of front and end trailing zeros
    audio_trimmed = trim_array(audio['audio_data'])
    audio_data.append(audio_trimmed)

    audio_length = len(audio_trimmed)
    audio_lengths.append(audio_length)
    valence.append(audio['valence'])


mean_length = np.mean(audio_lengths)
median_length = np.median(audio_lengths)
std_dev = np.std(audio_lengths)
min_length = np.min(audio_lengths)
max_length = np.max(audio_lengths)

#################################################################################################### Pad / Truncation (get all audio the same length)
def pad_trunc_audio(audio_data, target_length = int(np.percentile(audio_lengths, 95))):
    standardized_data = []
    for data in audio_data:
        if len(data) < target_length:
            padded_data = np.pad(data, (0, target_length - len(data)), 'constant', constant_values=(0, 0))
            standardized_data.append(padded_data)
        elif len(data) > target_length:
            truncated_data = data[:target_length]
            standardized_data.append(truncated_data)
        else:
            standardized_data.append(data)
    return standardized_data

standardized_audios = pad_trunc_audio(audio_data)
print("Done padding")

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

    def forward(self, x):
        with torch.no_grad():
            x = x - self.mean
            x = x / self.std
        return x

#################################################################################################### DataLoader Creation
class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

size_train = int(round(len(standardized_audios) * 0.8))
X_train = standardized_audios[:size_train]
X_test = standardized_audios[size_train:]
y_train = valence[:size_train]
y_test = valence[size_train:]

flatten = np.concatenate(X_train)
mean = np.mean(flatten)
std = flatten.std()
normalization = Normalization(mean, std)

batch_size = 64

train_dataset = AudioDataset(X_train, y_train)
test_dataset = AudioDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,) # you can speed up the host to device transfer by enabling pin_memory.
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,) # you can speed up the host to device transfer by enabling pin_memory.

print("Done with data loaders")
#################################################################################################### Model Architecture
class CNN1d(nn.Module):
    def __init__(self, pre_proocesses, hidden_sizes, activation_function):

        super(CNN1d, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.activation_function = activation_function

        self.width = 128

        self.layers = nn.ModuleList()

        # # add preprocessing steps
        # for process in pre_proocesses:
        #     self.layers.append(process)
        self.layers.append(normalization)

        for i in range(len(self.hidden_sizes)):
            self.layers.append(nn.Conv1d(1 if i ==0 else self.hidden_sizes[i-1], self.hidden_sizes[i], kernel_size=3))
            self.layers.append(nn.BatchNorm1d(self.hidden_sizes[i], eps=.00001, momentum=0.1, affine=True, track_running_stats=True))
            self.layers.append(nn.MaxPool1d(kernel_size=3))
            self.layers.append(self.activation_function())


        self.layers.append(nn.AdaptiveAvgPool1d(1))
        self.layers.append(nn.Flatten()) 
        self.layers.append(nn.Linear(in_features=self.hidden_sizes[-1], out_features=self.width))
        self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nn.Linear(in_features=self.width, out_features=1))

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
#################################################################################################### Train Model Loop
def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    
    return {
        "Mean Absolute Error (MAE)": mae,
        "Mean Squared Error (MSE)": mse,
        "Root Mean Squared Error (RMSE)": rmse,
        "R-squared (R^2)": r2
    }

def loss_plot(train_loss, validation_loss, filename):
    epochs = range(1, len(train_loss) + 1) # start at 1 instead of 0
    # Plotting the training and validation losses
    plt.figure(figsize=(5, 5))
    plt.plot(epochs, train_loss, label='Training Loss', color='blue')
    plt.plot(epochs, validation_loss, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)
    plt.show()
    

def train_model(net, optimizer, train_loader, val_loader, epochs):
    # Define the loss function
    criterion = nn.MSELoss()
    # Define the optimizer

    train_loss_lst = []
    val_loss_lst = []

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        # Iterate over training batches
        for audio, valence in train_loader:
            optimizer.zero_grad()  # Reset gradients
            valence = valence.float() # was double 

            audio = audio.unsqueeze(1) # [batch, channel=1, 128,145]
            audio, valence = audio.to(device), valence.to(device)

            outputs = net(audio)
            outputs = outputs.squeeze()  # Reshape the output to match target
            loss = criterion(outputs, valence)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_loss_lst.append(train_loss)

        # Validation
        net.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for audio, valence in val_loader:
                valence = valence.float() # was double

                audio = audio.unsqueeze(1) # [batch, channel=1, 128,145]
                audio, valence = audio.to(device), valence.to(device)
                outputs = net(audio)
                outputs = outputs.squeeze()  # Reshape the output to match target
                val_loss = criterion(outputs, valence)
                val_running_loss += val_loss.item()

        val_loss = val_running_loss / len(val_loader)
        val_loss_lst.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
    #display loss graph
    # loss_plot(train_loss_lst, val_loss_lst, "")
    print("Training for CNN is finished.")

    return train_loss_lst, val_loss_lst
#################################################################################################### Find Best Optimizer
names = [] # Initialize an empty list names to store optimizer names for visualization of results.
learning_rate = 0.001
num_epochs = 50


for opt in [optim.SGD, optim.Adagrad, optim.Adam]:
    names.append(opt.__name__)
    model_1d = CNN1d([], [16, 32, 64, 128, 256], nn.ReLU).to(device)

    if opt is optim.SGD:
        optimizer = opt(model_1d.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    else:
        optimizer = opt(model_1d.parameters(), lr=learning_rate)

    
    train_loss_lst, val_loss_lst = train_model(model_1d, optimizer, train_dataloader, test_dataloader, num_epochs)
    plt.plot(val_loss_lst)
    # Print the validation accuracy and loss in the last epoch
    print(f'{opt.__name__}\n\tValidation Loss: {val_loss_lst[-1]:.5}\n')
    

#Display a legend in the plot for optimizer names.
plt.legend(names)
#Label the x-axis as "Epoch" and the y-axis as "Validation Accuracy."
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
#Show the plot containing the accuracy evolution over epochs for each optimizer.
plt.savefig("optimizers_comparison.png")
plt.show()

#################################################################################################### Setup for Random Search

def generate_sequences(start=16, max_value=512):
    sequences = [[start]] 
    finished_sequences = []

    while sequences:
        new_sequences = []
        for seq in sequences:
            last_value = seq[-1]
            
            if seq.count(last_value) < 2:
                new_seq = seq + [last_value]
                new_sequences.append(new_seq)
            
            doubled_value = last_value * 2
            if doubled_value <= max_value:
                new_seq = seq + [doubled_value]
                new_sequences.append(new_seq)
        
        finished_sequences.extend(sequences)
        sequences = new_sequences

    return finished_sequences

def plot_search(results, x_str, y_str, res_str, scale=False):

    # Assuming coarse_results is a list of dictionaries with 'lr', 'hidden_size', 'val_loss', and 'accuracy'
    # Extract relevant information for the heatmap
    lr_values = [result[x_str] for result in results]
    hidden_size_values = [result[y_str] for result in results]
    val_loss_values = [result[res_str] for result in results]

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    heatmap = plt.scatter(lr_values, hidden_size_values, c=val_loss_values, cmap='RdYlGn', marker='o', s=100)
    plt.colorbar(heatmap, label=res_str)
    if scale:
        plt.xscale('log')  # Use a logarithmic scale for learning rates if appropriate

    # Set labels and title
    plt.xlabel(x_str)
    plt.ylabel(y_str)
    plt.title('Hyperparameter Search')
    plt.grid(True)

    # Show the plot
    plt.savefig("comparison_coarse_fine_hyperparameter_tuning.png")

    plt.show()

def hyper_train_setup(hidden_sizes, learning_rate, num_epochs):
  # Create the model
    model_1d = CNN1d([], hidden_sizes, nn.ReLU).to(device)
    optimizer = optim.Adam(model_1d.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(params=model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

    ## Train the model
    train_loss_lst, eval_loss_lst = train_model(model_1d, optimizer, train_dataloader, test_dataloader, num_epochs)
    loss_plot(train_loss_lst, eval_loss_lst, f"hyperparameter_tuning_hidden_size_{hidden_sizes}_lr_{learning_rate:.4}_epochs_{num_epochs}.png")
    return eval_loss_lst[-1] # last epoch loss

#################################################################################################### Coarse Random Search
coarse_trials = 30
num_epochs = 15
coarse_results = []

hidden_sizes_list = generate_sequences(16,128)
hidden_options = len(hidden_sizes_list)


for i in range(coarse_trials):
    lr = 10**random.uniform(math.log10(0.001), math.log10(0.1))
    index = int(2**random.uniform(math.log2(1), math.log2(hidden_options-1)))
    hidden_sizes = hidden_sizes_list[index]
    val_loss = hyper_train_setup(hidden_sizes, lr, num_epochs)

    coarse_results.append({'lr': lr, 'index':index, 'hidden_size': hidden_sizes, 'loss': val_loss})

    print(f"{i+1}. Learning rate: {lr:.4} and hidden sizes: {hidden_sizes}")
    print(f"\tValidation loss: {val_loss:.5}\n")

# Find the best parameters from coarse search
best_coarse_params = min(coarse_results, key=lambda x: x['loss'])
print(f"Best parameters found:\n - Learning rate: {best_coarse_params['lr']:.5}\n - Hidden sizes: {best_coarse_params['hidden_size']}\n - Validation loss: {best_coarse_params['loss']:.5}%")


fine_trials = 10
fine_results = []
hidden_options = len(hidden_sizes_list)


for _ in range(fine_trials):
    lr = 2**random.uniform(np.log2(0.5 * best_coarse_params['lr']), np.log2(1.5 * best_coarse_params['lr']))

    index = float('inf')
    while index > hidden_options-1:
        index = random.randint(int(0.8 * best_coarse_params['index']), int(1.2 * best_coarse_params['index']) + 1) # not inclusive on the end
    hidden_sizes = hidden_sizes_list[index]
    val_loss = hyper_train_setup(hidden_sizes, lr, num_epochs)

    fine_results.append({'lr': lr, 'index':index, 'hidden_size': hidden_sizes, 'loss': val_loss})

    print(f"Learning rate: {lr:.4} and hidden sizes: {hidden_sizes}")
    print(f"\tValidation loss: {val_loss:.5}\n")

# Find the best parameters from fine search
best_fine_params = min(fine_results, key=lambda x: x['loss'])

print(f"Best parameters found with coarse search:\n - Learning rate: {best_coarse_params['lr']:.5}\n - Hidden sizes: {best_coarse_params['hidden_size']}\n - Validation loss: {best_coarse_params['loss']:.5}%")
print(f"Best parameters found with fine search:\n - Learning rate: {best_fine_params['lr']:.5}\n - Hidden sizes: {best_fine_params['hidden_size']}\n - Validation loss: {best_fine_params['loss']:.5}%")
plot_search(coarse_results + fine_results, "lr", "index", 'loss')
#################################################################################################### Run and Save Best model
model_best = CNN1d([], best_coarse_params['hidden_size'], nn.ReLU).to(device)
optimizer = optim.Adagrad(model_best.parameters(), lr=best_coarse_params['lr'])
train_loss_lst, val_loss_lst = train_model(model_best, optimizer, train_dataloader, test_dataloader, 13)
loss_plot(train_loss_lst, val_loss_lst, f"best_model_hidden_size_{hidden_sizes}_lr_{learning_rate:.4}_epochs_{num_epochs}.png")

save_path = "best_coarse_model_adagrad_20_epochs_with_normalization"

torch.save(model_best, save_path)

# Function to process a single pickle file and return the filename and prediction
def process_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    #apply preprocessing
    audio_data = data["audio_data"]
    audio_trimmed = trim_array(audio_data)
    
    audio_pad_trunced = pad_trunc_audio([audio_trimmed])
    audio_data_tensor = torch.tensor(audio_pad_trunced).unsqueeze(0)

    # print(f"Original shape: {audio_data.shape}")
    # print(f"Trimmed shape: {audio_trimmed.shape}")
    # print(f"Pad/Truncated shape: {len(audio_pad_trunced[0])}")
    # print(f"Tensor shape: {audio_data_tensor.shape}")


    valence = model_best(audio_data_tensor).item()
    return os.path.basename(file_path), valence

# List to store results
results = []

# Iterate through all files in the folder
for filename in os.listdir("test"):
    if filename.endswith('.pkl'):
        file_path = os.path.join("test", filename)
        file_id, valence = process_file(file_path)
        results.append((file_id, valence))

# Create a DataFrame for better visualization and potential saving to CSV
results_df = pd.DataFrame(results, columns=['ID', 'valence'])

