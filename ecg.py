#Import packages
import numpy as np
import torchvision
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.nn import Linear, ReLU,LeakyReLU, CrossEntropyLoss, Sequential, Conv1d, MaxPool1d, Module, Softmax, Flatten, BatchNorm1d
from torch.optim import Adam, SGD
from torch import no_grad
from torch.utils.data import Dataset, DataLoader

import pywt

import scipy

from tqdm.auto import tqdm

!pip --quiet install torchsummaryX

import torchsummaryX

!pip --quiet install wfdb

!pip --quiet install matplotlib==3.1.3

import matplotlib.pyplot as plt

import os
import wfdb

#Installing the ECG recordings from the MIT BIH Arrhythmia Database onto Colab 
wfdb.dl_database('mitdb', os.path.join(os.getcwd(), 'mitdb'))

# Function to classify beat class

def classify_beat_class(symbol):
    if symbol in ["N", "R", "L", "e", "j"]: #Non-ectopic class according to AAMI standards
        return 0  
    elif symbol in ['S',"a", "A", "J"]:          #Supra-ventricular ectopic class
        return 1
    elif symbol in ['V', 'E', "!"]:              #Ventricular ectopic class
        return 2
    elif symbol =='F':                           #Fusion class
        return 3
    elif symbol in ['Q', '/', 'f']:              #Unknown class
        return 4

#Function classifies the beat type

def classify_beat(symbol):
    beat_types =["N", "R", "L", "e", "j",'S',"a", "A", "J", 'V', 'E', "!", 'F', 'Q', '/', 'f'] # ["N", "A", "V", "F", "/"] 
    for index, value in enumerate(beat_types):
      if symbol == value: 
        return index

# Function to return a sequence surrounding a beat 
# with window_size for each side

def get_sequence(signal, beat_loc, window_sec, fs):
    window_one_side = int(window_sec * fs)
    beat_start = beat_loc - window_one_side
    beat_end = beat_loc + window_one_side #+ 1
    if beat_end < signal.shape[0]:
        sequence = signal[beat_start:beat_end, 0]
        return sequence.reshape(1, -1, 1)
    else:
        return np.array([])

from sklearn.preprocessing import StandardScaler

all_sequences = []
all_labels = []
window_sec = 130/360

records = list(range(100, 110))+ list(range(111, 120)) +list(range(121, 125))+ [i for i in range(200,235) if (i not in [204, 206, 211, 216,218, 224,225,226,227,229])]
for subject in records:
    record = wfdb.rdrecord(f'mitdb/{subject}')
    annotation = wfdb.rdann(f'mitdb/{subject}', 'atr')
    atr_symbol = annotation.symbol
    atr_sample = annotation.sample

    fs = record.fs
    # Normalizing by mean and standard deviation
    scaler = StandardScaler()
    signal = scaler.fit_transform(record.p_signal)
    subject_labels = []
    for i, i_sample in enumerate(atr_sample):
        label = classify_beat(atr_symbol[i])
        sequence = get_sequence(signal, i_sample, window_sec, fs)
        if label is not None and sequence.size > 0:
            all_sequences.append(sequence)
            subject_labels.append(label)
    if len(subject_labels) == 0:
      continue
    all_labels.extend(subject_labels)

import scipy.signal

def detect_peaks(ecg_signal, sampling_rate):
  # Use Pan-Tompkins algorithm to detect R-peaks
  W1     = 5*2/sampling_rate                                    
  W2     = 15*2/sampling_rate                        
  b, a   = scipy.signal.butter(4, [W1,W2], 'bandpass')                  
  ecg_signal = scipy.signal.filtfilt(b,a, ecg_signal) 

  diff = - ecg_signal[:-2] - 2 * ecg_signal[1:-1] + 2 * ecg_signal[1:-1] + ecg_signal[2:]
  diff_squared = np.square(diff)

  window_size = int(0.12*sampling_rate)

  moving_avg = scipy.ndimage.uniform_filter1d(diff_squared,
                                              window_size,
                                              origin=(window_size-1)//2)

  diff_winsize = int(0.2*sampling_rate)
  moving_avg_diff = scipy.signal.convolve(moving_avg, np.ones(diff_winsize, ), mode='same')

  peaks = []
  for i in range(1, len(moving_avg_diff) - 1):
    h0, h, h1 = moving_avg_diff[i-1], moving_avg_diff[i], moving_avg_diff[i+1]

    if h > h0 and h > h1:
      peaks.append(i)

  prob_rpeaks = []
  for approx_rpeak in peaks:
    win = ecg_signal[max(0, approx_rpeak-(window_size)):
                     min(len(ecg_signal), approx_rpeak+(window_size))]
    peak_val = max(win)

    prob_rpeak = approx_rpeak-(window_size) + np.argmax(win)
    prob_rpeaks.append(prob_rpeak)

  rpeaks = []

  noise_level = 0
  signal_level = 0
  for i, peak in enumerate(prob_rpeaks):
    thresh = noise_level + 0.25 * (signal_level - noise_level)

    peak_height = moving_avg[peak]

    if peak_height > thresh:
      signal_level = 0.125 * peak_height + 0.875 * signal_level
      rpeaks.append(peak)
    else:
      noise_level = 0.125 * peak_height + 0.875 * noise_level

  filter_peaks = [rpeaks[0]]
  refractory_period = 0.2*sampling_rate
  twave_window_thresh = 0.3*sampling_rate

  for i in range(1, len(rpeaks)):
    if (rpeaks[i] - rpeaks[i - 1] < twave_window_thresh and
        rpeaks[i] - rpeaks[i - 1] > refractory_period):
      if moving_avg[rpeaks[i]] > (moving_avg[rpeaks[i - 1]] / 2):
        filter_peaks.append(rpeaks[i])
    else:
      filter_peaks.append(rpeaks[i])

  rpeaks = filter_peaks
  
  '''
  # Uncomment this section if you want to plot the peaks
  fig = plt.figure(figsize=(24, 3))
  plt.plot(ecg_signal)
  for rpeak in rpeaks:
    plt.plot(rpeak, ecg_signal[rpeak], 'ro')
  for peak in rpeaks:
    plt.axvspan(peak-window_size, peak+window_size, alpha=0.3)
  plt.show()'''

  # plt.plot(moving_avg)
  # for peak in peaks:
  #   plt.plot(peak, moving_avg[peak], 'go')
  # plt.show()

  return rpeaks

for i in range(5):
  detect_peaks(np.concatenate([
      all_sequences[j][0] for j in range(10*i, 10*(i+1))
  ]).squeeze(), 360)

  # TESTING R PEAK DETECTION

start, end = 3000, 12000
subject = 119
record = wfdb.rdrecord(f'mitdb/{subject}')
annotation = wfdb.rdann(f'mitdb/{subject}', 'atr')
atr_symbol = annotation.symbol
atr_sample = annotation.sample
rpeaks = detect_peaks(record.p_signal[start:end,0], 360)
peaks = np.array(rpeaks)

fig = plt.figure(figsize=(30, 5))
for peak in peaks:
  plt.plot(peak, record.p_signal[start+peak, 0], 'ro', ms=15)
for indx, sample in enumerate(atr_sample):
  if sample > start and sample < end:
    plt.plot(sample - start, record.p_signal[sample, 0], 'go', ms=8)
    plt.text(sample - start, record.p_signal[sample, 0], atr_symbol[i], size=20)
plt.plot(record.p_signal[start:end,0])
plt.show()

peaks, atr_sample

skip = 1

for i in range(109923):
  if all_labels[i] == 13:
    print('found')
    if skip:
      skip -=1
      continue
    print(i)
    plt.plot(all_sequences[i][0])
    break
squeezed_sequences = []
label_ends = []
for label in range(16):
  for signal in range(len(all_sequences)):
    if all_labels[signal] == label:
      squeezed_sequences.append(all_sequences[signal].squeeze())
  # save index where current label ends
  label_ends.append(len(squeezed_sequences)*260 - 1) 
concat_sequences = np.concatenate(squeezed_sequences)
peaks = detect_peaks(concat_sequences, 360)
len(peaks)

detected_sequences = []
detected_labels = []

index = 0
for peak in peaks:
  detected_sequences.append(concat_sequences[ peak - 130: peak + 130].reshape(1, 260, 1))
  if peak > label_ends[index]:
    index += 1
  detected_labels.append(index)

all_sequences = detected_sequences
all_labels = detected_labels

plt.plot(all_sequences[0][0])

undenoised_sequences = torch.as_tensor(np.array(all_sequences)).squeeze(3)
all_labels =  torch.as_tensor(np.array(all_labels))

def get_parent_class(index):
    if index < 5: #class N
      return 0
    elif index < 9:
      return 1 #class S
    elif index < 12:
      return 2 #class V
    elif index ==12:
      return 3 #class F
    else:
      return 4 #class Q
def get_counts(labels):
  class_counts = [0]*5
  # Count the number of labels in the training set that belong to each class
  for i, v in enumerate(labels):
      if v < 5:
        class_counts[0] += 1 #class N
      elif v < 9:
        class_counts[1] += 1 #class S
      elif v < 12:
        class_counts[2] += 1 #class V
      elif v ==12:
        class_counts[3] += 1 #class F
      else:
        class_counts[4] += 1 #class Q
  return class_counts

def synthetic_data(all_sequences, all_labels):
  
  class_counts = get_counts(all_labels)

  max_count_index = np.argmax(class_counts) # Is class N, but we'll do this anyway

  labels, label_counts = np.unique(all_labels, return_counts=True)

  indices_by_label = {labels[i]: [] for i in range(len(label_counts))}
  for indx, label in enumerate(all_labels):
    indices_by_label[label.item()].append(indx)

  final_indices = []
  final_labels = []
  for label, count in zip(labels, label_counts):
    # number of samples needed according to proportion 
    num_to_append = int(class_counts[max_count_index] * label_counts[label]/class_counts[get_parent_class(label)]) 
    final_indices.append(np.random.choice(indices_by_label[label], size = num_to_append))
    final_labels.append(np.repeat(label, num_to_append))

  final_indices = np.concatenate(final_indices)
  final_labels = np.concatenate(final_labels)

  shuffle_indices = np.arange(len(final_indices))
  np.random.shuffle(shuffle_indices)
  final_sequences = all_sequences[final_indices[shuffle_indices]]
  #final_sequences += np.random.normal(0, 0.05, final_sequences.shape)
  final_labels = final_labels[shuffle_indices]
  return final_sequences, final_labels
  #+ torch.normal(torch.zeros((1, 260, 1)), 0.05) 
  
  # Performs a 0.5 split because we will be adding more data into the training set later

noisy_train, noisy_test, noisy_train_labels, noisy_test_labels = train_test_split(np.array(undenoised_sequences), np.array(all_labels), test_size=0.32, random_state=69, stratify = np.array(all_labels))

noisy_final, noisy_final_labels = synthetic_data(noisy_train, noisy_train_labels)
#noisy_final =noisy_train[final_indices[shuffle_indices]]
#noisy_final_labels = final_labels[shuffle_indices]

# Convert the labels to their parent classes
for labels in (noisy_final_labels, noisy_test_labels):
  for i, v in enumerate(labels):
    labels[i] = get_parent_class(v)

noisy_final = torch.as_tensor(noisy_final)
noisy_final_labels = torch.as_tensor(noisy_final_labels) 

noisy_test = torch.as_tensor(noisy_test)
noisy_test_labels = torch.as_tensor(noisy_test_labels) 

# Do not oversample or generate synth data for testing
noisytrainingFeatures, noisytrainingLabels = noisy_final, noisy_final_labels
noisytestFeatures, noisytestLabels = noisy_test, noisy_test_labels

def denoise_signal(ecg_signal):
  # Perform daubhechies-6 wavelet decomposition
  coeffs = pywt.wavedec(ecg_signal, 'db6', mode='per')

  # Get mean absolute deviation of decomposed signal
  mean_abs_dev = np.mean(np.abs(coeffs[-1] - np.mean(coeffs[-1])))
  # Set threshold for wavelet damping
  threshold_const = 1.1
  uthresh = threshold_const*mean_abs_dev * np.sqrt(2 * np.log(len(ecg_signal)))
  
  coeffs[1:] = [pywt.threshold(coeff_val, value=uthresh, mode='hard') for
                coeff_val in coeffs[1:]]
  return pywt.waverec(coeffs, 'db6', mode='per')

all_sequences = torch.as_tensor(np.array(all_sequences)) # runtime for this cell went from ~3.5min to ~0sec. by changing method of converting all_* to a torch tensor

# Must denoise data before inputting into the CNN 

denoised_intermediate = torch.stack([
                    torch.from_numpy(denoise_signal(x_i.squeeze())) for i, x_i in enumerate(torch.unbind(all_sequences, dim=0), 0) ], dim=0)
denoised_sequences = denoised_intermediate[:, None, :]

# Performs a 0.32 split because we will be adding more data into the training set later

train_sequences, test_sequences, train_labels, test_labels = train_test_split(np.array(denoised_sequences), np.array(all_labels), test_size=0.32, random_state=69, stratify = np.array(all_labels))

final_sequences, final_labels = synthetic_data(train_sequences, train_labels)

# Convert the labels to their parent classes
for labels in (final_labels, test_labels):
  for i, v in enumerate(labels):
    labels[i] = get_parent_class(v)

final_sequences = torch.as_tensor(final_sequences)
final_labels = torch.as_tensor(final_labels) 

test_sequences = torch.as_tensor(test_sequences)
test_labels = torch.as_tensor(test_labels) 

# Do not oversample or generate synth data for testing
trainingFeatures, trainingLabels = final_sequences, final_labels
testFeatures, testLabels = test_sequences, test_labels

AllFeatures, AllLabels = torch.cat((final_sequences, test_sequences)), torch.cat((final_labels, test_labels)) 

class ECGDataset(Dataset):

    def __init__(self, features, labels):
     self.features = features.float() # input to torch.nn.Conv1d should be of shape (batch size, # channels, #length of sequence) ; use .squeeze(3) to get this ; also CNN needs input of type float
     self.labels = labels
     self.n_samples = labels.shape[0]

    def __getitem__(self, index):
      return self.features[index], self.labels[index]
    def __len__(self):
      return self.n_samples
  
trainingDataSet = ECGDataset(trainingFeatures, trainingLabels)
testDataSet = ECGDataset(testFeatures, testLabels)

noisytrainingDataSet = ECGDataset(noisytrainingFeatures, noisytrainingLabels)
noisytestDataSet = ECGDataset(noisytestFeatures, noisytestLabels)

CrossValidationDataSet = ECGDataset(AllFeatures, AllLabels)

#CNN Model

class CNN(Module):   
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn_layers = Sequential(

            # Defining a 1D convolution layer
            Conv1d(1,5, kernel_size=3, stride=1), # kernel size = 3 in paper
            
            
            LeakyReLU(),
            MaxPool1d(kernel_size=2, stride=2),
            
            # Defining another 1D convolution layer
            Conv1d(5, 10, kernel_size=4, stride=1),

            LeakyReLU(),
            MaxPool1d(kernel_size=2, stride=2),

            #And another
            Conv1d(10, 20, kernel_size=4, stride=1), # 10, 20 in paper
            
            LeakyReLU(),
            MaxPool1d(kernel_size=2, stride=2)
            )

        self.flatten = Flatten()

        self.linear_layers = Sequential(
            #Linear layer with 30 neurons
            Linear(20*30,30),  
            LeakyReLU(),

            #Linear layer with 20 neurons
            Linear(30, 20), 
            LeakyReLU(),

            #Linear layer with 5 neurons
            Linear(20, 5), 

        )
    
    def forward(self, features):
        features = self.cnn_layers(features)
        features = self.flatten(features)
        features = self.linear_layers(features)
        return features
  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #change colab runtime to GPU
from torchsummary import summary
test_model = CNN()
test_model.to(device)
summary(test_model, input_size = (1, 260))

class LSTM_CNN(Module):
  """This class contains the CNN and LSTM model."""
  def __init__(self):
      super(LSTM_CNN, self).__init__()

      self.cnn_layers = Sequential(

          # Defining a 1D convolution layer (1 channel in, 32 out)
          Conv1d(1,32, kernel_size=3, stride=1),
          LeakyReLU(inplace=True),
          MaxPool1d(kernel_size=2, stride=1),
          
          # Defining another 1D convolution layer (32 channels in, 64 out)
          Conv1d(32, 64, kernel_size=4, stride=1),
          LeakyReLU(inplace=True),
          MaxPool1d(kernel_size=2, stride=1),

          # Defining another 1D convolution layer (64 channels in, 32 out)
          Conv1d(64, 32, kernel_size=4, stride=1),
          LeakyReLU(inplace=True),
          MaxPool1d(kernel_size=2, stride=1),
      )

      self.flatten = Flatten()

      #self.LSTM_layers = sequential()
      # self.LSTM_layers = LSTM(input=30*32,hidden_size=10,num_layers=1)
      self.LSTM_layers = torch.nn.LSTM(input_size=249,hidden_size=10,num_layers=1)   

      self.linear_layers = Sequential(
          #Linear layer (64 channels in, 128 out)
          Linear(320, 128), 
          LeakyReLU(inplace=True),

          #Linear layer with 5 neurons
          Linear(128, 5),
      )
    
  def forward(self, features):
    features = self.cnn_layers(features)
    # features = self.flatten(features)
    features, _ = self.LSTM_layers(features)
    features = self.flatten(features)
    features = self.linear_layers(features)
    return features
  
  def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


def predict(model, features):
  return Softmax(1)(model(features))
 

def train(model, EPOCHS = 20, BATCH = 32, trainingDataSet = trainingDataSet):
  criterion = CrossEntropyLoss()
  #Define SGD optimizer
  optimizer =  Adam(model.parameters()) #SGD(model.parameters(), lr=3e-4, momentum = 0.7, weight_decay=0.2)

  avg_train_loss_epoch = []
  train_acc_epoch = []

  # num_samples = 1000
  # sample_ds = torch.utils.data.Subset(trainingDataSet, np.arange(num_samples))
  # random_loader = torch.utils.data.RandomSampler(sample_ds)

  # trainLoader = torch.utils.data.DataLoader(sample_ds, sampler=random_loader, batch_size=BATCH)

  trainLoader = torch.utils.data.DataLoader(trainingDataSet, batch_size=BATCH)

  for epoch in range(EPOCHS):
      running_train_loss = 0 # 
      train_acc = 0      # training accuracy per batch

      for batchIndex, trainingData in tqdm(enumerate(trainLoader, 0),
                                           total=len(trainLoader)):

        #reset gradients
        optimizer.zero_grad()

        #forward-pass
        tInputs, tLabels = trainingData
        #print(model(tInputs))
        predictedY = model(tInputs)
        loss = criterion(predictedY, tLabels)
        loss.backward()

        if batchIndex == 0:
          plot_grad_flow(model.named_parameters())
          plt.show()

        running_train_loss += loss.item()*BATCH

        #update weights and reset gradient
        optimizer.step()

        train_prediction = predict(model, tInputs).argmax(1)
        #print(predict(model, tInputs), '\n')
        # if  np.unique(train_prediction).shape[0] !=1:
        #   print(train_prediction) 
        #print(train_prediction,'\n', tLabels)
        
        train_acc += (train_prediction.flatten() == tLabels).sum()
      train_acc_epoch.append(train_acc/trainingDataSet.features.shape[0])
      avg_train_loss_epoch.append(running_train_loss/trainingDataSet.features.shape[0])

      print('Epoch %04d  Training Loss %f Training Accuracy %.2f' % (epoch + 1, running_train_loss/trainingDataSet.features.shape[0], 100*train_acc/trainingDataSet.features.shape[0]))
  
  #Plot training loss
  plt.title("Train Loss")
  plt.plot(avg_train_loss_epoch, label="Train")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.show()

  plt.title("Train Accuracy")
  plt.plot(train_acc_epoch, label="Train")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy (%)")
  plt.show()
  
  # Run this cell to initialize CNN model before training (Phase 1)
cnn_model = CNN()
train(cnn_model)

# Test with noise removal
def test(model, BATCH = 32, DataSet = testDataSet):
  test_acc_epoch = []
  testLoader = torch.utils.data.DataLoader(DataSet, batch_size=BATCH)
  test_acc = 0
  class_sens = np.zeros(5)
  class_counts = np.zeros(5)
  class_spec = np.zeros(5)
  class_spec_counts = np.zeros(5)
  class_ppv = np.zeros(5)
  class_ppv_counts = np.zeros(5)
  for batchIndex, testData in tqdm(enumerate(testLoader, 0), total=len(testLoader)):
    #forward-pass
    tInputs, tLabels = testData
    test_prediction = predict(model, tInputs).argmax(1)
    labels, counts = np.unique(tLabels, return_counts = True)
    pred_labels, pred_counts = np.unique(test_prediction, return_counts = True)
    correct = test_prediction.flatten() == tLabels

    for label in range(5):
      label_predicted = test_prediction.flatten() == label 
      label_not_pred = ~ label_predicted
      tp = (label_predicted*correct).sum()
      tn = (label_not_pred*correct).sum()
      class_spec[label] += tn  # tn
      class_spec_counts[label] += label_not_pred.sum()  # tn + fn
      class_ppv[label] += tp
      class_ppv_counts[label] += label_predicted.sum()  # tp + fp
      class_sens[label] += tp 
    for index, label in enumerate(labels):
      class_counts[label] += counts[index]  # tp + fn
    test_acc += (test_prediction.flatten() == tLabels).sum()
  test_acc = test_acc/testDataSet.features.shape[0]
  class_sens = class_sens/class_counts
  class_ppv = class_ppv/class_ppv_counts
  class_spec = class_spec/class_spec_counts
  return test_acc, class_sens, class_ppv, class_spec

acc, class_sens, class_ppv, class_spec = test(cnn_model)
print(f"Accuracy: {acc.item()}\nClass Sensitivities: {class_sens}\nAverage Sensitivity: {class_sens.mean()}\nClass PPV: {class_ppv}\n Average PPV: {class_ppv.mean()}\nClass Specificities: {class_spec}\n Average Specificity: {class_spec.mean()}\n")

acc, class_sens, class_ppv, class_spec = test(lstm_cnn_model)
print(f"Accuracy: {acc.item()}\n Class Sensitivities: {class_sens}\nAverage Sensitivity: {class_sens.mean()}\nClass PPV: {class_ppv}\n Average PPV: {class_ppv.mean()}\nClass Specifities: {class_spec}\n Average Specificity: {class_spec.mean()}\n")

noisy_cnn_model = CNN()
train(noisy_cnn_model, 20, 32, noisytrainingDataSet)

#EDITS: Replaced TrainingDataSet with CrossValidationDataset, since the 10 folds are created using the entire dataset. 

#Initialize NN and list of trained models and loss function

criterion = CrossEntropyLoss()
k_folds = 10
kfold = KFold(n_splits=k_folds, shuffle=True) #(ten-fold cross-validation, 9/10 training 1/10 testing)

#Iterates for num of folds
class_acc = []
class_sens = []
class_ppv = []
class_spec = []
for fold, (train_ids, test_ids) in enumerate(kfold.split(CrossValidationDataSet)): 
  
  # Sample elements randomly from a given list of ids, no replacement.
  trainSubsampler = torch.utils.data.SubsetRandomSampler(train_ids)
  #testSubsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
  # Define data loaders for training and testing data in this fold
  trainLoader = torch.utils.data.DataLoader(CrossValidationDataSet, batch_size=10, sampler=trainSubsampler)
  #testLoader = torch.utils.data.DataLoader(CrossValidationDataSet, batch_size=256, sampler=testSubsampler)

  #Initialize CNN
  model = CNN()
  model.to(device)
  #Need to reset weights every fold
  
  #Define optimizer
  optimizer = Adam(model.parameters())

  #Train CNN
  epochs = 20
  for epoch in range(epochs):
    TotalLoss = 0
    #print("Epoch", epoch+1)
    for batchIndex, trainingData in enumerate(trainLoader, 0):

      #reset gradients
      optimizer.zero_grad()

      #forward-pass
      tInputs, tLabels = trainingData
      tInputs, tLabels = tInputs.to(device), tLabels.to(device)
      predictedY = model(tInputs)
      loss = criterion(predictedY, tLabels)

      #backward-pass
      loss.backward()

      #update weights and reset gradient
      optimizer.step()
      optimizer.zero_grad()

      #Display performance
      TotalLoss += loss.item()
      if (batchIndex+1) % 2000 == 0:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %(epoch + 1, batchIndex + 1, TotalLoss / 2000))
        TotalLoss = 0.0
  
  # Test on test set, report accuracy, specificity and sensitivity
  test_prediction = predict(model, CrossValidationDataSet.features[test_ids]).argmax(1)
  conf = confusion_matrix(CrossValidationDataSet.labels[test_ids], test_prediction)
  acc = []
  sens = []
  ppv = []
  spec = []
  for i in range(5):
    tp = conf[i,i] 
    indices = np.append(np.arange(i), np.arange(i+1, 5)) # all except ith index
    tn = conf[indices, indices].sum()
    fp = conf[i, indices].sum()
    fn = conf[indices, i].sum()
    acc.append((tp + tn)/ (tp + tn + fp + fn))
    sens.append(tp/(tp + fn))
    ppv.append(tp/(tp + fp))
    spec.append(tn/(tn + fp))
  class_acc.append(np.array(acc))
  class_sens.append(np.array(sens))
  class_ppv.append(np.array(ppv))
  class_spec.append(np.array(spec))

  print("Training for fold %d has been completed, saving model" %(fold))

  # Saving the model
  save_path = f'./model-fold-{fold}.pth'
  torch.save(model.state_dict(), save_path)

# Get average values for model performance measures
avg_acc = np.array(class_acc).mean(0)
avg_sens = np.array(class_sens).mean(0)
avg_ppv = np.array(class_ppv).mean(0)
avg_spec = np.array(class_spec).mean(0)
