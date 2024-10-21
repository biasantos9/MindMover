import numpy as np
import matplotlib.pyplot as plt
import time
import joblib
import mne


from Data_extractions import  extract_data_from_subject
from Data_processing import filter_by_condition, filter_by_class

from scipy.signal import filtfilt, firwin
from scipy.signal import welch
from scipy.integrate import simps

import serial_comunnication

# General inicializations
# change directory as necessary 
directory = 'C:/Users/serpr/OneDrive - Universidade do Porto/Documentos/4ºano_2ºsemestre/robotica/projeto/inner-speech-recognition'

#conditions according to the data set, change as necessary 
subject = 1
datatype ='eeg'
condition = "inner"
classe = "all"

#channels from left hemisphere
# selected_channels = ['A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18',
#                      'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31', 'C32',
#                      'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19',
#                      'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32'] 

#channels from frontal and parietal region of both hemispheres
selected_channels =['A1', 'A2', 'A3', 'A4','A19', 'A23', 'B1', 'B2', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 
                    'B26', 'B27', 'B28', 'B29', 'B30', 'B31', 'B32', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 
                    'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31', 'C32', 'D1', 'D2', 
                    'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23',
                    'D24', 'D25', 'D26', 'D27', 'D28']

fs = 256
lowcut = 8
highcut = 80
filter_order = 10


#FIR filter for signal processing
def fir_bandpass_filter(X, fs, lowcut, highcut, filter_order):
    # normalized cut-off frequencies
    nyquist_freq = 0.5 * fs
    lowcut_normalized = lowcut / nyquist_freq
    highcut_normalized = highcut / nyquist_freq

    # filter design
    lowpass_filter = firwin(filter_order, highcut_normalized, pass_zero=True, window='hamming')
    highpass_filter = firwin(filter_order, lowcut_normalized, pass_zero=True, window='hamming')

    # applying lowpass filter
    X_lowpass = filtfilt(lowpass_filter, 1, X, axis=-1)

    # applying highpass filter
    X_highpass = filtfilt(highpass_filter, 1, X_lowpass, axis=-1)

    return X_highpass




# Simulated EEG data and baseline acquisition function
def acquire_eeg_signal(directory, subject, datatype, selected_channels):
    X, Y = extract_data_from_subject(directory, subject, datatype, selected_channels)
    
    X_filt, Y_filt = filter_by_condition(X, Y, condition)
    X_filt, Y_filt = filter_by_class(X_filt,Y_filt,classe)

    #Convert to microvolts
    X_resize = X_filt * (10**6)

    num_trials = X_resize.shape[0]
    num_channels = X_resize.shape[1]
    num_samples = X_resize.shape[2]

    # eeg data inicialization
    eeg_data = np.empty((num_channels, num_trials * num_samples))

    # concatenating trials 
    for i in range(num_trials):
        X_data = X_resize[i, :, :]
        eeg_data[:, i * num_samples: (i + 1) * num_samples] = X_data

    return eeg_data

def acquire_baseline(subject, selected_channels, fs, lowcut, highcut, filter_order):
    # directory - change as necessary 
    root_dir = r"C:\Users\serpr\OneDrive - Universidade do Porto\Documentos\4ºano_2ºsemestre\robotica\projeto\inner-speech-recognition\derivatives\sub-"
    n_s = subject
    baseline_data=[]
    for n_session in [1, 2, 3]:
        # full path
        file_name = f"{root_dir}{n_s:02d}\ses-0{n_session}\sub-{n_s:02d}_ses-0{n_session}_baseline-epo.fif"
        # loads data 
        baseline_epochs = mne.read_epochs(file_name, verbose='WARNING')
        # selects desired channels
        baseline_epochs.pick_channels(selected_channels)
        baseline = baseline_epochs._data
        baseline_data.append(baseline)
        
    baseline_data = np.array(baseline_data)
    average_baseline = np.mean(baseline_data, axis=0)
    average_baseline = average_baseline * (10**6)
    average_baseline = fir_bandpass_filter(average_baseline,  fs, lowcut, highcut, filter_order)
    average_baseline = average_baseline.reshape(average_baseline.shape[1],average_baseline.shape[2])
   
    t_start_baseline =10
    t_end_baseline = 15
    t_max = average_baseline.shape[1]
    start_b = max(round(t_start_baseline * fs), 0)
    end_b = min(round(t_end_baseline * fs), t_max)
    average_baseline = average_baseline[:, start_b:end_b]
    
    return(average_baseline)


#loading data
eeg_signal = acquire_eeg_signal(directory, subject, datatype, selected_channels)
print(eeg_signal.shape)

baseline = acquire_baseline(subject, selected_channels,  fs, lowcut, highcut, filter_order)

#Signal processing
def baseline_removal(X_window_b, X_window):
    
    num_channels = X_window.shape[0]
    num_channels_b = X_window_b.shape[0]

    assert num_channels == num_channels_b, "Mismatch in number of channels between X_window and X_window_b"

    # Average signal for each channel and each trial in X_window_b
    average_signal = np.mean(X_window_b, axis=1)  # Shape: (num_channels, samples)

    # Initializing array to store corrected data
    X_window_corrected = np.zeros_like(X_window)  

    
    for j in range(num_channels):
        # Subtract the average signal from the current trial and channel
        X_window_corrected[ j, :] = X_window[ j, :] - average_signal[j]

    return X_window_corrected


def process_eeg_signal(eeg_signal, baseline):

    min_length = 30 
    if eeg_signal.shape[1] < min_length:
        raise ValueError(f"Input signal length must be at least {min_length} samples.")

    # signal processing
    # filtering - FIR 
    signal_filtered = fir_bandpass_filter(eeg_signal, fs, lowcut, highcut, filter_order)
    signal_filtered = baseline_removal(signal_filtered, baseline)

    return signal_filtered


# ERP detection and trigger action
def erp_detection(signal, max_threshold):
    erp = False
    
    if np.any(signal <= max_threshold):
        erp = True
    
    return erp


#feature extraction and classification
def compute_absolute_power(data, idx, f_res):
    
    num_channels = data.shape[0]
    absolute_power = np.zeros(num_channels)

    for channel_index in range(num_channels):
        # Compute the absolute delta power for the current trial and channel
        power = simps(data[channel_index, idx], dx=f_res)
        # Append the result to the list
        absolute_power[channel_index] = power

    return absolute_power


def compute_alpha_beta_ratio(data, alpha_idx, beta_idx, f_res):
    
    # Compute absolute power for alpha and beta bands
    alpha_power = compute_absolute_power(data, alpha_idx, f_res)
    beta_power = compute_absolute_power(data, beta_idx, f_res)

    # Compute the ratio between alpha and beta waves
    alpha_beta_ratio = alpha_power / beta_power

    return alpha_beta_ratio


# def compute_variance(data, idx):
#     data_within_beta_band = data[:, idx]
#     channel_variances = np.var(data_within_beta_band, axis=1)
#     return channel_variances


def classification(X_data):
    # load model with joblib
    loaded_model = joblib.load('selectedmodel.sav')

    # evaluate model 
    y_predict = loaded_model.predict(X_data)

    return y_predict

    

def classify_next_2_seconds(signal_filtered):
   
    #winodw size
    length = signal_filtered.shape[1]
    nperseg = length//5
   
    f, Pxx = welch(signal_filtered, fs=fs, nperseg=nperseg) 
    res = f[1]-f[0]
    low = 13
    high = 30
    idx = np.logical_and(f >= low, f <= high)

    alpha_low = 8
    alpha_high = 13
    alpha_idx = np.logical_and(f >= alpha_low, f <= alpha_high)

    #considering beta band variance as selected feature
    # variance_beta_welch = compute_variance(Pxx, idx)
    # variance_beta_welch =np.transpose(variance_beta_welch)
    # variance_beta_welch= variance_beta_welch.reshape(-1,87)
    # prediction = classification(variance_beta_welch)

    #considering alpha/beta ratio as selected feature
    alpha_beta_ratio_welch = compute_alpha_beta_ratio(Pxx,alpha_idx,idx,res)
    alpha_beta_ratio_welch  = np.transpose(alpha_beta_ratio_welch )
    alpha_beta_ratio_welch = alpha_beta_ratio_welch .reshape(-1,87)
    prediction = classification(alpha_beta_ratio_welch)


    return prediction


#variable inicializations
max_threshold= -200
std = 20

last_erp_time = time.time()
fig, ax = plt.subplots()
chunk_size = 128  
buffer_size = 2596  
fs = 256 

# buffer incialization
data_buffer = np.zeros((87, buffer_size))
filtered_buffer = np.zeros((87, buffer_size))

# time variables inicialization
plot_update_interval = 1 / fs
last_plot_update = time.time()
prediction_due = False
erp_blocked_until = 0
prediction_start_time = None

start_time = time.time()
segment_length = int(2.5 * fs)  
time_elapsed = 0

#Main loop
for i in range(0, eeg_signal.shape[1], chunk_size):
    chunk = eeg_signal[:, i:i + chunk_size]
    if chunk.shape[1] < chunk_size:
        break  
    
    filtered_chunk = process_eeg_signal(chunk, baseline)  
    chunk_sum = np.sum(filtered_chunk, axis=0)
    
    # erp detection
    if time.time() >= erp_blocked_until:
        erp = erp_detection(chunk_sum, max_threshold)
        if erp:
            print('ERP detected at ' + str(time_elapsed))
            last_erp_time = time.time()
            prediction_due = True
            prediction_start_time = time.time() + 0.5  # begins prediction 0.5s after erp detection
            erp_blocked_until = time.time() + 2.5  # blocks erp detection for 2.5s

    
    if prediction_due and time.time() >= prediction_start_time:
        # determination of the chunk indices
        start_idx = int((prediction_start_time - start_time) * fs)
        end_idx = start_idx + segment_length

        # checks if we have 2.5s of data 
        if end_idx <= eeg_signal.shape[1]:
            next_signal = eeg_signal[:, start_idx:end_idx]
            
            if next_signal.shape[1] == segment_length:
                predicted_class = classify_next_2_seconds(next_signal)
                print('Predicted class for next 2.5 seconds:', predicted_class)
                serial_comunnication.send(predicted_class)
                prediction_due = False  

    # updates plot
    current_time = time.time()
    if current_time - last_plot_update >= plot_update_interval:
        ax.clear()
        for j, channel_data in enumerate(eeg_signal[:, :i + chunk_size]):
            ax.plot(np.arange(len(channel_data)) / fs, channel_data, label=f'Canal {j + 1}')
        ax.set_title('EEG signal')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (uV)')
        plt.pause(0.001)  
        plt.draw()
        last_plot_update = current_time

    # updates time
    time_elapsed = (i + chunk_size) / fs

    # waiting time for real time signal acquisition
    elapsed_time = time.time() - start_time
    expected_time = (i + chunk_size) / fs
    sleep_time = expected_time - elapsed_time
    if sleep_time > 0:
        time.sleep(sleep_time)