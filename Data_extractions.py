
import mne
import numpy as np
from Utilitys import sub_name, unify_names     
import pickle
from mne.io import Raw


def extract_data_from_subject(root_dir: str, n_s: int, datatype: str, selected_channels: list) -> tuple:
   
    data = dict()
    y = dict()
    n_b_arr = [1, 2, 3]
    datatype = datatype.lower()

    for n_b in n_b_arr:
        # Name correction if N_Subj is less than 10
        num_s = sub_name(n_s)

        y[n_b] = load_events(root_dir, n_s, n_b)

        if datatype == "eeg":
            # Load data and events
            file_name = (
                f"{root_dir}/derivatives/{num_s}/ses-0{n_b}/{num_s}_ses-0{n_b}_eeg-epo.fif"             # noqa
            )
            X = mne.read_epochs(file_name, verbose='WARNING')
            X.pick_channels(selected_channels)  # Select specific channels
            data[n_b] = X._data

        elif datatype == "exg":
            file_name = (
                f"{root_dir}/derivatives/{num_s}/ses-0{n_b}/{num_s}_ses-0{n_b}_exg-epo.fif"             # noqa
            )
            X = mne.read_epochs(file_name, verbose='WARNING')
            data[n_b] = X._data

        elif datatype == "baseline":
            file_name = (
                f"{root_dir}/derivatives/{num_s}/ses-0{n_b}/{num_s}_ses-0{n_b}_baseline-epo.fif"        # noqa
            )
            X = mne.read_epochs(file_name, verbose='WARNING')
            #X.pick_channels(selected_channels)  # Select specific channels
            data[n_b] = X._data

        else:
            raise ValueError("Invalid Datatype")

    X_stacked = np.vstack((data[1], data[2], data[3]))
    Y_stacked = np.vstack((y[1], y[2], y[3]))

    return X_stacked, Y_stacked



def load_events(root_dir: str, n_s: int, n_b: int):
    
    num_s = sub_name(n_s)

    # Create file name
    file_name = f"{root_dir}/derivatives/{num_s}/ses-0{n_b}/{num_s}_ses-0{n_b}_events.dat"     

    # Load events
    events = np.load(file_name, allow_pickle=True)

    return events
