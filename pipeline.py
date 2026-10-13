import numpy as np
import scipy.io as sio
import mne

FS = 250
WANTED = ['POZ', 'OZ', 'O1', 'O2', 'PO3', 'PO4', 'PO5', 'PO6', 'PZ']


def read_ch_names(data_dir='data'):
    # channel names live in a separate .loc file, 4th column
    with open(f'{data_dir}/64-channels.loc') as f:
        return [l.split()[3] for l in f if len(l.split()) >= 4]


def load_subject(n, data_dir='data'):
    data = sio.loadmat(f'{data_dir}/S{n}.mat')['data']   # (64, 1500, 40, 6)
    freqs = sio.loadmat(f'{data_dir}/Freq_Phase.mat')['freqs'].flatten()

    # (targets, blocks, channels, samples) -> (240, 64, 1500), target-major
    x = data.transpose(2, 3, 0, 1).reshape(240, 64, 1500)
    labels = np.repeat(np.arange(40), 6)

    info = mne.create_info(read_ch_names(data_dir), sfreq=FS, ch_types='eeg')
    epochs = mne.EpochsArray(x * 1e-6, info, tmin=-0.5, verbose=False)  # uv -> v

    # keep occipital channels only
    upper = {c.upper(): c for c in epochs.ch_names}
    epochs.pick([upper[w] for w in WANTED])

    # 50 hz notch on the array, then wrap back into epochs
    d = mne.filter.notch_filter(epochs.get_data(), Fs=FS, freqs=50,
                                notch_widths=2, trans_bandwidth=2,
                                verbose=False)
    epochs = mne.EpochsArray(d, epochs.info, tmin=-0.5, verbose=False)
    epochs.filter(6, 90, verbose=False)

    return epochs.get_data(), labels, freqs   # (240, 9, 1500)