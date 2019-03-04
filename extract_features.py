
"""
Extract features based on a custom algorithm
1. Seperate signal into different frequency bands
2. Apply PCA to each band
3. Concatenate bands columns
4.
"""

from my_functions.processing import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def resample(x, newFs):
    idx = [1 if num % newFs == 0 else 0 for num in range(len(x))]
    idx = np.array(idx)
    #return idx
    return x[idx==1][:]


def rms(x):
    """
    Root-mean square function
    :param x: n-dim numpy array
    :return: rms of input
    """
    return np.sqrt(np.mean(np.square(x), axis=0))


def find_pca(x):
    """
    Principal component analysis
    :param x: n-dim numpy array
    :return: coef, score, latent
    """
    pca = PCA(n_components=x.shape[1])
    pca.fit(x);
    score = pca.transform(x)
    coef = pca.components_
    explained = pca.explained_variance_ratio_
    return coef, score, explained


def find_pca_contributions(x, coef):
    """
    Finds signal contributions on PCs
    :param x: original array
    :param coef: PC coefficients
    :return: contrinutions (rms power)
    """
    x = x - np.mean(x, axis=0)
    cont = np.zeros(coef.shape)
    for i, row in enumerate(coef):
        cont[:,i] = rms(x*row)
    return cont


def extract_features(data):

    ## Get parameters
    common_chans = data['channels']
    fs = data['fs']

    ## Feature extraction
    lps = [100, 200, 300, 400, 500, 600, 700, 800]
    hps = [200, 300, 400, 500, 600, 700, 800, 1500]

    input_data = []
    used_pcs = []
    for lp, hp in zip(lps, hps):
        band = apply_fancy_filter(data['raw'][:, common_chans], fs, lp, hp, order=10, ftype='bandpass')
        band = StandardScaler().fit_transform(band)
        coef, score, explained = find_pca(band)
        contributions = find_pca_contributions(band, coef)
        score = score[:,np.cumsum(explained)<=0.95] # Only PCs that explains 95% of the variance
        input_data.append(score)
        used_pcs.append(score.shape[1])

    input_data = np.concatenate(input_data, axis=1)
    emg_data = data['emg']

    # Find signal envelopes
    input_data = find_envelope(input_data, fs, N=512)
    emg_data = find_envelope(emg_data, fs, N=512)

    ## Recover actual signal part
    NEURAL = []
    EMG = []
    new_cut_times = []
    previous_length = 0
    new_length = 0

    for idx in range(len(data['cut-times'])):
        st = data['cut-times'][idx][0] + previous_length
        ed = data['cut-times'][idx][1] + previous_length
        previous_length = previous_length + data['trial-lengths'][idx]
        new_length = new_length + (ed - st)
        NEURAL.append(input_data[st:ed, :])
        EMG.append(emg_data[st:ed, :])
        new_cut_times.append(new_length)

    NEURAL = np.concatenate(NEURAL, axis=0)
    EMG = np.concatenate(EMG, axis=0)

    ## Downsample
    NEURAL = resample(NEURAL, 100)
    EMG = resample(EMG, 100)

    del (input_data, emg_data)

    ## Prepare input for regression
    data = {
        "input": NEURAL,
        "output": EMG,
        "cut-times": new_cut_times,
        "contributions": contributions,
        "num_of_components": used_pcs
    }

    del (NEURAL, EMG)

    return data