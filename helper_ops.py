import numpy as np
import scipy.signal as sp_sig
import scipy.io.wavfile


def rms(x):
    """Compute the root-mean-square value along the first axis."""
    sq = np.mean(np.square(x), axis=0)
    return np.sqrt(sq)


def slice_1dsignal(
    input_signal, window_size, winshift, minlength, left_context=256, right_context=256
):
    """Return context-padded windows from a 1D signal."""
    n_samples = input_signal.shape[0]
    slices = []

    for beg_i in range(0, n_samples, winshift):
        beg_i_context = beg_i - left_context
        end_i = beg_i + window_size + right_context
        if n_samples - beg_i < minlength:
            break
        if beg_i_context < 0:
            slice_ = np.concatenate(
                (
                    np.zeros((1, left_context - beg_i)),
                    np.array([input_signal[:end_i]]),
                ),
                axis=1,
            )
        elif end_i <= n_samples:
            slice_ = np.array([input_signal[beg_i_context:end_i]])
        else:
            slice_ = np.concatenate(
                (
                    np.array([input_signal[beg_i_context:]]),
                    np.zeros((1, end_i - n_samples)),
                ),
                axis=1,
            )
        slices.append(slice_)

    if not slices:
        return np.empty((0, window_size + left_context + right_context, 1))

    slices = np.vstack(slices)
    return np.expand_dims(slices, axis=2)


def QERB_calculation(bmm, cfs, fs):
    central = cfs.shape[0]
    samples = bmm.shape[1]
    half = samples // 2
    spectrum = np.zeros((samples, central))
    max_val = np.zeros(central)
    energy = np.zeros(central)
    bandwidth = np.zeros(central)
    qerb = np.zeros(central)

    for i in range(int(central)):
        spectrum[:, i] = (2 * abs(np.fft.fft(bmm[i, :])) / samples) ** 2
        max_val[i] = spectrum[:, i].max()
        energy[i] = spectrum[: half + 1, i].sum()
        bandwidth[i] = (energy[i] / max_val[i]) * fs / samples
        qerb[i] = cfs[i] / bandwidth[i]
    return qerb


def get_dpoae(tl_bmm, cf_location=0, sig_start=0):
    """Return the positive-frequency DPOAE magnitude spectrum for one CF."""
    oae_sig = tl_bmm[0, sig_start:, cf_location]
    oae_fft = np.fft.fft(oae_sig)
    nfft = oae_fft.shape[0]
    return np.absolute(oae_fft[: int(nfft / 2)]), nfft


def concatenate_tl_pred(tl_prediction):
    """Concatenate batched TL predictions along the time axis."""
    return np.expand_dims(np.vstack(tl_prediction), axis=0)


def undo_window(tl_prediction, winlength, winshift, ignore_first_set=0, fs=20e3):
    """Reconstruct a full signal from overlapping model-output windows."""
    del fs  # Kept for backward-compatible notebook calls.
    trailing_zeros = 0
    nframes = tl_prediction.shape[0]
    slength = (nframes - 1) * winshift + winlength
    tl_2d = np.zeros((slength, tl_prediction.shape[2]))
    scale_ = np.zeros((slength, 1))
    dummyones = np.ones((tl_prediction.shape[0], tl_prediction.shape[1]))
    sigrange = range(winlength)
    tl_2d[sigrange, :] = tl_2d[sigrange, :] + tl_prediction[0]
    scale_[sigrange, 0] = scale_[sigrange, 0] + dummyones[0]
    for i in range(1, nframes):
        sigrange = range(i * winshift + ignore_first_set, (i * winshift) + winlength)
        tl_2d[sigrange, :] = (
            tl_2d[sigrange, :] + tl_prediction[i, ignore_first_set:, :]
        )
        scale_[sigrange, 0] = scale_[sigrange, 0] + dummyones[i, ignore_first_set:]

    tl_2d /= scale_
    return np.expand_dims(tl_2d[trailing_zeros:, :], axis=0)


def wavfile_read(wavfile, fs=None):
    """Read a WAV file, normalize integer samples, and optionally resample."""
    fs_signal, speech = scipy.io.wavfile.read(wavfile)
    if fs is None:
        fs = fs_signal

    if np.issubdtype(speech.dtype, np.integer):
        max_value = np.iinfo(speech.dtype).max + 1.0
        speech = speech.astype(np.float32) / max_value

    if fs_signal != fs:
        signalr = sp_sig.resample_poly(speech, fs, fs_signal)
    else:
        signalr = speech

    return signalr, fs
