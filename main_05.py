import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal
from matplotlib.animation import ArtistAnimation


def dft(x):
    """
    x: 1D array of f(x) to compute F(nu)
    n: number of data points
    T: sampling period
    dt = T/n
    1/dt: sampling frequency
    0.5/dt: critical frequency

    Returns array F(nu)
    """
    a = np.copy(x)
    output = []
    for k in range(a.shape[0]):
        f_k = 0
        for n in range(a.shape[0]):
            f_k += a[n] * np.exp(-2j * np.pi * k * n/a.shape[0])
        output.append(f_k)
    return output


def prepare_time(n, T):
    t_min = 0
    t_max = T
    return np.linspace(t_min, t_max, n, endpoint=False)


def prepare_freq(n, T):
    """
    Return DFT frequencies.
    n: number of samples (window size?)
    T: sample spacing (1/sample_rate)
    """
    freq = np.empty(n)
    scale = 1/(n * T)

    if n % 2 == 0:  # Even
        N = int(n/2 + 1)
        freq[:N] = np.arange(0, N)
        freq[N:] = np.arange(-(n/2) + 1, 0)
    else:
        N = int((n-1)/2)
        print(N)
        freq[:N] = np.arange(0, N)
        freq[N:] = np.arange(-N - 1, 0)

    return freq*scale


def load_wav(filename, preparetime=False):
    fs_rate, signal = wavfile.read(filename)
    if len(signal.shape) == 2:
        signal = signal.sum(axis=1)/2  # Povpreci oba channela
    n = signal.shape[0]  # Number of samples
    secs = n / fs_rate
    T = 1/fs_rate
    print("Sampling frequency: {}\nSample period: {}\nSamples: {}\nSecs: {}\n".format(fs_rate, T, n, secs))
    if preparetime:
        # t = np.linspace(0, secs, n)
        t = np.arange(0, secs, T)
        return signal, t
    else:
        return signal, fs_rate, n


def analyze_wav(filename, splits=10, onesided=False):
    fs_rate, signal = wavfile.read(filename)
    if len(signal.shape) == 2:
        signal = signal.sum(axis=1)/2  # Povpreci oba channela
    n = signal.shape[0]  # Number of samples
    secs = n / fs_rate
    T = 1/fs_rate
    interval = secs/splits
    print("Sampling frequency: {}\nSample period: {}\nSamples: {}\nSecs: {}\n".format(fs_rate, T, n, secs))
    ft = np.fft.fft(signal)
    t = np.arange(0, secs, T)
    if onesided:
        return t, signal, prepare_freq(np.array(ft).size//2, t[1]-t[0]), ft[:n//2], n, fs_rate
    return t, signal, prepare_freq(np.array(ft).size, t[1]-t[0]), ft, n, fs_rate


def wav_spectrum(filename, splits=100, onesided=False):  # Broken too
    fs_rate, signal = wavfile.read(filename)
    if len(signal.shape) == 2:
        signal = signal.sum(axis=1)/2  # Povpreci oba channela
    n = signal.shape[0]  # Number of samples
    secs = n / fs_rate
    T = 1/fs_rate
    print("Sampling frequency: {}\nSample period: {}\nSamples: {}\nSecs: {}\n".format(fs_rate, T, n, secs))

    interval = int(n/splits)
    spectrum = []
    freq_spectrum = []
    for split in range(0, splits):
        data = signal[split*interval:(split+1)*interval]
        if onesided:
            spectrum.append(np.fft.fft(data)[:interval//2])
            freq_spectrum.append(prepare_freq(interval//2, T))
        else:
            spectrum.append(np.fft.fft(data))
            freq_spectrum.append(prepare_freq(interval, T))
    t = np.arange(0, secs, T)
    # return t, signal, prepare_freq(np.array(ft).size, t[1]-t[0]), ft, n, fs_rate
    return t, freq_spectrum, spectrum


#def correlate(g, h):  # Broken because its shit
#    """
#    USAGE:
#        Calculate signal correlation for two arrays
#
#    INPUT:
#        g, h: input signal arrays of same dimension
#        n: sample lag
#
#    RETURNS:
#        phi: array representing correlation phi(n)
#    """
#    N = g.shape[0]  # Dimension of signal
#    n = np.arange(0, N)
#    # g_pad = np.append(np.copy(g), np.append(np.repeat(0, N - 1), g[0]))  # Add first value to close period?
#    # g_pad = np.append(np.copy(g), np.repeat(0, N - 1))
#    # h_pad = np.append(np.copy(h), np.append(np.repeat(0, N - 1), h[0]))
#    # h_pad = np.append(np.copy(h), np.repeat(0, N - 1))
#
#    G = np.fft.ifftshift(np.fft.fft(g))
#    H = np.fft.ifftshift(np.fft.fft(h))
#    phi = 1/(N - n) * np.fft.ifft(G * np.conj(H))
#    phi = np.fft.fftshift(phi)
#
#    return np.real(phi)

def correlate(g, h, norm_lenght=True, norm=False, pad=True):
    """
    Version 2 of correlate function. Can take signals of different length
    and normalise to the shorter signal sample count. Normalisation part is fucky and breaks everything.
    Outputs signal at half the sample rate of inputs!
    """
    first = False
    sam1 = g.shape[0]
    sam2 = h.shape[0]
    x_range = np.arange(-sam2, sam2)
    if norm_lenght:
        if sam1 > sam2:
            first = True
            g = g[:sam2]
            x_range = np.arange(-sam2, sam2)
        elif sam1 < sam2:
            h = h[:sam1]
            x_range = np.arange(-sam1, sam1)
        elif sam1 == sam2:
            x_range = np.arange(-sam2, sam2)
        else:
            raise ImportError("Something's fucky..")

    if norm:
        g = (g - np.mean(g))/np.std(g)
        h = (h - np.mean(h))/np.std(h)
    else:
        g = g
        h = h

    if pad:
        if first:
            g = np.concatenate((g, np.zeros(sam2)))
            h = np.concatenate((h, np.zeros(sam2)))
        else:
            g = np.concatenate((g, np.zeros(sam1)))
            h = np.concatenate((h, np.zeros(sam1)))

    G = np.fft.fft(g)
    H = np.fft.fft(h)
    p = G * np.conj(H)
    pi = 1/sam1*np.fft.ifft(p)
    pi = np.fft.ifftshift(pi)/np.max(pi)

    return x_range, np.real(pi)


def correlate2(g, h):  # Simple and quick version
    N = g.shape[0]
    g = np.concatenate((g, np.zeros(N)))
    h = np.concatenate((h, np.zeros(N)))
    n = np.arange(0, 2*N)

    G = np.fft.fft(g)
    H = np.fft.fft(h)
    p = G * np.conj(H)
    pi = 1/N*np.fft.ifft(p)
    return n, np.real(np.fft.ifftshift(pi))

# def correlate(g, h):
#     return np.convolve(g, reverse_conjugate(h))


def reverse_conjugate(x):
    reverse = (slice(None, None, -1),) * x.ndim
    return x[reverse].conj()


# Spectrogram plot
filename = r"src_Pulsar\B1933+16_ALFA.wav"
signal, fs, samples = load_wav(filename)
xdata, data = correlate(signal, signal)
plt.specgram(data, Fs=fs, scale="dB", cmap="plasma")
plt.title("Spektrogram {}".format(filename))
plt.ylabel('Frekvenca [Hz]')
plt.xlabel('Čas [s]')
plt.yscale("linear")
plt.colorbar(label="Moč [dB]")
plt.show()


# Autocorrelation/correlation test plot
# x = np.linspace(-3, 3, 100)
# y1 = np.append(np.zeros(40), np.append(np.repeat(1, 20), np.zeros(40)))  # Box signal
# y2 = np.append(np.zeros(40), np.append(np.arange(20, 0, -1)/20, np.zeros(40)))  # Sawtooth signal
#
# # y1 = np.sin(2*np.pi*x) + 3*np.random.random(100)
# # y2 = np.cos(2*np.pi*x) + 3*np.random.random(100)
# # y1 = np.random.uniform(-1, 1, 10000)
# # y2 = np.random.uniform(-1, 1, 10000)
# xdata, data = correlate(y1, y2, norm=False)
# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.plot(x, y1, c="#AF1F87")
# ax1.set_title("Poskusni signal 1")
# ax1.set_xlabel("Čas [s]")
# ax1.set_ylabel("Normirana amplituda")
# ax2.plot(x, y2, c="#782766")
# ax2.set_xlabel("Čas [s]")
# ax2.set_title("Poskusni signal 2")
# ax2.set_ylabel("Normirana amplituda")
# fig.subplots_adjust(top=0.94, hspace=0.45, bottom=0.13)
# plt.show()
#
# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.plot(data, c="#46F29C")
# ax1.set_xlabel("Vzorci")
# ax1.set_title("DIY correlate(sig1, sig2)")
# ax1.set_ylabel("Normirana amplituda")
# scipy_data = scipy.signal.correlate(y1, y2, method="fft")
# scipy_data = scipy_data/np.max(scipy_data)
# ax2.plot(scipy_data, c="#5BC8AF")
# ax2.set_title("scipy.signal.correlate(sig1, sig2)")
# ax2.set_xlabel("Vzorci")
# ax2.set_ylabel("Normirana amplituda")
# fig.subplots_adjust(top=0.94, hspace=0.45, bottom=0.13)
# plt.show()

# Error between correlate(g, h) and scipy builtin
# error = np.abs(scipy_data - data[:-1])/scipy_data
# plt.plot(error, color="#CA2E55")
# plt.title(r"Relativna napaka $\mathrm{scipy}$ vgrajene in DIY korelacijske funkcije")
# plt.ylabel(r"Relativna napaka $\mathrm{abs}\left[\frac{scipy - correlate}{scipy}\right]$")
# plt.xlabel("Vzorci")
# plt.yscale("log")
# plt.show()

# Crosscorrelation between 2 signals
filename = r"src_Pulsar\B1933+16_ALFA.wav"
filename2 = r"src_Pulsar\B1933+16_ALFA.wav"
outputname = r"src_Pulsar\B1933+16_ALFA_autocorr.wav"
signal, fs, samples = load_wav(filename)
signal2, fs2, samples2 = load_wav(filename2)
# signal2 = np.random.uniform(0, 1, signal2.shape[0])  # Debug signal

xdata, data = correlate(signal, signal2)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6))

ax1.plot(np.arange(0, samples)/fs, signal, c="#FF84E8")
ax1.set_title("Signal {}".format(filename))
ax1.set_xlabel("Čas [s]")
ax1.set_ylabel("Normirana amplituda")

ax2.plot(np.arange(0, samples2)/fs2, signal2, c="#7F2CCB")
ax2.set_title("Signal {}".format(filename2))
ax2.set_xlabel("Čas [s]")
ax2.set_ylabel("Normirana amplituda")

# outputname = str(input("Filename for correlation:  ") + ".wav")

ax3.set_title("Korelacija signalov: {}".format(outputname))
ax3.plot(xdata, data, c="#42489D")
ax3.set_xlabel("Zamik vzrocev")
ax3.set_ylabel("Normirana amplituda")

fig.subplots_adjust(top=0.94, hspace=0.55, bottom=0.08, left=0.16)
scipy.io.wavfile.write(outputname, fs//2, data)  # Output .wav file
plt.show()

# Spectrogram
plt.specgram(data, Fs=fs, scale="dB", cmap="plasma")
plt.title("Spektrogram {}".format(outputname))
plt.ylabel('Frekvenca [Hz]')
plt.xlabel('Čas [s]')
plt.yscale("linear")
plt.colorbar(label="Moč [dB]")
plt.show()

# Standard Fourier spectrum
fig, ax = plt.subplots(figsize=(7, 5))
time, sig, freq, trans, sam, fs_rate = analyze_wav(outputname, onesided=True)
power = np.abs(trans)**2
t = np.arange(sam/2)
plt.scatter(freq, power, s=3, c=t, cmap="plasma")
plt.title("Cel spekter {}".format(outputname))
plt.xlabel("Frekvenca [Hz]")
plt.ylabel("Amplituda")
plt.xscale("log")
plt.yscale("log")
plt.colorbar(label="Vzorec")
plt.show()

# Multiple correlations plot (works but looks horrible)
# base = r"src\bubomono.wav"  # Base file
# signal, fs, samples = load_wav(base)
# filenames = [r"src\bubomono.wav", r"src\bubo2mono.wav", r"src\mix.wav",
#              r"src\mix1.wav", r"src\mix2.wav", r"src\mix22.wav"]
# for file in np.flip(filenames):
#     signal2, fs2, samples2 = load_wav(file)
#     xdata, data = correlate(signal, signal2)
#     plt.plot(xdata, data, label="{}".format(file), alpha=0.5)
#
# plt.legend()
# plt.show()

# Multiple autocorrelations plot
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 4))
# file1 = r"src\bubomono.wav"
# signal1, fs1, samples1 = load_wav(file1)
# xdata1, data1 = correlate(signal1, signal1)
# ax1.plot(xdata1, data1, label="Avtokorelacija {}".format(file1), c="#650369")
# ax1.set_title(file1)
# ax1.set_ylim(-1, 1)
# ax1.set_xlabel("Zamik")
#
# file2 = r"src\bubo2mono.wav"
# signal2, fs2, samples2 = load_wav(file2)
# xdata2, data2 = correlate(signal2, signal2)
# ax2.plot(xdata2, data2, label="Avtokorelacija {}".format(file2), c="#94199A")
# ax2.set_title(file2)
# ax2.set_ylim(-1, 1)
# ax2.set_xlabel("Zamik")
#
# file3 = r"src\mix.wav"
# signal3, fs3, samples3 = load_wav(file3)
# xdata3, data3 = correlate(signal3, signal3)
# ax3.plot(xdata3, data3, label="Avtokorelacija {}".format(file3), c="#65529F")
# ax3.set_title(file3)
# ax3.set_ylim(-1, 1)
# ax3.set_xlabel("Zamik")
#
# for ax in fig.get_axes():
#     ax.label_outer()
#
# plt.suptitle("Avtokorelacijske funkcije vseh posnetkov")
# fig.subplots_adjust(top=0.85, hspace=0.55, bottom=0.12)
# plt.show()
#
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 4))
# file1 = r"src\mix1.wav"
# signal1, fs1, samples1 = load_wav(file1)
# xdata1, data1 = correlate(signal1, signal1)
# ax1.plot(xdata1, data1, label="Avtokorelacija {}".format(file1), c="#92E98A")
# ax1.set_title(file1)
# ax1.set_ylim(-1, 1)
# ax1.set_xlabel("Zamik")
#
# file2 = r"src\mix2.wav"
# signal2, fs2, samples2 = load_wav(file2)
# xdata2, data2 = correlate(signal2, signal2)
# ax2.plot(xdata2, data2, label="Avtokorelacija {}".format(file2), c="#59C987")
# ax2.set_title(file2)
# ax2.set_ylim(-1, 1)
# ax2.set_xlabel("Zamik")
#
# file3 = r"src\mix22.wav"
# signal3, fs3, samples3 = load_wav(file3)
# xdata3, data3 = correlate(signal3, signal3)
# ax3.plot(xdata3, data3, label="Avtokorelacija {}".format(file3), c="#85BBDB")
# ax3.set_title(file3)
# ax3.set_ylim(-1, 1)
# ax3.set_xlabel("Zamik")
#
# for ax in fig.get_axes():
#     ax.label_outer()
#
# plt.suptitle("Avtokorelacijske funkcije vseh posnetkov")
# fig.subplots_adjust(top=0.85, hspace=0.55, bottom=0.12)
# plt.show()
