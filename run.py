import qam
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt


M = 64
Fs = 10e9
N = 100000
factor = 1 # upsampling factor

constel = qam.constellation(M)
qam.plot_constel(constel)
input = np.random.choice(M, (N,)).astype(int)
iq = qam.modulate(input, constel)

BER = []
EbN0_db = []
theory_BER = []
OSNR_dB = np.arange(10, 25)
for OSNR_val_dB in OSNR_dB:
    SNR_dB = OSNR_val_dB - 10 * np.log10(Fs / 12.5e9)
    EbN0 = 10 ** (SNR_dB / 10) / np.log2(M)
    EbN0_db.append(10 * np.log10(EbN0))
    SNR = 10 ** (-SNR_dB / 20)

    iq_up = sp.resample(iq, N * factor)
    iq_up = iq_up / np.sqrt(np.var(constel))

    i_noise = np.random.randn(N * factor)
    q_noise = np.random.randn(N * factor)
    noise = SNR * (i_noise + 1j * q_noise) / np.sqrt(2 / factor)
    iq_up_noisy = iq_up + noise

    iq_noisy = sp.resample(iq_up_noisy, N)
    
    output = qam.demodulate(iq_noisy, constel)
    BER.append(qam.ber(input, output, M))
    theory_BER.append(qam.theory_ber(EbN0, M))
    

#qam.save_ber(OSNR_dB, EbN0_db, BER, M)

plt.figure(figsize=(12,8))
qam.add_ber_plot(EbN0_db, OSNR_dB, theory_BER)
qam.add_ber_plot(EbN0_db, OSNR_dB, BER, 'ro')
plt.legend(['Theory', 'Simulation'])
plt.suptitle('QAM{}'.format(M))
plt.tight_layout()
plt.show()