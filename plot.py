import qam
import numpy as np
import matplotlib.pyplot as plt

QAM4   = np.load('QAM4.npy', allow_pickle='TRUE').item()
QAM16  = np.load('QAM16.npy', allow_pickle='TRUE').item()
QAM32  = np.load('QAM64.npy', allow_pickle='TRUE').item()
QAM256 = np.load('QAM256.npy', allow_pickle='TRUE').item()

qam.add_ber_plot(QAM4['EbN0_db'], QAM4['OSNR_dB'], QAM4['BER'])
qam.add_ber_plot(QAM16['EbN0_db'], QAM16['OSNR_dB'], QAM16['BER'])
qam.add_ber_plot(QAM32['EbN0_db'], QAM32['OSNR_dB'], QAM32['BER'])
qam.add_ber_plot(QAM256['EbN0_db'], QAM256['OSNR_dB'], QAM256['BER'])
plt.legend(['QAM4', 'QAM16', 'QAM32', 'QAM256'])
plt.show()


