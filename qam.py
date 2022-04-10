import numpy as np
from math import log
from scipy.special import erf
import matplotlib.pyplot as plt


def modulate(x, constel):
    return constel[x]


def demodulate(x, constel):
    samples = len(x)
    result = np.zeros(x.shape)
    for i in range(samples):
        # Медленно, но работает для любой модуляции
        j = np.argmin(np.abs(constel - x[i]))
        result[i] = j
    
    return result.astype(int)


def norm(constel):
    dot = np.dot(constel, np.conj(constel))
    return constel / np.sqrt(dot / len(constel))


def constellation(M):
    if np.fix(log(M, 4)) != log(M,4):
        raise ValueError("M must be power of 4!")

    nbits = int(np.log2(M))
    x = np.arange(M)

    nbitsBy2 = nbits >> 1
    symbolI = x >> nbitsBy2
    symbolQ = x & ((M-1) >> nbitsBy2)

    i = 1
    while i < nbitsBy2:
        tmpI = symbolI
        tmpI = tmpI >> i
        symbolI = symbolI ^ tmpI

        tmpQ = symbolQ
        tmpQ = tmpQ >> i
        symbolQ = symbolQ ^ tmpQ
        i = i + i

    gray = (symbolI << nbitsBy2) + symbolQ

    x = x[gray]
    c = int(np.sqrt(M))
    I = -2 * np.mod(x, c) + c - 1
    Q = 2 * np.floor(x / c) - c + 1
    IQ = I + 1j*Q
    IQ = -np.transpose(np.reshape(IQ, (c, c)))
    return norm(IQ.flatten())


def qfunc(x):
    return 0.5 - 0.5 * erf(x / np.sqrt(2))


# Only for square QAM
# Source: https://www.mathworks.com/help/comm/ug/analytical-expressions-used-in-berawgn-function-and-bit-error-rate-analysis-app.html 
def theory_ber(EbN0, M):
    if np.fix(log(M, 4)) != log(M,4):
        raise ValueError("M must be power of 4!")
    
    if M == 4:
        ber = qfunc(np.sqrt(2*EbN0))
    elif M == 16:
        ber = 3/4*qfunc(np.sqrt(4/5*EbN0)) 
        + 1/2*qfunc(3*np.sqrt(4/5*EbN0)) 
        - 1/4*qfunc(5*np.sqrt(4/5*EbN0))
    elif M == 64:
        ber = 7/12*qfunc(np.sqrt(2/7*EbN0)) 
        + 1/2*qfunc(3*np.sqrt(2/7*EbN0)) 
        - 1/12*qfunc(5*np.sqrt(2/7*EbN0)) 
        + 1/12*qfunc(9*np.sqrt(2/7*EbN0)) 
        - 1/12*qfunc(13*np.sqrt(2/7*EbN0))
    else:
        k = np.log2(M)
        c = np.sqrt(M)
        ber = np.zeros(EbN0.shape)
        for i in range(1, round(np.log2(c)) + 1):
            berk = np.zeros(EbN0.shape)
            for j in range(0,round((1-2**(-i))*c)):
                berk = berk + (-1)**(np.floor(j*2**(i-1)/c)) * (2**(i-1) 
                - np.floor(j*2**(i-1)/c+1/2)) * qfunc((2*j+1) * np.sqrt(6*k*EbN0/(2*(M-1))))
            berk = berk * 2 / c
            ber = ber + berk

        ber = ber / np.log2(c)

    return ber


def plot_constel(iq):
    count = len(iq)
    bits = np.log2(count)
    spec = '#0{}b'.format((bits + 2).astype(int))

    plt.figure(dpi = 80)
    for n in range(count):
        factor = np.log2(np.sqrt(count))
        d = 0.04 / factor
        i = np.real(iq[n])
        q = np.imag(iq[n])
        label = format(n, spec)
        plt.text(i + d, q + d, label[2::], fontsize = 30 / factor)

    scale = np.max(np.abs(iq))
    plt.scatter(np.real(iq), np.imag(iq), s = 120 / factor)
    plt.title('QAM{} constellation'.format(count))
    plt.xlim([-scale, scale])
    plt.ylim([-scale, scale])
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.grid()
    plt.show()


def add_ber_plot(EbN0, OSNR, BER, spec='--'):
    plt.subplot(1,2,1)
    plt.semilogy(EbN0, BER, spec, markersize=8, linewidth=3)
    plt.xlabel('Eb/N0, dB')
    plt.ylabel('BER')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')

    plt.subplot(1,2,2)
    plt.semilogy(OSNR, BER, spec, markersize=8, linewidth=3)
    plt.xlabel('OSNR, dB')
    plt.ylabel('BER')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')


def hamming(str1,str2):
    result=0
    for _,(i,j) in enumerate(zip(str1, str2)):
        if i!=j:
            result+=1
    return result


def ber(input, output, M):
    bits_per_symbol = np.log2(M)
    symbols = len(input)
    diff = []
    for i in range(symbols):
        in_bits = format(input[i], '016b')
        out_bits = format(output[i], '016b')
        diff.append(hamming(in_bits, out_bits))

    return np.sum(diff) / (symbols * bits_per_symbol)


def save_ber(OSNR_dB, EbN0_db, BER, M):
    data = {
    'M' : M,
    'OSNR_dB' : OSNR_dB,
    'EbN0_db' : EbN0_db,
    'BER' : BER
    }
    np.save('QAM{}.npy'.format(M), data)