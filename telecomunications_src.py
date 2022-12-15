# -*- coding: utf-8 -*-
"""telecoms.ipynb


Ο κώδικας αυτός είναι ο ίδιος για τους δύο συνεργάτες , μόνο που σε μερικά σημεία λόγω της παραμετροποίησης αλλάζουν οι αρχικές μεταβλητές. Πχ στο ερώτημα 1 από fm=2 το κάνουμε fm=3 για τα αντίστοιχα διαγράμματα του συνεργάτη.

# Α ερώτημα
"""

import numpy as np 
import matplotlib.pyplot as plt


def func_creation(fm,OverSampleRate,periods):
  T = 1/fm
  fs = OverSampleRate*fm 
  Ts = 1/fs
  t = np.arange(0,periods*T, Ts)
  y = np.cos(2*np.pi*fm*t)*np.cos(2*np.pi*6*fm*t)
  return(t,y)

(t1,y1) = func_creation(2,20,4)
plt.stem(t1,y1,'r--')
plt.title('Y(t) for fs = 20fm') # plot title
plt.xlabel('Time (msec)') # x-axis label
plt.ylabel('Amplitude (V)') # y-axis label
plt.show() # display the figure

(t2,y2) = func_creation(2,100,4)
plt.stem(t2,y2,'r--')
plt.title('Y(t) for fs = 100fm') # plot title
plt.xlabel('Time (msec)') # x-axis label
plt.ylabel('Amplitude (V)') # y-axis label
plt.show() # display the figure

plt.stem(t2,y2,linefmt='blue')
plt.stem(t1,y1,linefmt='red')
plt.title('Common plot') # plot title
plt.xlabel('Time (msec)') # x-axis label
plt.ylabel('Amplitude (V)') # y-axis label
plt.show() # display the figure

(t3,y3) = func_creation(2,5,4)
plt.stem(t3,y3,'r--')
plt.title('Y(t) for fs = 5fm') # plot title
plt.xlabel('Time (msec)') # x-axis label
plt.ylabel('Amplitude (V)') # y-axis label
plt.show() # display the figure

#Δεν ζητείται αλλά είναι η γραφική αναπαράσταση του σήματος στο οποίο θα κάνουμε δειγματοληψία
t = np.linspace(0,4/2e3,100000)
Y = np.cos(2*np.pi*2e3*t)*np.cos(2*np.pi*12*2e3*t)
plt.plot(t,Y,'black')
plt.title('Y(t) arxiko') # plot title
plt.xlabel('Time (sec)') # x-axis label
plt.ylabel('Amplitude (V)') # y-axis label
plt.show()

"""# Β ερώτημα"""

import numpy as np
import matplotlib.pyplot as plt
from sympy.combinatorics.graycode import GrayCode

#declaration of parameters
amp=1
b=4 #gia fm=2 

def func_creation(fm,OverSampleRate,periods):
    T = 1/fm
    fs = OverSampleRate*fm 
    Ts = 1/fs
    t = np.arange(0,periods*T,Ts)
    y = amp*np.cos(2*np.pi*fm*t)*np.cos(2*np.pi*6*fm*t)
    return(t,y)

#input for the quantist
(t1,y1) = func_creation(8,20,4)

#creation of the 4bit quantist
levels=pow(2,b)
Step=2.0*amp/levels #step_size=2*max_amp/(2^b), b=number of bits of the quantist
quantized_signal = Step*np.round(y1/Step)
plt.xlabel("Time(msec)")
plt.ylabel(str(b)+"bits Gray Code")
plt.title(str(b)+"-bits Quantized Signal")
i=GrayCode(b)
g_code=list(i.generate_gray())
plt.yticks(np.arange(-amp, amp, step=Step),g_code)
plt.stem(t1,quantized_signal, 'r--')
plt.legend(loc='upper left')
plt.grid()
plt.show()

def quantum_error(NumberOfZeros):
    e=np.zeros(int(NumberOfZeros))
    count=0
    for i in range(0,NumberOfZeros):
        e[i]=quantized_signal[i]-y1[i]
        count+=e[i]
    count=count/NumberOfZeros
    return count
  
#N=10
count10 = quantum_error(10)
print("The quantist error for the 10 first samples is= " +str(count10))

#N=20
count20 = quantum_error(20)
print("The quantist error for the 20 first samples is=  " +str(count20))

# Experimental Signal-to-Noise-Ratio(SNR) ΚΒΑΝΤΙΣΗΣ
def SNR(NumberOfZeros):
    s_2=np.zeros(NumberOfZeros)
    n_2=np.zeros(NumberOfZeros)
    count_quantized_singal=0
    count_error=0
    e=np.zeros(int(NumberOfZeros))
    for i in range(0,NumberOfZeros):
        e[i]=quantized_signal[i]-y1[i]
    for i in range(0,NumberOfZeros):
        s_2[i]=quantized_signal[i]**2.0
        n_2[i]=e[i]**2.0
        count_quantized_singal+=s_2[i]
        count_error+=n_2[i]
    count_quantized_singal/=NumberOfZeros
    count_error/=NumberOfZeros
    snr=count_quantized_singal/count_error
    snr=pow(snr,0.5)
    return snr
#N=10
snr10 = SNR(10)
print("The experimental SNR = " +str(snr10))
#N=20
snr20 = SNR(20)
print("The experimental SNR = " +str(snr20))

"""Γ ερώτημα

# Γ Ερώτημα
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import scipy.stats as scst
import scipy.io.wavfile as wvf

#Α=0+7+4=11=2
A=2

#Ακολουθία Bit
Bitseq = []
for i in range(0, 46):
  bit = random.randint(0, 1)
  Bitseq.append(bit)

Tb = 0.5 
#fb=1/Tb
fb = 2

# Δημιουργία BPAM Διαμόρφωσης πλάτους Α
BPAMs = []
t = np.linspace(0, 46 * Tb, 4600)
for i in range(0, 46):
    if Bitseq[i] == 1:
        for j in range(0, 100):
            BPAMs.append(A)
    else:
        for j in range(0, 100):
            BPAMs.append(-A)
BPAMs[0] = 0
BPAMs[4599] = 0
plt.step(t, BPAMs, where='pre',color='orange')
plt.title('BPAM transmission of Bit sequence')
plt.ylabel('Amplitude[V]')
plt.xlabel('Time(sec)')
plt.show()

Eb = A * A * Tb
Bpam = []
for i in range(0, 36):
    if Bitseq[i] == 0:
        Bpam.append(-A)
    else:
        Bpam.append(A)
plt.scatter(np.sqrt(Eb) / A * np.real(Bpam), np.sqrt(Eb) / A * np.imag(Bpam),color='orange')
plt.title('Constellation of BPAM modulated Bit sequence')
plt.xlabel('Real Part [V]')
plt.ylabel('Imaginary Part [V]')
plt.show()

s1 = 5
db1 = pow(10, -s1 / 10) * Eb / 2
s2 = 15
db2 = pow(10, -s2 / 10) * Eb / 2
t = np.linspace(0, 46 * Tb, 4600)

#noise1 5dΒ
noise1 = np.zeros(len(BPAMs), dtype=np.complex64)
noise1 = BPAMs + np.random.normal(0, 1, len(noise1)) * np.sqrt(db1)
plt.step(t, noise1,color='orange')
plt.xlabel('Time[sec]')
plt.ylabel('Amplitude[V]')
plt.title('B-PAM transmission of Bit sequence with noise 5db')
plt.show()

#noise2 15dΒ
noise2 = np.zeros(len(BPAMs), dtype=np.complex64)
noise2 = BPAMs + np.random.normal(0, 1, len(noise2)) * np.sqrt(db2)
plt.step(t, noise2,color='orange')
plt.xlabel('Time[sec]')
plt.ylabel('Amplitude[V]')
plt.title('B-PAM transmission of Bit sequence with noise 15db')
plt.show()

# Προσθήκη Θορύβων 5 και 15 dB στο σήμα
BPAMs[0] = BPAMs[1]
BPAMs[4599] = BPAMs[4598]
noise1 = BPAMs + (1j * np.random.normal(0, 1, len(noise1)) + np.random.normal(0, 1, len(noise1))) * np.sqrt(db1)
noise2 = BPAMs + (1j * np.random.normal(0, 1, len(noise2)) + np.random.normal(0, 1, len(noise2))) * np.sqrt(db2)
plt.scatter(np.real(noise1) * np.sqrt(Eb) / A, np.imag(noise1) * np.sqrt(Eb) / A,color='orange')
plt.xlabel('Real Part [V]')
plt.ylabel('Imaginary Part [V]')
plt.title('Constellation of BPAM modulated Bit sequence')
plt.show()
plt.scatter(np.real(noise2) * np.sqrt(Eb) / A, np.imag(noise2) * np.sqrt(Eb) / A,color='orange')
plt.xlabel('Real Part [V]')
plt.ylabel('Imaginary Part [V]')
plt.title('Constellation of BPAM modulated Bit sequence')
plt.show()

bpams = []
Bitseq = []
 # Δημιουργία σήματος 10000 bit
bits = 10000
for i in range(0, bits):
    bit = random.randint(0, 1)
    Bitseq.append(bit)
for i in range(0, bits):
    if Bitseq[i] == 1:
        bpams.append(math.sqrt(Eb))
    else:
        bpams.append(-math.sqrt(Eb))
Noise = np.zeros(len(bpams), dtype=np.complex64)

# υπολογισμός πειραματικού ποσοστού σφάλματος για db 0-15 με προσθήκη θορύβου
BER = []
for i in range(0, 16):
    ERRORS = 0
    db = pow(10, -i / 10)
    N0 = db * Eb
    s = math.sqrt(N0 / 2)
    Noise = bpams + np.random.normal(0, s, len(bpams)) * 1j + np.random.normal(0, s, len(bpams))
    for j in range(0, bits):
        if np.real(Noise[j]) > 0:
            if bpams[j] <= 0:
                ERRORS += 1
        else:
            if bpams[j] > 0:
                ERRORS += 1
    BER.append(ERRORS / bits)

# Υπολογισμός Θεωρητικού Ποσοστού Σφάλματος για dB 0-15 με προσθήκη θορύβου
BERtheo = []
for i in range(0, 1600):
    numb = math.pow(10, (i / 100) / 10)
    BERtheo.append(scst.norm.sf(math.sqrt(2 * numb)))

x_th = np.linspace(0, 15, 1600)
x_exp = np.linspace(0, 15, 16)
plt.plot(x_th, BERtheo, color='orange', label='Theoritical BER')
plt.scatter(x_exp, BER, color='black', marker='x', label='Experimental BER')
plt.xlabel('Eb/N0 in dB')
plt.ylabel('BER (Theoritical and Experimental)')
plt.title('Bit Error Rate (BER) as a function of Eb/N0')
plt.legend()
plt.show()

"""# Δ ερώτημα"""

Α=2 #εδώ αλλάζει για τον συνεργάτη
Tb = 0.5  # sec
number = 46
En = (A ** 2) * Tb
energy = math.sqrt(En)
# Τυχαία παλμοσειρά από bits
bits = Bitseq  
plot_bits = []  
for i in range(0, 46):
    rand = bits[i]
    for j in range(0, 100):
        plot_bits.append(rand)
x = np.linspace(0, 46 * 0.2, 4600)
plt.title('Bitseq 46 bits')
plt.plot(x, plot_bits, color='orange')
plt.xlabel('time (sec)')
plt.ylabel('Bit')
plt.show()

# QPSK Διαμόρφωση - Κυματομορφή
def functionqpsk(bits, Tb, number, A):
    qpsq = np.zeros(40 * number)
    t = np.linspace(0, number * Tb, 40 * number)
    fc = 2 / Tb
    for i in range(0, number, 2):
        if bits[i] == 0 and bits[i + 1] == 0:
            for j in range(0, 80):
                qpsq[j + i * 40] = - A * np.cos(2 * np.pi * fc * t[j + i * 40]) - A * np.sin(
                    2 * np.pi * fc * t[j + i * 40])
        elif bits[i] == 0 and bits[i + 1] == 1:
            for j in range(0, 80):
                qpsq[j + i * 40] = - A * np.cos(2 * np.pi * fc * t[j + i * 40]) + A * np.sin(
                    2 * np.pi * fc * t[j + i * 40])
        elif bits[i] == 1 and bits[i + 1] == 1:
            for j in range(0, 80):
                qpsq[j + i * 40] = A * np.cos(2 * np.pi * fc * t[j + i * 40]) + A * np.sin(
                    2 * np.pi * fc * t[j + i * 40])
        else:
            for j in range(0, 80):
                qpsq[j + i * 40] = A * np.cos(2 * np.pi * fc * t[j + i * 40]) - A * np.sin(
                    2 * np.pi * fc * t[j + i * 40])
    return qpsq

qpsq = functionqpsk(bits, Tb, number, A)  # QPSK Modulation
t = np.linspace(0, 46 * Tb, 40 * 46)
plt.title('Κυματομορφή QPSK of Bitseq')
plt.plot(t, qpsq, color='orange')
plt.xlabel('time (sec)')
plt.ylabel('Amplitude (Volts)')
plt.show()

# Σύμβολα, στο επίπεδο των μιγαδικών
sym = [energy + energy * 1j, -energy + energy * 1j, -energy - energy * 1j,
            energy - energy * 1j]
for i in range(0, 4):
    sym[i] = sym[i] / math.sqrt(2)
X = [x.real for x in sym]
Y = [x.imag for x in sym]
t = np.linspace(0, 2 * np.pi, 101)
k = np.linspace(-energy / math.sqrt(2), energy / math.sqrt(2), 100)
z = k

array = ['s1(1 1)', 's2(0 1)', 's3(0 0)', 's4(1 0)']
for i, type in enumerate(array):
    plt.scatter(X[i], Y[i], color='black')
    plt.text(X[i] + 0.03, Y[i] + 0.03, type, fontsize=9)
plt.plot(energy * np.cos(t), energy * np.sin(t), color='orange')
plt.plot(k, z, color='orange')
plt.plot(k, -z, color='orange')
plt.title('Constellation')
plt.grid(True)
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')

plt.show()

# Παραγωγή θορύβου AWGN

rat1 = 5
rat2 = 15
def functionawgn(ratiodB, number, En, c=True):
    ratio = math.pow(10, -ratiodB / 10)
    if c:
        NO = (En / 2) * ratio
    else:
        NO = En * ratio
    sigma = math.sqrt(NO / 2)
    Awgn = np.random.normal(0, sigma, int(number / 2)) + 1j * np.random.normal(0, sigma, int(number / 2))
    return Awgn
awgn1 = functionawgn(rat1, number, En)  # Εb/N0 = 5dB
awgn2 = functionawgn(rat2, number, En)  # Eb/N0 = 15 dB

# QPSK σήμα, στο επίπεδο των μιγαδικών
def QPSK_Const(bits_Qpsk, samples, energy):
    QPSKConstell = np.zeros(int(samples / 2), dtype=np.complex64)
    counter = 0
    for i in range(0, samples, 2):
        if bits_Qpsk[i] == 0 and bits_Qpsk[i + 1] == 0:
            QPSKConstell[counter] += -energy - energy * 1j
        elif bits_Qpsk[i] == 0 and bits_Qpsk[i + 1] == 1:
            QPSKConstell[counter] += -energy + energy * 1j
        elif bits_Qpsk[i] == 1 and bits_Qpsk[i + 1] == 1:
            QPSKConstell[counter] += energy + energy * 1j
        else:
            QPSKConstell[counter] += energy - energy * 1j
        counter += 1
    return QPSKConstell / math.sqrt(2)
QPSKConstella = QPSK_Const(bits, number, energy)
S1 = awgn1 + QPSKConstella
S2 = awgn2 + QPSKConstella
# Constellation for Eb/N0 = 5 dB

realignal = [x.real for x in S1]
imagesignal = [x.imag for x in S1]
plt.scatter(realignal, imagesignal, color='orange', marker="o")
plt.grid(True)
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('Constellation for Eb/N0 = 5 dB')
plt.show()
# Constellation for Eb/N0 = 15 dB
realignal = [x.real for x in S2]
imagesignal = [x.imag for x in S2]
plt.scatter(realignal, imagesignal, color='orange', marker="o")
plt.grid(True)
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('Constellation for Eb/N0 = 15 dB')
plt.show()

#gia 100 deigmata
#na epanalabo gia 1000 kai 10000
samples = 100
bits_array = []
for i in range(0, samples):
    random_bit = random.randint(0, 1)
    bits_array.append(random_bit)
QPSK_Constellation_c = QPSK_Const(bits_array, samples, energy)

# Συνάρτηση για υπολογισμό πειραματικoύ BER
def functionEXPBER(QPSK_Const, Noise, energy):
    s = np.zeros(4, dtype=np.complex64)
    s[0] = (energy + energy * 1j) / math.sqrt(2) 
    s[1] = (-energy + energy * 1j) / math.sqrt(2)  
    s[2] = (-energy - energy * 1j) / math.sqrt(2)  
    s[3] = (energy - energy * 1j) / math.sqrt(2)  
    Rec = QPSK_Const + Noise
    BER = 0
    for i in range(0, len(Rec)):
        if (Rec[i].real > 0 and Rec[i].imag > 0):
            ind = 0
        elif (Rec[i].real < 0 and Rec[i].imag > 0):
            ind = 1
        elif (Rec[i].real < 0 and Rec[i].imag < 0):
            ind = 2
        else:
            ind = 3
        # Έλεγχος για διαφορετικό τεταρτημόριο 
        if (QPSK_Const[i].real * s[ind].real < 0 or QPSK_Const[i].imag * s[
            ind].imag < 0):
            BER += 1
    BER = BER / (2 * len(Rec))
    return BER

EXPBER = np.zeros(16)
THEORYBER = np.zeros(1600)
#Πειραματικό BER
for i in range(0, 16):
    awgn_c = functionawgn(i, samples, En)
    EXPBER[i] = functionEXPBER(QPSK_Constellation_c, awgn_c, energy)
    ratio = math.pow(10, i / 10)
#Θεωριτικό BER
for i in range(0, 1600):
    ratio = math.pow(10, (i / 100) / 10)
    THEORYBER[i] = scst.norm.sf(math.sqrt(ratio) * math.sqrt(2))

x_theo = np.linspace(0, 15, 1600)
x_experiment = np.linspace(0, 15, 16)
plt.plot(x_theo, THEORYBER, color='orange', label='Theoritical BER')
plt.scatter(x_experiment, EXPBER, color='black', marker='x', label='Experimental BER')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER (Theoritical and Experimental)')
plt.legend()
plt.show()

#gia 1000 deigmata

samples = 1000
bits_array = []
for i in range(0, samples):
    random_bit = random.randint(0, 1)
    bits_array.append(random_bit)
QPSK_Constellation_c = QPSK_Const(bits_array, samples, energy)

# Συνάρτηση για υπολογισμό πειραματικoύ BER
def functionEXPBER(QPSK_Const, Noise, energy):
    s = np.zeros(4, dtype=np.complex64)
    s[0] = (energy + energy * 1j) / math.sqrt(2) 
    s[1] = (-energy + energy * 1j) / math.sqrt(2)  
    s[2] = (-energy - energy * 1j) / math.sqrt(2)  
    s[3] = (energy - energy * 1j) / math.sqrt(2)  
    Rec = QPSK_Const + Noise
    BER = 0
    for i in range(0, len(Rec)):
        if (Rec[i].real > 0 and Rec[i].imag > 0):
            ind = 0
        elif (Rec[i].real < 0 and Rec[i].imag > 0):
            ind = 1
        elif (Rec[i].real < 0 and Rec[i].imag < 0):
            ind = 2
        else:
            ind = 3
        # Έλεγχος για διαφορετικό τεταρτημόριο 
        if (QPSK_Const[i].real * s[ind].real < 0 or QPSK_Const[i].imag * s[
            ind].imag < 0):
            BER += 1
    BER = BER / (2 * len(Rec))
    return BER

EXPBER = np.zeros(16)
THEORYBER = np.zeros(1600)
#Πειραματικό BER
for i in range(0, 16):
    awgn_c = functionawgn(i, samples, En)
    EXPBER[i] = functionEXPBER(QPSK_Constellation_c, awgn_c, energy)
    ratio = math.pow(10, i / 10)
#Θεωριτικό BER
for i in range(0, 1600):
    ratio = math.pow(10, (i / 100) / 10)
    THEORYBER[i] = scst.norm.sf(math.sqrt(ratio) * math.sqrt(2))

x_theo = np.linspace(0, 15, 1600)
x_experiment = np.linspace(0, 15, 16)
plt.plot(x_theo, THEORYBER, color='orange', label='Theoritical BER')
plt.scatter(x_experiment, EXPBER, color='black', marker='x', label='Experimental BER')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER (Theoritical and Experimental)')
plt.legend()
plt.show()

#gia 10000 deigmata
samples = 10000
bits_array = []
for i in range(0, samples):
    random_bit = random.randint(0, 1)
    bits_array.append(random_bit)
QPSK_Constellation_c = QPSK_Const(bits_array, samples, energy)

# Συνάρτηση για υπολογισμό πειραματικoύ BER
def functionEXPBER(QPSK_Const, Noise, energy):
    s = np.zeros(4, dtype=np.complex64)
    s[0] = (energy + energy * 1j) / math.sqrt(2) 
    s[1] = (-energy + energy * 1j) / math.sqrt(2)  
    s[2] = (-energy - energy * 1j) / math.sqrt(2)  
    s[3] = (energy - energy * 1j) / math.sqrt(2)  
    Rec = QPSK_Const + Noise
    BER = 0
    for i in range(0, len(Rec)):
        if (Rec[i].real > 0 and Rec[i].imag > 0):
            ind = 0
        elif (Rec[i].real < 0 and Rec[i].imag > 0):
            ind = 1
        elif (Rec[i].real < 0 and Rec[i].imag < 0):
            ind = 2
        else:
            ind = 3
        # Έλεγχος για διαφορετικό τεταρτημόριο 
        if (QPSK_Const[i].real * s[ind].real < 0 or QPSK_Const[i].imag * s[
            ind].imag < 0):
            BER += 1
    BER = BER / (2 * len(Rec))
    return BER

EXPBER = np.zeros(16)
THEORYBER = np.zeros(1600)
#Πειραματικό BER
for i in range(0, 16):
    awgn_c = functionawgn(i, samples, En)
    EXPBER[i] = functionEXPBER(QPSK_Constellation_c, awgn_c, energy)
    ratio = math.pow(10, i / 10)
#Θεωριτικό BER
for i in range(0, 1600):
    ratio = math.pow(10, (i / 100) / 10)
    THEORYBER[i] = scst.norm.sf(math.sqrt(ratio) * math.sqrt(2))

x_theo = np.linspace(0, 15, 1600)
x_experiment = np.linspace(0, 15, 16)
plt.plot(x_theo, THEORYBER, color='orange', label='Theoritical BER')
plt.scatter(x_experiment, EXPBER, color='black', marker='x', label='Experimental BER')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER (Theoritical and Experimental)')
plt.legend()
plt.show()



"""επεξεργασία κειμένου"""

#AM 03117074 opote 0+7+4=11 opote xrhsimopoio to shannon_odd.txt
#AM synergath =03117048 = 0+4+8=12 = artios 
#anebazo ta arxeia sto collab

AM=11  
if (AM % 2 == 0):
    f = open("shannon_even.txt", 'r')
else:
    f = open("shannon_odd.txt", 'r')
string = f.read()
f.close()
binary_array = ''.join([bin(ord(x))[2:].zfill(8) for x in string]) # Μετατροπή από ASCII σε Binary
#edo gia kapoio logo an to trekso 2 fores xtypaei alla sth proth leitoygei sosta

# Μετατροπή δυαδικής συμβολοσειράς σε πίνακα, όπου κάθε στοιχείο του πίνακα αντιστοιχεί σε ένα bit
arrayint = np.zeros(len(binary_array))
for i in range(0, len(binary_array)):
    arrayint[i] = int(binary_array[i])

# Πίνακας για αναπαράσταση της δυαδικής συμβολοσειράς
bin_plot = np.zeros(100 * len(arrayint))
for i in range(0, len(arrayint)):
    for j in range(0, 100):
        bin_plot[i * j + j] = arrayint[i]

####### Plot το δυαδικό σήμα. Θα πραγματοποιήσουμε plot μονάχα ένα μέρος του σήματος, καθώς τα δείγματα είναι πάρα
####### πολλά για παρατήρηση του γραφήματος. Επιλέξαμε διάρκεια bit ίση με Tb.

t = np.linspace(0, Tb * 100, 100)
plt.title('Δυαδική Κωδικοποίηση του Αρχείου Κειμένου')
plt.step(t, bin_plot[0:100],color="orange")
plt.xlabel('Χρόνος(sec)')
plt.ylabel('Bits')

plt.show()

Ab = 1
Es_B = (Ab ** 2) * Tb
Root_Energy_B = math.sqrt(Es_B)
Qpsk_Bin = functionqpsk(arrayint, Tb, len(arrayint), Ab)  # QPSK Modulation
t = np.linspace(0, 100 * Tb, 4000)# Plot μόνο τα πρώτα 100 bits για καλύτερη εποπτεία.
plt.plot(t, Qpsk_Bin[0:4000],color="orange")  
plt.xlabel('Χρόνος(sec)')
plt.ylabel('Amplitude(Volts)')
plt.title('Κυματομορφή QPSK της Δυαδικής Ακολουθίας του Αρχείου Κειμένου')
plt.show()
QPSK_ConstB = QPSK_Const(arrayint, len(binary_array),Root_Energy_B)  # QPSK στον χώρο των μιγαδικών
# Δημιουργία Θορύβου
awgn1B = functionawgn(rat1, len(binary_array), Es_B, False)
awgn2B = functionawgn(rat2, len(binary_array), Es_B, False)
# Διαμορφωμένα σήματα
Signal1B = awgn1B + QPSK_ConstB
Signal2B = awgn2B + QPSK_ConstB

# Διάγραμμα Αστερισμού για Es/N0 = 5 dB
REAL_SignalB = [x.real for x in Signal1B]
IMAG_SignalB = [x.imag for x in Signal1B]
t = np.linspace(0, len(arrayint) * Tb, 100 * len(arrayint))
plt.scatter(REAL_SignalB, IMAG_SignalB, color='orange', marker="v")
plt.grid(True)
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('Διάγραμμα Αστερισμού για Es/N0 = 5 dB')
plt.show()
# Διάγραμμα Αστερισμού για Es/N0 = 15 dB
REAL_SignalB = [x.real for x in Signal2B]
IMAG_SignalB = [x.imag for x in Signal2B]
plt.scatter(REAL_SignalB, IMAG_SignalB, color='orange', marker="v")
plt.grid(True)
plt.xlabel('Πραγματικο Μέρος')
plt.ylabel('Φανταστικό Μέρος')
plt.title('Διάγραμμα Αστερισμού για Es/N0 = 15 dB')
plt.show()

def functiondemodulation(Signal, Root_Energy):
    s_array = np.zeros(4, dtype=np.complex64)
    s_array[0] = (Root_Energy + Root_Energy * 1j) / math.sqrt(2)  # Σύμβολο 11
    s_array[1] = (-Root_Energy + Root_Energy * 1j) / math.sqrt(2)  # Σύμβολο 01
    s_array[2] = (-Root_Energy - Root_Energy * 1j) / math.sqrt(2)  # Σύμβολο 00
    s_array[3] = (Root_Energy - Root_Energy * 1j) / math.sqrt(2)  # Σύμβολο 10
    DemoS = []

# Επιλογή μεταδιδόμενου συμβόλου, με κριτήριο την ελάχιστη απόσταση

    for i in range(0, len(Signal)):
        if (Signal[i].real > 0 and Signal[i].imag > 0):
            minimum_index = 0
        elif (Signal[i].real < 0 and Signal[i].imag > 0):
            minimum_index = 1
        elif (Signal[i].real < 0 and Signal[i].imag < 0):
            minimum_index = 2
        else:
            minimum_index = 3
        if (minimum_index == 0):
            DemoS.append(1)
            DemoS.append(1)
        elif (minimum_index == 1):
            DemoS.append(0)
            DemoS.append(1)
        elif (minimum_index == 2):
            DemoS.append(0)
            DemoS.append(0)
        else:
            DemoS.append(1)
            DemoS.append(0)
    return DemoS

DemoS5dB = functiondemodulation(Signal1B, Root_Energy_B)  # Αποδιαμόρφωση για Εs/N0 = 5 dB
DemoS15dB = functiondemodulation(Signal2B, Root_Energy_B)  # Αποδιαμόρφωση για Es/N0 = 15 dB

Exp_BER_5dB = functionEXPBER(QPSK_ConstB, awgn1B, Root_Energy_B)  # Πειραματικό BER για Es/N0 = 5 dB
Exp_BER_15dB = functionEXPBER(QPSK_ConstB, awgn2B, Root_Energy_B)  # Πειραματικό BER για Es/N0 = 15 dB

Theor_BER_5dB = scst.norm.sf(math.sqrt(math.pow(10, rat1 / 10)))  # Θεωρητικό BER για Es/N0 = 5 dB
Theor_BER_15dB = scst.norm.sf(math.sqrt(math.pow(10, rat2 / 10)))  # Θεωρητικό BER για Es/N0 = 15 dB

print("For Es/N0 = 5dB :", "\n", "Experimental BER = ", Exp_BER_5dB, "\n", "Theoritical BER = ", Theor_BER_5dB)
print("For Es/N0 = 15dB :", "\n", "Experimental BER = ", Exp_BER_15dB, "\n", "Theoritical BER = ", Theor_BER_15dB)

# Binary to Ascii
# Μετατροπή των στοιχείων του πίνακα αποδιαμόρφωσης από string σε int
DemoString1 = []
for i in range(0, len(DemoS5dB)):
    DemoString1.append(str(DemoS5dB[i]))
DemoS1 = ''.join(DemoString1)

DemoString2 = []
for i in range(0, len(DemoS15dB)):
    DemoString2.append(str(DemoS15dB[i]))
DemoS2 = ''.join(DemoString2)

# Συνάρτηση μετατροπής από Binary σε Decimal
def functionbtod(binary):
    Decimal = np.zeros(int(len(binary) / 8))
    counter = 0
    for i in range(0, len(binary), 8):
        for j in range(0, 8):
            if (binary[i + j] == "1"):
                Decimal[counter] += np.power(2, 7 - j)
        counter += 1
    return Decimal
DemoD1 = functionbtod(DemoS1)
DemoD2 = functionbtod(DemoS2)
text1 = []
text2 = []
# Μετατροπή Decimal σε ASCII
for i in range(0, len(DemoD1)):
    c1 = int(DemoD1[i])
    c2 = int(DemoD2[i])
    text1.append(chr(c1))
    text2.append(chr(c2))

text1 = ''.join(text1)  # Τελικό Κείμενο για Εs/N0 = 5 dB
text2 = ''.join(text2)  # Τελικό Κείμενο για Εs/N0 = 15 dB
print(text1)
print(text2)

# Δημιουργία αρχείων .txt
if AM % 2 == 0:
    f1 = open("03117048_Text_1.txt", "w+", encoding="utf-8")
    f2 = open("03117048_Text_2.txt", "w+", encoding="utf-8")
else:
    f1 = open("03117074_Text_1.txt", "w+", encoding="utf-8")
    f2 = open("03117074_Text_2.txt", "w+", encoding="utf-8")

f1.write(text1)
f2.write(text2)
f1.close()
f2.close()



"""# E ερώτημα"""

A=11 #synergaths 12
if A % 2 == 0:
    fs, sound = wvf.read('soundfile2_lab2.wav')
else:
    fs, sound = wvf.read('soundfile1_lab2.wav')

# Plot 
t = np.linspace(0, 7 - A % 2, len(sound))
plt.title('Ηχητικό Σήμα')
plt.plot(t, sound,color='orange')
plt.xlabel('Time(sec)')
plt.ylabel('Amplitude(Volts)')
plt.show()

# 8-bit ΚΒΑΝΤΙΣΤΗΣ
def function8bit(data):
    nolevels = 2 ** 8
    maximum = max(np.amax(data), -np.amin(data))
    distance = (2 * maximum) / (nolevels - 1)
    levels = np.zeros(nolevels)
    for i in range(0, nolevels):
        levels[i] = - maximum + i * distance
    dataquant = np.zeros(len(data))
    dataquantindex = np.zeros(len(data))
    count = 0
    for i in range(0, len(data)):
        count += 1
        index = 128
        change = 128
        c = True
# Εύρεση σωστής στάθμης με Διαδική Αναζήτηση
        while c:
            data_distance = math.fabs(data[i] - levels[index - 1])
            if data_distance < distance:
                dataquant[i] = levels[index - 1]
                dataquantindex[i] = index - 1
                c = False
            elif data[i] - levels[index - 1] > 0:
                change = change // 2
                index = index + change
            else:
                change = change // 2
                index = index - change
    return dataquant, dataquantindex, maximum, levels

Quant_S, Quant_S_Index, maximum, levels = function8bit(sound)  # ΚΒΑΝΤΙΣΗ
# Quant_S-> Κβαντισμένο σήμα με κβαντισμένες τιμές
# Quant_S_Index -> Κβαντισμένο σήμε με τιμές τις αντίστοιχες στάθμες
x = np.linspace(0, 7 - A % 2, len(Quant_S))
plt.step(x, Quant_S,color="orange")
plt.title('Κβαντισμένο σήμα Ήχου')
plt.xlabel('Time(sec)')
plt.ylabel('Amplitude(Volts)')
plt.show()

bin = []
for i in range(0, len(Quant_S_Index)):
    c1 = int(Quant_S_Index[i])
    c2 = format(c1, "b").zfill(8)
    bin.append(c2)

binary = ''.join(bin)
# Μετατροπή των δυαδικών συμβόλων σε int
binary_int = np.zeros(len(binary))
for i in range(0, len(binary)):
    binary_int[i] = int(binary[i])

fs = 44100
Ts = 1 / fs
As = 1
Ess = (As ** 2) * Ts
Root_Energy_S = math.sqrt(Ess)
QPSK_Constellation_Sound = QPSK_Const(binary_int, len(binary_int),
                                      Root_Energy_S)  # QPSK στο επίπεδο των μιγαδικών

# Παραγωγή ζητούμενων AWGN θορύβων
awgn1S = functionawgn(4, len(binary_int), Ess, False)
awgn2S = functionawgn(14, len(binary_int), Ess, False)

# Άθροισμα QPSK και AWGN - Τελικό σήμα
Signal1S = QPSK_Constellation_Sound + awgn1S
Signal2S = QPSK_Constellation_Sound + awgn2S

# Διάγραμμα Αστερισμού για Es/N0 = 4 dB
REAL_SignalS = [x.real for x in Signal1S]
IMAG_SignalS = [x.imag for x in Signal1S]
plt.scatter(REAL_SignalS, IMAG_SignalS, color='orange', marker="v")
plt.grid(True)
plt.title('Διάγραμμα Αστερισμού για Es/N0 = 4 dB')
plt.xlabel('Πραγματικό Μέρος')
plt.ylabel('Φανταστικό Μέρος')
plt.show()
# Διάγραμμα Αστερισμού για Es/N0 = 14 dB
REAL_SignalS = [x.real for x in Signal2S]
IMAG_SignalS = [x.imag for x in Signal2S]
plt.scatter(REAL_SignalS, IMAG_SignalS, color='orange', marker="v")
plt.grid(True)
plt.title('Διάγραμμα Αστερισμού για Es/N0 = 14 dB')
plt.xlabel('Πραγματικό Μέρος')
plt.ylabel('Φανταστικό Μέρος')
plt.show()

# Αποδιαμόρφωση
DemoS4dB = functiondemodulation(Signal1S, Root_Energy_S)
DemoS14dB = functiondemodulation(Signal2S, Root_Energy_S)

# Πειραματικό BER
ExpBER4dB = functionEXPBER(QPSK_Constellation_Sound, awgn1S, Root_Energy_S)
ExpBER14dB = functionEXPBER(QPSK_Constellation_Sound, awgn2S, Root_Energy_S)
# Θεωρητικό BER
TheorBER4dB = scst.norm.sf(math.sqrt(math.pow(10, 4 / 10)))
TheorBER14dB = scst.norm.sf(math.sqrt(math.pow(10, 14 / 10)))
print("For Es/N0 = 4dB :", "\n", "Experimental BER = ", ExpBER4dB, "\n", "Theoritical BER = ", TheorBER4dB)
print("For Es/N0 = 14dB :", "\n", "Experimental BER = ", ExpBER14dB, "\n", "Theoritical BER = ", TheorBER14dB)
print('\n')

# Μετατροπή των στοιχείων του πίνακα αποδιαμόρφωσης από string σε int
DemoSound1 = []
for i in range(0, len(DemoS4dB)):
    DemoSound1.append(str(DemoS4dB[i]))
DemoSound1 = ''.join(DemoSound1)

DemoSound2 = []
for i in range(0, len(DemoS14dB)):
    DemoSound2.append(str(DemoS14dB[i]))
DemoSound2 = ''.join(DemoSound2)

# Από Δυαδικό σε Δεκαδικό στο Αποδιαμορφωμένο Σήμα
DemoDec1S = functionbtod(DemoSound1)
DemoDec2S = functionbtod(DemoSound2)

# Δημιουργία Αρχείων Ήχου
Final1 = np.zeros(len(DemoDec1S), dtype=np.uint8)
Final2 = np.zeros(len(DemoDec2S), dtype=np.uint8)

for i in range(0, len(DemoDec1S)):
    Final1[i] = int(DemoDec1S[i])
    Final2[i] = int(DemoDec2S[i])

if A % 2 == 0:
    wvf.write('03117048_Sound_1.wav', fs, Final1)
    wvf.write('03117048_Sound_2.wav', fs, Final2)
else:
    wvf.write('03117074_Sound_1.wav', fs, Final1)
    wvf.write('03117074_Sound_2.wav', fs, Final2)