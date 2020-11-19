


def bit2sym(abit):
    """Bit to Symbol Conversion"""
    from numpy import array
    return array([b[0]*8+b[1]*4+b[2]*2+b[3] for b in abit.reshape(len(abit)//4,4)])

def sym2bit(asym):
    """Symbol to Bit Conversion"""
    from numpy import array,floor,r_
    abit=array([],dtype=int)
    for sym in asym:
        b0 = floor(sym/8)
        b1 = floor(sym/4)-2*b0
        b2 = floor(sym/2)-2*b1-4*b0
        b3 = sym%2
        abit = r_[abit,array([b0,b1,b2,b3],dtype=int)]
    return abit

def sym2chip(asym):
    """Symbol to Chip Conversion"""
    from numpy import array
    IEEE805154_MAP = np.array([1,1,0,1,1,0,0,1,1,1,0,0,0,0,1,1,0,1,0,1,0,0,1,0,0,0,1,0,1,1,1,0,1,1,1,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1,1,0,1,0,1,0,0,1,0,0,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1,1,0,1,0,1,0,0,1,0,0,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1,1,0,1,0,1,0,1,0,1,0,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1,1,0,0,1,1,0,1,0,1,0,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,0,0,1,1,1,0,0,1,1,0,0,0,0,1,1,0,1,0,1,0,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,0,0,1,1,0,0,1,1,1,0,0,0,0,1,1,0,1,0,1,0,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,0,0,0,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,0,0,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,0,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,0,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,0,1,1,0,0,1,0,0,1,0,1,1,0,0,1,1,0,0,0,0,0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,1,1,0,0,0,0,0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,0,1,1,0,0,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,0])
    IEEE805154_MAP = IEEE805154_MAP.reshape(16,32)
    achip = array([IEEE805154_MAP[s] for s in asym])
    achip = achip.reshape(len(achip.flatten()))
    return achip

def half_sine_shape(data,Fc,Fs):
    """Half Sine Shaping according to IEEE 802.15.4 OQPSK Modulation"""
    from numpy import sin,pi,array,arange
    half_sine = sin(pi*Fc/Fs*arange(int(Fs/Fc)))
    output = array([half_sine if d==1 else -half_sine for d in data])
    return output.reshape(output.size)

    
    #Shift the signal to IF
def shift_up(i_in,q_in,F,Fs=16e6):
    """Shift IQ Signal Up to Frequency F"""
    from numpy import sin,cos,arange,pi
    n = arange(len(i_in))
    i_out = i_in*cos(2*pi*F/Fs*n) - q_in*sin(2*pi*F/Fs*n)
    q_out = i_in*sin(2*pi*F/Fs*n) + q_in*cos(2*pi*F/Fs*n)
    return (i_out,q_out)

def shift_down(i_in,q_in,F,Fs):
    """Shift IQ Signal Down to Frequency F"""
    from numpy import sin,cos,arange,pi
    n = arange(len(i_in))
    i_out =  i_in*cos(2*pi*F/Fs*n) + q_in*sin(2*pi*F/Fs*n)
    q_out = -i_in*sin(2*pi*F/Fs*n) + q_in*cos(2*pi*F/Fs*n)
    return (i_out,q_out)

def plot_fft(a,f_s=16e6,start=0,end=4e6,L=int(32e6)):
    import numpy as np
    import matplotlib.pyplot as plt
    F = np.fft.fftfreq(L,d=1.0/f_s)
    A = np.abs(np.fft.fft(a,L))/L*2
    plt.figure(figsize=(8,4),dpi=80)
    index = (F>start) & (F<end)
    plt.plot(F[index],A[index])
    plt.grid("on",which="major")
    plt.show()
    
    
def plot_fft_w(a,f_s=100e9,start=0,end=6e9,L=int(1e5)):
    import numpy as np
    import matplotlib.pyplot as plt
    
    #Zycling
    x = np.r_[a,a[0:L-len(a)%L]]  
    
    x = x.reshape(len(x)/L,L)
              
        
    F = np.fft.fftfreq(L,d=1.0/f_s)
    A = [np.abs(np.fft.fft(x_,L))/L*2 for x_ in x]
    A_m = np.mean(A,axis=0)
    plt.figure(figsize=(8,4),dpi=80)
    index = (F>start) & (F<end)
    plt.plot(F[index],A_m[index])
    plt.grid("on",which="major")
    plt.show()   
    
#plot_fft_w(i_rf_out)
#plot_fft_w(noise_in_i)
    



def modOQPSK(data_in,Fs=16e6,Fc=2e6):
    from numpy import r_,zeros,round
    iq_d = data_in.copy().reshape(len(data_in)//2,2)
    i_out = r_[half_sine_shape(iq_d[:,0],Fc/2,Fs),\
               zeros(int(round(Fs/Fc)))]
    q_out = r_[zeros(int(round(Fs/Fc)))       ,\
               half_sine_shape(iq_d[:,1],Fc/2,Fs)]
    return i_out,q_out


# Create random data Sequence
def randomSequence(bits):
    """Create random data Sequence"""
    raw_data = np.random.randint(0,2,bits)
    return raw_data
bits = 32 #Data Length

# Save the Generated Data

def saveBits(filename,raw_data):
    """Save Raw Bit Data"""
    fid = open(filename,'w')
    for sym in raw_data.reshape(bits/4,4):
        str_in = "%s\n"%(np.array2string(sym).strip('[]').replace(' ',''))
        print( "Write: <%s> from %s"%(str_in.strip(),np.array2string(sym)))
        fid.write(str_in)
    fid.close()


#Read symbol stream from a file

def readBits(filename_in,raw_data):
    """"Read Raw Bit Data"""
    fid = open(filename_in,'r')
    raw_data = np.array([],dtype=int)
    for sym in fid:
        isym = int(sym,2)
        i=3
        print( isym)
        while i >= 0:
            if( isym >= 2**i):
                raw_data = np.r_[raw_data,1]
                isym -= 2**i
            else:
                raw_data = np.r_[raw_data,0]
            i-=1;    
    fid.close()
    print( raw_data)



# Encapsulate Data to a Packet

def sym2packet(dsymbols,packetsize):
    """Encapsule Symbol Array to A Packet"""
    if(len(dsymbols)>(2*packetsize)): #WARNING TRUNCATION
        print( "!!!The Stream is too long!!!! Truncating")
        dsymbols = dsymbols[0:2*packetsize] # Times 2 because Octet has two symbols
    elif(len(dsymbols)<(2*packetsize)):
        print("!!!The Stream is too short!!! Zero Padding")
        rest = 2*packetsize-len(dsymbols)
        dsymbols = np.r_[np.zeros(rest),dsymbols]
            
    #print "DEBUG: Symbols to send %s"%np.array2string(dsymbols)
    
    frame_length_s= hex(packetsize)[2:]
    if( len(frame_length_s) < 2 ):  #In case packet size is smaller than 16
        frame_length_s = "0"+frame_length_s
    
    #print "DEBUG: The Frame Length is %s"%frame_length_s
    
    preamble = np.zeros(8,dtype=int)
    print( "DEBUG: preable is :"+np.array2string(preamble))
    sfd = np.array([7,10],dtype=int)
    print( "DEBUG: sfd is :"+np.array2string(sfd))
    phr = np.array([int(frame_length_s[1],16),int(frame_length_s[0],16)])
    print( "DEBUG: phr is :"+np.array2string(phr))
    
    packet = np.r_[preamble,sfd,phr,dsymbols]
    return packet


def plot_IQ(I,Q,start=0,end=16e-6,Fs=16e6,linewidth=4,figsize=(16,4)):
    import numpy as np
    import matplotlib.pyplot as plt
    """Plot Data"""
    
    k_end=int(end*Fs) # 16us
    k_start = int(start*Fs)
    plt.figure(figsize=figsize,dpi=80)
    
    axI=plt.subplot2grid((2,1),(0,0))
    plt.plot(np.arange(k_start,k_end)/Fs*1e6, \
             I[k_start:k_end], \
             color='blue', \
             linewidth=linewidth)
    plt.grid("on")
    plt.title("IQ Signal")
    axQ=plt.subplot2grid((2,1),(1,0))
    plt.plot(np.arange(k_start,k_end)/Fs*1e6,Q[k_start:k_end],'red', 
             linewidth=linewidth)
    plt.grid("on")
    plt.xlabel("t($\mu s$)")
    plt.show()
    
def oversample(data_in, Fin,Fout):
    """Upsampling from frequency Fin to Fout"""
    from numpy import ceil,array,linspace,zeros
    OSR = int(ceil(Fout/Fin))
    
    data_out = zeros(len(data_in)*OSR)
    
    data_out[0::OSR] = data_in[:] # Simple
    
    for i in range(0,len(data_in)-1):
        data_out[i*OSR:(i+1)*OSR]=linspace(data_in[i],data_in[i+1],num=OSR+1)[0:-1] #Smooth
    
    return data_out
def plot_fft2(a,b,f_s=16e6,start=0,end=4e6, \
              L=int(32e6),xscale='log',yscale='log', \
                linewidth=4, \
                figsize=(8,4)):
    from numpy import pi,abs,fft
    import matplotlib.pyplot as plt
    F = fft.fftfreq(L,d=1.0/f_s)
    A = abs(fft.fft(a,L)/float(L)*2)
    B = abs(fft.fft(b,L)/float(L)*2)
    plt.figure(figsize=(8,4),dpi=80)
    index = (F>start) & (F<end)
    plt.plot(F[index],A[index],'blue',\
             F[index],B[index],'red', \
            linewidth=linewidth)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.grid("on",which="major")
    plt.xlabel("Frequency (Hz)")
    plt.show()
    
def awgn(Iin,Qin,SNR_dB):
    """Introduction of The White Gaussian Noise to the Channel"""
    import numpy as np
    snr_linear = 10.0**(float(SNR_dB)/10.0)
    #print( "SNR LINEAR FACTOR: %e"%snr_linear)
    i_avg_energy = np.sum(Iin*Iin)/len(Iin)
    #print( "AVERAGE SIGNAL ENERGY %.2f"%i_avg_energy)
    noise_variance_in_i = i_avg_energy/snr_linear
    #print("NOISE VARIANCE: %e"%noise_variance_in_i)
    noise_in_i = np.sqrt(noise_variance_in_i)*np.random.randn(len(Iin))
    #noise_in_i = np.random.normal(0,noise_variance_in_i,len(i_rf_out))
    Iout = Iin + noise_in_i
    
    q_avg_energy = Qin.dot(Qin)/len(Qin)
    noise_variance_in_q = q_avg_energy/snr_linear
    noise_in_q = np.sqrt(noise_variance_in_q)*np.random.randn(len(Qin))
    Qout = Qin + noise_in_q
    
    #Sanity Check
    #print "SNRi=%.3f"%(10*np.log10(np.sum(Iin**2)/np.sum(noise_in_i**2)))
    #print "SNRq=%.3f"%(10*np.log10(np.sum(Qin**2)/np.sum(noise_in_q**2)))
    
    return Iout,Qout

def saveWave(filename,wave,Fs=None):
    import numpy as np
    fid = open(filename,'w')
    if Fs != None:
        for i in range(len(wave)):
            fid.write('%e\t%e\n'%(float(i)/float(Fs),wave[i]))
    else:
        for sample in wave:
            fid.write("%e\n"%sample)
 

def sample_down(inp,OSR,method='sample only'):
    if(method == 'cascaded down'):
        print( 'Oversampling Rate: %d'%OSR)
        outp = inp.copy()
        #Separate the process in multiple steps    
        aOSR=[10,5,5,5,5]
        print( "Decimation Steps:" + '->'.join(["%d"%d for d in aOSR]))
        for i in aOSR:
            outp = signal.decimate(outp,i)
            
    elif(method=='sample only'):
        print( 'Sample Only; Oversampling Rate: %d'%OSR)
        outp = inp[0::OSR]
    elif(method=='mean sample'):
        # Downsample ATTENTION implicit filter because of mean! 
        outp = np.mean(inp.reshape(len(inp)/OSR,OSR),axis=1)
    return outp
    
def safeIQIn(i,q,filename):
    fid = open(filename,'w')
    for k in range(0,len(i)):
        fid.write("%d\t%d\n"%(i[k],q[k]))
    fid.close()
    
def safeRealIQIn(i,q,filename):
    fid = open(filename,'w')
    for k in range(0,len(i)):
        fid.write("%e\t%e\n"%(i[k],q[k]))
    fid.close()
    
def safeRealIQInNPY(i,q,filename):
    from numpy import save, stack
    iq = stack([i,q],axis=1)
    save(filename,iq)
def loadRealIQFromNPY(filename):
    from numpy import load,array
    iq = load(filename)
    return iq[:,0], iq[:,1]
    
# Safe raw Symbols

def safePacket(pct,filename):
    fid = open(filename,'w')
    for sym in pct:
        fid.write('%s\n'%hex(sym)[2:-1])
    fid.close()



#Big Loop
def big_loop(f, \
             snr=20, \
             F_if=1.33e6, \
             F_rf=2.4e9, \
             STEP=1e-11, \
             S=0.5, \
             outWavefile='outWave.txt', \
             outPacketfile='outPacket', \
             createPacket=False, \
             readPacket=False, \
             packet_size = 20, \
             plot = False):
    F_s = 16e6 #Sample Frequency
    F_c = 2e6  #Chip Rate
    R_d = 250e3 # Data Rate (Fixed and not used in script)
    #F_if = 1.33e6 # Intermediate Frequency
    #F_rf = 2.4e9 # RF Carrier Frequency
    #STEP = 1e-11 # Analog Resolution
    my_snr = snr
    print( "Target SNR %.3f"%my_snr)
    SNRdB = my_snr + 10*np.log10(3e6*STEP) #SNR TODO: the second factor is BW ratio is it fixed??
    print( "SNR set for This run %.3f"%SNRdB)
    seed = f+10 #seed for random noise generator TODO: COMMENTED OUT
    F_rf_cutoff = 5e6 # Cutoff Frequency of the RF Filter
    filter_option = 'buter' # RF Filter Type
    BWIDTH = 11 #Modem Input Signal Bitwidth Sign Bit not included,
    #S = 0.5 # Scaling of the Signal (1=100% of bitwidth)
    
    iqIn_filename = outWavefile
    packet_filename = outPacketfile
    packet = []
    if (createPacket):
        #packet_size = 127 # Packet Size in Octets
        print( "GENERATE PACKET WITH SNR of %e dB (Iteration %d)"%(my_snr,f))
        raw_bits = randomSequence(packet_size*8)
        d_symbols = bit2sym(raw_bits)
        #d_symbols = np.r_[np.arange(16,dtype=int),11,0,0,11]
        #d_symbols = np.r_[d_symbols,d_symbols]
        packet = sym2packet(d_symbols,packet_size)
    elif(readPacket):
        fid = open(outPacketfile,'r')
        tmp = fid.readlines()
        packet = np.array([int(h,16) for h in tmp],dtype=int)
        print( "Modulate Packe from File")
    else:
        print( "GENERATE LONG IEEE STREAM of %d octets"%packet_size)
        raw_bits = randomSequence(packet_size*8)
        packet = bit2sym(raw_bits)
    
    safePacket(packet,packet_filename)
    #Convert Symbols to Chips
    #TODO ??? 
    
    print( "packet to send" + np.array2string(packet))
    
    #chip_seq = np.array([IEEE805154_MAP[s] for s in d_symbols]).flatten()
    data_in = packet
    
    chip_seq = sym2chip(data_in)
    print( "First  <Symbol>-<Chip>: <%d> - <"%packet[0] + "".join(["%d"%d for d in chip_seq[0:32]])+">")
    print( "Second <Symbol>-<Chip>: <%d> - <"%packet[1] + "".join(["%d"%d for d in chip_seq[32:64]])+">")
    print( "Third  <Symbol>-<Chip>: <%d> - <"%packet[2] + "".join(["%d"%d for d in chip_seq[64:96]])+">")
    
    # OQPSK Modulation with Half Sine Shaping
    
    
    
    # ----------------------------DIGITAL/ANALOG-----------------------#
    
    

            
    #INPUT: chip_seq
    
    
    print( "Data Rate %.1f"%R_d)
    print( "Symbol Rate:%.3f, Symbol Duration:%f us"%(R_d/4,4/R_d*1e6))
    print( "Chip Rate:%.3fMc/s"%(R_d/4*32/1e6))
    print( "Chip Rate for every Path:%.3fMc/s (%.2fus for a chip)"%(R_d/4*32/2/1e6,4/R_d/32*2*1e6))
    
    
    i_out,q_out = modOQPSK(chip_seq,F_s,F_c)
    
    if(plot):
        plot_IQ(i_out,q_out)
    #
    
    #n = np.arange(len(i_out))
    #i_if_out = i_out*np.cos(2*np.pi*F_if/F_s*n) - q_out*np.sin(2*np.pi*F_if/F_s*n)
    #q_if_out = i_out*np.sin(2*np.pi*F_if/F_s*n) + q_out*np.cos(2*np.pi*F_if/F_s*n)
    
    i_if_out,q_if_out = shift_up(i_out,q_out,F_if,F_s)
    
    #Plot Data
    if(plot):
        plot_IQ(i_if_out,q_if_out)
    
    
    
    #Shift the signal down from IF (Sanity Check)
    
    
    
    #n = np.arange(len(i_out))
    #i_b_out = i_if_out*np.cos(2*np.pi*F_if/F_s*n) + q_if_out*np.sin(2*np.pi*F_if/F_s*n)
    #q_b_out = -i_if_out*np.sin(2*np.pi*F_if/F_s*n) + q_if_out*np.cos(2*np.pi*F_if/F_s*n)
    #i_b_out,q_b_out = shift_down(i_if_out,q_if_out,F_if,F_s)
    
    #Plot Data
    #plot_IQ(i_b_out,q_b_out)
     
    #plot_fft(i_out,L=len(i_out)/100)
    
    
    
    #Oversample  
    
    
    #------------------------------------------------->Upsample
    
    OSF = int(np.floor(1/(F_s*STEP)))
    ones = np.ones(OSF)
    
    i_if_os_out = oversample(i_if_out,F_s,1/STEP)
    q_if_os_out = oversample(q_if_out,F_s,1/STEP)
    #plot_fft(i_if_os_out,f_s=1/STEP,end=4e6)
    
    
    
    #Plot Data
    
    #start=int(0/STEP) # 100ns
    #stop =int(1e-6/STEP) # 150ns 
    #plt.figure(figsize=(16,4),dpi=80)
    #plt.plot(np.arange(start,stop)*STEP*1e9,i_if_os_out[start:stop])
    #plt.grid("on")
    #plt.ylabel("I Path")
    #plt.xlabel("t(ns)")
    #
    #
    #plt.show()
    
    
    
    #--------------------------------------->Mixup to RF
    
    t=np.arange(len(i_if_os_out),dtype=float)*STEP
    
    i_rf_out = i_if_os_out * np.cos(2*np.pi*F_rf*t) - q_if_os_out * np.sin(2*np.pi*F_rf*t)
    q_rf_out = i_if_os_out * np.sin(2*np.pi*F_rf*t) + q_if_os_out * np.cos(2*np.pi*F_rf*t)
    
    
    #plot_fft2(i_if_os_out,i_rf_out,f_s=1/STEP,end=3e9,L=len(i_if_os_out)/1000)
    
    #Plot Data
    
    #start=int(0/STEP) # 100ns
    #stop =int((16e-6)/STEP) # 150ns 
    #plt.figure(figsize=(16,4),dpi=80)
    #plt.plot(np.arange(start,stop)*STEP*1e9,i_rf_out[start:stop])
    #plt.grid("on")
    #plt.ylabel("I Path")
    #plt.xlabel("t(ns)")
    #
    #
    #plt.show()
    
    
    # Tripple the message with 3 symbols break
    #pause = np.zeros(int(np.round(48e-6/STEP)))
    #i_rf_out = np.r_[pause,i_rf_out,pause,i_rf_out,pause,i_rf_out]
    #q_rf_out = np.r_[pause,q_rf_out,pause,q_rf_out,pause,q_rf_out]
    
    
    #-------------------------------------->Add Noise
    
    
    # Consider to Increase SNR correspondingly to signal BW
    # For BW of 1.5MHz SNR_effective is SNR+10*log10(SignalBW/NoiseBw)
    
    i_rf_in,q_rf_in = awgn(i_rf_out,q_rf_out,SNRdB)
    
    # Mix Down From RF Frequency
    t=np.arange(len(i_if_os_out),dtype=float)*STEP
    i_if_os_in = i_rf_in * np.cos(2*np.pi*F_rf*t) + q_rf_in * np.sin(2*np.pi*F_rf*t)
    q_if_os_in = -i_rf_in * np.sin(2*np.pi*F_rf*t) + q_rf_in * np.cos(2*np.pi*F_rf*t)
    
    #Plot Data
    #start=int(0/STEP) # 100ns
    #stop =int(16e-6/STEP) # 150ns 
    #plt.figure(figsize=(16,4),dpi=80)
    #plt.plot(np.arange(start,stop)*STEP*1e9,i_if_os_in[start:stop])
    #plt.grid("on")
    #plt.ylabel("I Path")
    #plt.xlabel("t(ns)")
    #plt.show()
    
    
    #---------------------------------------> Low Pass Filter
    from scipy import signal
    import matplotlib.pyplot as plt
    
        
    i_loc_out = 0
    q_loc_out = 0
    
    
    if(filter_option=='fir'):
        taps=256
        F_cutoff=F_rf_cutoff*STEP
        b = signal.firwin(taps,F_cutoff,nyq=1/STEP)
        i_loc_out = signal.filtfilt(b,(1),i_if_os_in)
        q_loc_out = signal.filtfilt(b,(1),q_if_os_in)
    elif(filter_option == 'buter'):
        W_cutoff = F_rf_cutoff*STEP
        N,Wn = signal.buttord(W_cutoff,W_cutoff+0.1,1,200)
        b,a = signal.butter(N,Wn,btype='low')
        print( 'Apply Butterworth Fitler Wn=%e,N=%d'%(Wn,N))
        print( 'b='+np.array2string(a))
        print( 10*'-')
        print( 'a='+np.array2string(b))
        i_loc_out = signal.filtfilt(b,a,i_if_os_in,method='gust')
        q_loc_out = signal.filtfilt(b,a,q_if_os_in,method='gust')
       
    
    #Plot Data
    #start=int(0e-6/STEP) # 100ns
    #stop =int(16e-6/STEP) # 150ns 
    #plt.figure(figsize=(16,4),dpi=80)
    #plt.plot(#np.arange(start,stop)*STEP*1e9,i_if_os_in[start:stop],'red',\
    #         np.arange(start,stop)*STEP*1e9,i_loc_out[start:stop],'blue',\
    #         np.arange(start,stop)*STEP*1e9,i_if_os_out[start:stop],'red')
    ##plt.ylim([-1.5,1.5])
    #plt.grid("on")
    #plt.ylabel("I Path")
    #plt.xlabel("t(ns)")
    #plt.show()
    
    #------------------------------>SAMPLE DOWN
    
    i_if_os_f_in = i_loc_out
    q_if_os_f_in = q_loc_out
    
    
    
    
    i_loc_in = i_if_os_f_in
    q_loc_in = q_if_os_f_in
    
    i_if_in = sample_down(i_loc_in,int(1/F_s/STEP))
    q_if_in = sample_down(q_loc_in,int(1/F_s/STEP))
    
    #Plot Data
    
    #start1=int(0/STEP) # 0
    #start2=int(0e-6*F_s) 
    #stop1 =int(16e-6/STEP) # 16us 
    #stop2 =int(16e-6*F_s)
    #plt.figure(figsize=(16,4),dpi=80)
    #plt.plot(np.arange(start2,stop2)/F_s*1e9,\
    #         i_if_in[start2:stop2], \
    #         np.arange(start2,stop2)/F_s*1e9,\
    #         i_if_out[start2:stop2])
    #plt.grid("on")
    #plt.ylabel("I Path")
    #plt.xlabel("t(ns)")
    #
    #
    #plt.show()
    
    
    
    # Ideal Dynamic range of i,q signal = [-1,1]
    # TODO: Is amplification needed?? 
    #
    # ----------------------------ANALOG/DIGITAL-----------------------#
    
    
    #Scale: 1.0 Maximal Input Power
    #
    
    i_if_s_in = S * i_if_in
    q_if_s_in = S * q_if_in
    
    # Quantize 
    
    A_MAX = 2**BWIDTH-1
    i_if_qt_in = np.array([np.round(i*A_MAX) for i in i_if_s_in],dtype=int)
    q_if_qt_in = np.array([np.round(i*A_MAX) for i in q_if_s_in],dtype=int )
    
    # ADC: Clipping
    i_if_c_in = np.array([min(A_MAX,max(-A_MAX,i)) for i in i_if_qt_in],dtype=int)
    q_if_c_in = np.array([min(A_MAX,max(-A_MAX,i)) for i in q_if_qt_in],dtype=int)
    
    
    if(plot):
        #Plot Data
        start=int(0e-6*F_s)
        stop =int(16e-6*F_s) #16us 
        t = np.arange(start,stop)/F_s*1e6 # In us
        plt.figure(figsize=(16,2),dpi=80)
        plt.plot(t, i_if_c_in[start:stop],'blue', \
             t, S*float(2**BWIDTH-1)*i_if_out[start:stop],'red')
        plt.grid("on")
        plt.ylabel("I Path")
        plt.xlabel("t(ns)")
        plt.show()
    
    safeIQIn(i_if_c_in,q_if_c_in,iqIn_filename)
#np.random.seed(0)    
#for i in range (0,50):    
#    big_loop(i);    
    
#%% Not Executed yet
def mergeIQFiles(outfilename,\
                 basefilename,\
                 nums, \
                 pause=48e-6,\
                 Fs=16e6,\
                 replace_char='<n>'):
    fid_out = open(outfilename,'w')
    for i in nums:
        for k in range(0,int(round(pause*Fs))):
            fid_out.write('0\t0\n')
        fid_in = open(basefilename.replace(replace_char,'%d'%i),)
        fid_out.write(fid_in.read())
        fid_in.close()
    fid_out.close()
#mergeIQFiles('S0p5SNRm10to50IQ.txt','S0p5SNRm10to50IQ<n>.txt',7)
#mergeIQFiles('S0p1SNRm40to20IQ.txt','S0p1SNRm40to20IQ<n>.txt',7)
#mergeIQFiles('S0p75SNRm10to50IQ.txt','SNRm10to50IQ<n>.txt',7) 
#mergeIQFiles('S0p5SNR1p2to3p9IQ_1.txt','S0p5SNR1p2to3IQ<n>.txt',10,pause=300e-6)  
#%% Recover from file
def readIQIn(filename):
    from re import compile
    from numpy import array
    i = []
    q = []
    p = compile(r'^(-?\d+)\s+(-?\d+)$')
    fid = open(filename,'r')
    for line in fid:
        m = p.match(line.strip())
        if(m):
            i.append(int(m.group(1)))
            q.append(int(m.group(2)))
        else:
            print( "Line <%s> did not match")
    fid.close()
    return array(i,dtype=int),array(q,dtype=int)
#filename = 'S0p5SNR0to9IQ9.txt'
#(i,q) = readIQIn(filename)

#p1 = np.sum(i**2)/len(i)
#p2 = np.sum(i_if_out**2)/len(i_if_out)*(S*float(2**BWIDTH-1))**2

#print "Real SNR:%.2f"%(10*np.log10(p2/(p2-p1)))

#%% is Filtering necessary
#plot_fft(q)
    
#%% Shift to Baseband WARNING NOT NECESSARY FOR demodZBinpIQ.txt the signal is on baseband already and is normally shifted in the testbench!!
def shift_to_baseband(): #TODO NOT USED YET AS A FUNCTION
    F_if = 1.44e6
    i_loc_in = i
    q_loc_in = q
    
    i_loc_out,q_loc_out = shift_down(i_loc_in,q_loc_in,F_if,F_s)
    
    #Plot Data    
    
    start=int(32e-6*F_s) 
    stop =int(48e-6*F_s) #16us 
    t = np.arange(start,stop)/F_s*1e6 # In us
    plt.figure(figsize=(16,4),dpi=80)
    plt.plot(t, i_loc_out[start:stop],'blue', \
    #         t, S*float(2**BWIDTH-1)*i_out[start2:stop2],'red'\
    )
    plt.grid("on")
    plt.ylabel("I Path")
    plt.xlabel("t(ns)")
    
    #Go on
    i_in_noisy = i_loc_out
    q_in_noisy = q_loc_out
    

#%% Low Pass Filter for Fs=16Mhz
def filter_bb_signal(): #TODO NOT USED YET AS A FUNCTION
    Wn = 7e6/F_s
    b,a = signal.butter(8,Wn,'low')
    w,h = signal.freqz(b,a)
    print ("H = (%s)/(%s)"%(np.array2string(b),np.array2string(a)))
    plt.figure(figsize=(8,4),dpi=80)
    plt.plot(w,20*np.log10(h))
    plt.xscale('log')
    plt.title('Butterworth bandpass filter fit to constraints')
    plt.xlabel('Frequency (rad)')
    plt.ylabel('Amplitude [dB]')
    plt.grid("on")
    plt.show()
    #%% Low Passband Filter
    i_loc_in = i
    q_loc_in = q
    Wn = 3e6/F_s
    taps = 4
    b,a = signal.butter(taps,Wn,'low')
    w,h = signal.freqz(b,a)
    
    print( "Filter Data with: H = (%s)/(%s)"%(np.array2string(b),np.array2string(a)))
    plt.figure(figsize=(8,4),dpi=80)
    plt.plot(w,20*np.log10(h))
    plt.xscale('log')
    plt.title('Butterworth bandpass filter fit to constraints')
    plt.xlabel('Frequency (rad)')
    plt.ylabel('Amplitude [dB]')
    plt.grid("on")
    plt.show()
    
    i_loc_out = signal.lfilter(b,a,i_loc_in)
    q_loc_out = signal.lfilter(b,a,q_loc_in)
    
    
    i_loc_out = np.r_[i_loc_out[taps:],np.zeros(taps)]
    q_loc_out = np.r_[q_loc_out[taps:],np.zeros(taps)]
    #Plot Data    
    start=0 
    stop =int(16e-6*F_s) #16us 
    t = np.arange(start,stop)/F_s*1e6 # In us
    plt.figure(figsize=(16,4),dpi=80)
    plt.plot(t, i_loc_out[start:stop],'blue', \
    #         t, S*float(2**BWIDTH-1)*i_out[start2:stop2],'red'\
    )
    plt.grid("on")
    plt.ylabel("I Path")
    plt.xlabel("t(ns)")
    plt.show()
    plt.figure(figsize=(16,4),dpi=80)
    plt.plot(t, q_loc_out[start:stop],'blue', \
    #         t, S*float(2**BWIDTH-1)*q_out[start2:stop2],'red'\
    )
    plt.grid("on")
    plt.ylabel("I Path")
    plt.xlabel("t(ns)")
    plt.show()
    
    i_in_filtered = i_loc_out
    q_in_filtered = q_loc_out
    

#%% Signal Synchronisation
def synchronize(): #TODO NOT USED YET AS A FUNCTION
    # Correlate preamble and whole input
    preamble = np.array([0,0,0,0,0,0,0,0])
    
    cpreamble = sym2chip(preamble)
    #cpreamble = cpreamble[::-1]
    cpreamble = cpreamble.reshape(len(cpreamble)/2,2)
    ipreamble = half_sine_shape(cpreamble[:,0],F_c/2,F_s)
    qpreamble = half_sine_shape(cpreamble[:,1],F_c/2,F_s)
    
    avr_ppi = np.sqrt(np.sum(np.power(ipreamble,2))/len(ipreamble))
    avr_ppq = np.sqrt(np.sum(np.power(qpreamble,2))/len(qpreamble))
    
    avr_ipi = np.sqrt(np.sum(np.power(i,2))/len(i))
    avr_ipq = np.sqrt(np.sum(np.power(q,2))/len(q))
    
    ic_preamble = np.correlate(i,ipreamble,mode='full')/(avr_ppi*avr_ipi)
    qc_preamble = np.correlate(q,qpreamble,mode='full')/(avr_ppq*avr_ipq)



#%% Find Peaks in the Correlation
def find_peaks_in_corr(): #TODO NOT USED YET AS A FUNCTION
    index = np.arange(len(ic_preamble))
    threshold = 1680
    ic_preamble[ic_preamble<threshold]=0
    ipeaks = index[ic_preamble>=threshold]
    print( ipeaks)
    #%% Find the peak (preamble detection)
    start=int(318e-6*F_s) 
    stop =int(322e-6*F_s) #16us 
    start = 5120
    stop = 5125
    t = np.arange(start,stop)#/F_s*1e6 # In us
    plt.plot(t,ic_preamble[start:stop])
    plt.show()
    #start=0 
    #stop =int(360*16e-6*F_s) #16us 
    #t = np.arange(start,stop)/F_s*1e3 # In us
    #plt.plot(t,qc_preamble[start:stop])
    #plt.show()
    
    #First max at 5123
#%%
def plot_results_not_verified():
    #Plot Data   
    ksynch = 5124
     
    start= 0
    stop = int(64e-6*F_s) #16us 
    t = np.arange(start,stop)/F_s*1e6 # In us\
    plt.figure(figsize=(16,4),dpi=80)
    plt.plot(t, i[ksynch-len(ipreamble)+start:stop+ksynch-len(ipreamble)],'blue', \
             t, avr_ipi*ipreamble[0:stop],'red'\
    )
    plt.grid("on")
    plt.ylabel("I Path")
    plt.xlabel("t(ns)")
    plt.show()
    
    
    ksynch_1 = (ksynch-len(ipreamble))%int(np.round(F_s/F_c*32))
    print( "Synch Factor:%d"%ksynch_1)
    
    

#%% OQPSK Demodulation Synchron Approach

def OQPSK_synch_demod_todo():
    i_loc_in = i
    q_loc_in = q
    
    #Down Sampling to chip rate (for each path separately)
    OSF1 = int(np.floor(2*F_s/F_c))
    
    
    
    
    #Synch the Data
    i_loc_in = i_loc_in[ksynch_1:]
    q_loc_in = q_loc_in[ksynch_1:]
    
    q_shift = int(np.ceil(F_s/F_c))
    #Shift Q path by Tc = 1/Fc
    q_loc_in = q_loc_in[q_shift:]
    i_loc_in = i_loc_in[0:-(q_shift)]
    
    
                        
    
    D1 = int(np.round(len(i_loc_in))/OSF1)
    
    print( D1)
    
    #Zero Padding                 
    if(D1*OSF1 > len(i_loc_in)):
        i_loc_in = np.r_[i_loc_in, np.zeros(D1*OSF1-len(i_loc_in))]
        q_loc_in = np.r_[q_loc_in, np.zeros(D1*OSF1-len(q_loc_in))]
    elif(D1*OSF1 < len(i_loc_in)):
        D1+=1
        i_loc_in = np.r_[i_loc_in, np.zeros(D1*OSF1-len(i_loc_in))]
        q_loc_in = np.r_[q_loc_in, np.zeros(D1*OSF1-len(q_loc_in))]
                         
    print( D1*OSF1)
    print( len(i_loc_in))
    print( len(q_loc_in))
    # Downsample WARN implicit filter
    i_d = np.mean(i_loc_in.reshape(D1,OSF1),axis=1)
    q_d = np.mean(q_loc_in.reshape(D1,OSF1),axis=1)
    #
    ##Zero Comparator
    i_b = np.zeros(len(i_d),dtype=int)
    i_b[i_d>=0] = 1 
       
    q_b = np.zeros(len(q_d),dtype=int)
    q_b[q_d>=0] = 1 
    
    ##Combine to Chip   
    
    chip_seq_r = np.c_[i_b,q_b].reshape(2*len(i_b))   
     
    print( np.array2string(i_b[0:8]))
    print( np.array2string(q_b[0:8]))
      
    print( "First  Sent <Symbol>-<Chip>: < %d> - <"%d_symbols[0] + "".join(["%d"%d for d in chip_seq[0:32]])+">")
    print ("First  Sent <Symbol>-<Chip>: <XX> - <"+"".join(["%d"%d for d in chip_seq_r[0:32]])+">")
    
    print ("First  Sent <Symbol>-<Chip>: < %d> - <"%d_symbols[1] + "".join(["%d"%d for d in chip_seq[32:64]])+">")
    print( "First  Sent <Symbol>-<Chip>: <XX> - <"+"".join(["%d"%d for d in chip_seq_r[32:64]])+">")
    
    print( "First  Sent <Symbol>-<Chip>: < %d> - <"%d_symbols[1] + "".join(["%d"%d for d in chip_seq[32:64]])+">")
    print( "First  Sent <Symbol>-<Chip>: <XX> - <"+"".join(["%d"%d for d in chip_seq_r[32:64]])+">")
                                                         
                                                         

#%% Chip Correlation 1

#Calculate Correlation

def chip_corr_todo():
    tmp_in = chip_seq_r.reshape(len(chip_seq_r)/32,32)
    
    
    #c_{av}[k] = sum_over_n a[n+k] * conj(v[n])
    #
    symbol_c_1 = np.array([ np.correlate(a,v) for a in tmp_in for v in IEEE805154_MAP ]).reshape(len(chip_seq_r)/32,16)
    np.set_printoptions(edgeitems=8)
    print( symbol_c_1)
    
    recovered_data_1 = [i for sym in symbol_c_1 for i in range(0,16) if sym[i]==sym.max()]
    print( recovered_data_1)

#%% Chip Correlation 2
def chip_corr_2_todo():
    tmp1 = chip_seq_r.copy()
    tmp1[tmp1==0]=-1
    tmp1 = tmp1.reshape(len(tmp1)/32,32)
    tmp2 = IEEE805154_MAP.copy()
    tmp2[tmp2==0]=-1
    symbol_c_2 = np.array([a.dot(b) for a in tmp1 for b in tmp2]).reshape(len(tmp1),16)
    print( symbol_c_2)
    recovered_data_2 = [i for sym in symbol_c_2 for i in range(0,16) if sym[i]==max(sym.max(),abs(sym.min()))]
    print( recovered_data_2)

#%% BER Computation

def comp_ber_todo():
    fid = open('output.txt','r')
    modem_out_txt = fid.read()
    fid.close()
    
    fid = open("snrPacket1.txt",'r')
    modem_in = []
    modem_in.append(fid.read())
    fid.close()
    
    for i in range(2,10):
        fid = open('snrPacket%d.txt'%i,'r')
        modem_in.append(fid.read())
        fid.close()

    for packet in modem_in:
        print( packet.replace('\n',''))

    import re
    symbol_p = re.compile(r'\w+')
    p1 = symbol_p.findall(modem_in[0])
    p1 = [int(d,16) for d in p1[10:]]
    print( p1      )
    
    pout = symbol_p.findall(modem_out_txt)
    pout = [int(d) for d in pout]
    print( pout)
    
    p1_bit = sym2bit(np.array(p1,dtype=int))
    pout_bit = sym2bit(np.array(pout,dtype=int))
    
    pct_num = len(pout_bit)/len(p1_bit)
    pout_bit = pout_bit[0:pct_num*len(p1_bit)]
    pout_bit = pout_bit.reshape(pct_num,len(p1_bit))
    print( pout_bit )
    p1_bit = np.array([p1_bit for i in range(0,pct_num)])
    print( p1_bit)
    
    ber = np.sum(p1_bit^pout_bit,axis=1)/float(len(p1_bit[0,:]))
    print( ber)
    print( len (ber))
    
    tmp = np.c_[p1_bit[0,:] ,pout_bit[0,:]]
    
    for i in range(len(p1)):
        if (p1[i]==pout[i]):
            print( '<%s>\t<%s>'%(bin(p1[i]),bin(pout[i])))
        else:
            print( 'Error->\t\t<%s>\t<%s>'%(bin(p1[i]),bin(pout[i])))
