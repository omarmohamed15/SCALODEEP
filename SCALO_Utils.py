from scipy.signal import butter, lfilter, lfilter_zi
from keras.models import load_model
from scipy import ndimage,misc
import time
import numpy as np
from obspy.signal.tf_misfit import cwt

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
    
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter_zi(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    zi = lfilter_zi(b, a)
    y,zo = lfilter(b, a, data, zi=zi*data[0])
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def load_SCALODEEP_Model(filname):
    model =load_model(filname)
    return model

def Check_Sampling_Return_Data(st):
    print('Checking the Sampling Rate')

    if st[0].stats.sampling_rate==100:
        print('The Sampling Rate is 100 Hz')
        data_E = st[0].data
        data_N = st[1].data
        data_Z = st[2].data
    else:
        print('Resampling the data to 100 Hz')
        st_E1 = st[0]
        data_E = st_E1.resample(100)
        data_E = data_E.data

        st_N1 = st[1]
        data_N = st_N1.resample(100)
        data_N = data_N.data

        st_Z1 = st[2]
        data_Z = st_Z1.resample(100)
        data_Z = data_Z.data
        
    return data_E,data_N, data_Z

def Return_Data(st):

    data_E = st[0].data
    data_N = st[1].data
    data_Z = st[2].data

        
    return data_E,data_N, data_Z

def SCALO_Continous_Windows_WIthoutOverlapping(st,data_E,data_N,data_Z,model,thre):
    astart = st[0].stats.starttime
    eventime=[]
    labindx=[]
    evne=[]
    windindex=[]
    laball=[]

    i=0
    kk = 0
    dt = 1/st[0].stats.sampling_rate
    f_min = 1
    f_max = 45*st[0].stats.sampling_rate/100
    winlen = 3000
    le =3000

    for kq in range(0,int(np.size(data_E)/winlen)):
        
        print(str(kq) + '/' + str(int(np.size(data_E)/winlen))) 
        a0=np.zeros((6000))
        a1=np.zeros((6000))
        a2=np.zeros((6000))
        a0[0:winlen]= data_E[kk:kk+winlen]
        #a0[winlen:] = data_E[kk+winlen]
        a1[0:winlen] = data_N[kk:kk+winlen]
        #a1[winlen:] = data_N[kk+winlen]
        a2[0:winlen] = data_Z[kk:kk+winlen]

        kk = kk+winlen



        sig0 = butter_bandpass_filter_zi(a0, f_min, f_max, st[0].stats.sampling_rate, order=10)
        sig0 = sig0 / np.max(np.abs(sig0))

        sig1 = butter_bandpass_filter_zi(a1, f_min, f_max, st[0].stats.sampling_rate, order=10)
        sig1 = sig1 / np.max(np.abs(sig1))

        sig2 = butter_bandpass_filter_zi(a2, f_min, f_max, st[0].stats.sampling_rate, order=10)
        sig2 = sig2 / np.max(np.abs(sig2))


        temp0 = cwt(sig0, dt, 8, f_min, f_max, nf=10,  wl='morlet')
        temp0 = np.clip(np.abs(temp0)[-1::-1], 0, 100)
        temp0 = temp0[3:,:]
        temp0 = temp0 / np.max(np.abs(temp0))
        temp0x = temp0
        #scalogram0_1.append(temp0)

        temp1 = cwt(sig1, dt, 8, f_min, f_max, nf=10,  wl='morlet')
        temp1 = np.clip(np.abs(temp1)[-1::-1], 0, 100)
        temp1 = temp1[3:,:]
        temp1 = temp1 / np.max(np.abs(temp1))

        #scalogram0_2.append(temp1)
        #scalogram1 = scalogram1 / np.max(np.abs(scalogram1))

        temp2 = cwt(sig2, dt, 8, f_min, f_max, nf=10,  wl='morlet')
        temp2 = np.clip(np.abs(temp2)[-1::-1], 0, 100)
        temp2 = temp2[3:,:]
        temp2 = temp2 / np.max(np.abs(temp2))

        temp0 = np.reshape(temp0, (1,temp0.shape[0],temp0.shape[1] ,1 ))
        temp1 = np.reshape(temp1, (1,temp1.shape[0],temp1.shape[1] ,1 ))
        temp2 = np.reshape(temp2, (1,temp2.shape[0],temp2.shape[1] ,1 ))

        temp0p=np.zeros((1,7,3000,1))
        temp1p=np.zeros((1,7,3000,1))
        temp2p=np.zeros((1,7,3000,1))
        temp0p[:,:,0:le,:]= temp0[:,:,0:le,:]
        temp1p[:,:,0:le,:]= temp1[:,:,0:le,:]
        temp2p[:,:,0:le,:]= temp2[:,:,0:le,:]


        lab = model.predict([temp0p,temp1p,temp2p])
        lab = np.where(lab>thre,1,0)
        lab = lab [0,0,:,0]
        #ascent = misc.ascent()
        #lab = ndimage.median_filter(lab, size=5)

        #print(kq,kk)
        if len(np.where(lab==1)[0])>0:
            zoo = astart + (np.where(lab==1)[0][0] /st[0].stats.sampling_rate) + ((kk-3000)/st[0].stats.sampling_rate)
            eventime.append(zoo) 
            #labindx.append((np.where(lab==1)[0][0]))
            windindex.append(kq)
            laball.append(lab)
            
    return eventime, windindex, laball


def SCALO_Continous_Windows_Overlapping(st,data_E,data_N,data_Z,model,thre):
    astart = st[0].stats.starttime
    eventime=[]
    labindx=[]
    evne=[]
    windindex=[]
    laball=[]

    i=0
    kk = 0
    dt = 1/st[0].stats.sampling_rate
    f_min = 1
    f_max = 45*st[0].stats.sampling_rate/100
    fs = st[0].stats.sampling_rate
    winlen = 3000
    le =3000
    ccx = 6000

    labf=[]
    #for kq in range(0,int(len(tr1)/winlen)-1):
    #for kq in range(0,int(len(tr1)/winlen)):
    labfx = np.zeros((int(len(data_E)/1000),3000))
    labfinal = np.zeros((len(data_E)))
    for kq in range(0,int(len(data_E)/1000)-2):
        
        print(str(kq) + '/' + str(int(len(data_E)/1000)-2))
        a0=np.zeros((ccx))
        a1=np.zeros((ccx))
        a2=np.zeros((ccx))
        a0[0:winlen]= data_E[kk:kk+winlen]
        #a0[winlen:] = data_E[kk+winlen]
        a1[0:winlen] = data_N[kk:kk+winlen]
        #a1[winlen:] = data_N[kk+winlen]
        a2[0:winlen] = data_Z[kk:kk+winlen]
        #a2[winlen:] = data_Z[kk+winlen]
        kk = kk+1000

        sig0 = butter_bandpass_filter_zi(a0, f_min, f_max, fs, order=10)
        sig0 = sig0 / np.max(np.abs(sig0))

        sig1 = butter_bandpass_filter_zi(a1, f_min, f_max, fs, order=10)
        sig1 = sig1 / np.max(np.abs(sig1))

        sig2 = butter_bandpass_filter_zi(a2, f_min, f_max, fs, order=10)
        sig2 = sig2 / np.max(np.abs(sig2))


        #sig0 = a0
        #sig1 = a1
        #sig2 = a2

        temp0 = cwt(sig0, dt, 8, f_min, f_max, nf=10,  wl='morlet')
        temp0 = np.clip(np.abs(temp0)[-1::-1], 0, 100)
        temp0 = temp0[3:,:]
        temp0 = temp0 / np.max(np.abs(temp0))


        temp1 = cwt(sig1, dt, 8, f_min, f_max, nf=10,  wl='morlet')
        temp1 = np.clip(np.abs(temp1)[-1::-1], 0, 100)
        temp1 = temp1[3:,:]
        temp1 = temp1 / np.max(np.abs(temp1))

        temp2 = cwt(sig2, dt, 8, f_min, f_max, nf=10,  wl='morlet')
        temp2 = np.clip(np.abs(temp2)[-1::-1], 0, 100)
        temp2 = temp2[3:,:]
        temp2 = temp2 / np.max(np.abs(temp2))

        temp0 = np.reshape(temp0, (1,temp0.shape[0],temp0.shape[1] ,1 ))
        temp1 = np.reshape(temp1, (1,temp1.shape[0],temp1.shape[1] ,1 ))
        temp2 = np.reshape(temp2, (1,temp2.shape[0],temp2.shape[1] ,1 ))


        temp0p=np.zeros((1,7,3000,1))
        temp1p=np.zeros((1,7,3000,1))
        temp2p=np.zeros((1,7,3000,1))
        temp0p[:,:,0:le,:]= temp0[:,:,0:le,:]
        temp1p[:,:,0:le,:]= temp1[:,:,0:le,:]
        temp2p[:,:,0:le,:]= temp2[:,:,0:le,:]


        lab = model.predict([temp0p,temp1p,temp2p])
        #lab = np.where(lab>0.8,1,0)
        lab = lab [0,0,:,0] 



        #print(count)

        #count = count +1
        #labf = np.append(labf,lab[0:winlen])
        #labfx[kq,:] = lab[0:winlen];
        if kq ==0:
            labfinal[kq*1000:kq*1000+1000] =lab[0:winlen][0:1000]
            tmeplab0 = lab[0:winlen]
        elif kq ==1:
            labfinal[kq*1000:kq*1000+1000] =(tmeplab0[1000:2000] +  lab[0:winlen][0:1000] )/2
            tmeplab1 = lab[0:winlen]
        else:
            labfinal[kq*1000:kq*1000+1000] = (tmeplab0[2000:3000] + tmeplab1[1000:2000] + lab[0:winlen][0:1000])/3
            tmeplab0 = np.copy(tmeplab1)
            tmeplab1 = lab[0:winlen]
            
            
    labf = np.where(labfinal>thre,1,0)
    
    return labf



def SCALO_Continous_Windows_Overlapping_EvenTime(st,data_E,data_N,data_Z,model,thre,ReturnLabel=False):
    astart = st[0].stats.starttime
    eventime=[]
    labindx=[]
    evne=[]
    windindex=[]
    laball=[]
    ci=0
    
    i=0
    kk = 0
    dt = 1/st[0].stats.sampling_rate
    f_min = 1
    f_max = 45*st[0].stats.sampling_rate/100
    fs = st[0].stats.sampling_rate
    winlen = 3000
    le =3000
    ccx = 6000

    labf=[]
    #for kq in range(0,int(len(tr1)/winlen)-1):
    #for kq in range(0,int(len(tr1)/winlen)):
    labfx = np.zeros((int(len(data_E)/1000),3000))
    labfinal = np.zeros((len(data_E)))
    for kq in range(0,int(len(data_E)/1000)-2):
        
        print(str(kq) + '/' + str(int(len(data_E)/1000)-2))
        a0=np.zeros((ccx))
        a1=np.zeros((ccx))
        a2=np.zeros((ccx))
        a0[0:winlen]= data_E[kk:kk+winlen]
        #a0[winlen:] = data_E[kk+winlen]
        a1[0:winlen] = data_N[kk:kk+winlen]
        #a1[winlen:] = data_N[kk+winlen]
        a2[0:winlen] = data_Z[kk:kk+winlen]
        #a2[winlen:] = data_Z[kk+winlen]
        kk = kk+1000

        sig0 = butter_bandpass_filter_zi(a0, f_min, f_max, fs, order=10)
        sig0 = sig0 / np.max(np.abs(sig0))

        sig1 = butter_bandpass_filter_zi(a1, f_min, f_max, fs, order=10)
        sig1 = sig1 / np.max(np.abs(sig1))

        sig2 = butter_bandpass_filter_zi(a2, f_min, f_max, fs, order=10)
        sig2 = sig2 / np.max(np.abs(sig2))


        #sig0 = a0
        #sig1 = a1
        #sig2 = a2

        temp0 = cwt(sig0, dt, 8, f_min, f_max, nf=10,  wl='morlet')
        temp0 = np.clip(np.abs(temp0)[-1::-1], 0, 100)
        temp0 = temp0[3:,:]
        temp0 = temp0 / np.max(np.abs(temp0))


        temp1 = cwt(sig1, dt, 8, f_min, f_max, nf=10,  wl='morlet')
        temp1 = np.clip(np.abs(temp1)[-1::-1], 0, 100)
        temp1 = temp1[3:,:]
        temp1 = temp1 / np.max(np.abs(temp1))

        temp2 = cwt(sig2, dt, 8, f_min, f_max, nf=10,  wl='morlet')
        temp2 = np.clip(np.abs(temp2)[-1::-1], 0, 100)
        temp2 = temp2[3:,:]
        temp2 = temp2 / np.max(np.abs(temp2))

        temp0 = np.reshape(temp0, (1,temp0.shape[0],temp0.shape[1] ,1 ))
        temp1 = np.reshape(temp1, (1,temp1.shape[0],temp1.shape[1] ,1 ))
        temp2 = np.reshape(temp2, (1,temp2.shape[0],temp2.shape[1] ,1 ))


        temp0p=np.zeros((1,7,3000,1))
        temp1p=np.zeros((1,7,3000,1))
        temp2p=np.zeros((1,7,3000,1))
        temp0p[:,:,0:le,:]= temp0[:,:,0:le,:]
        temp1p[:,:,0:le,:]= temp1[:,:,0:le,:]
        temp2p[:,:,0:le,:]= temp2[:,:,0:le,:]


        lab = model.predict([temp0p,temp1p,temp2p])
        #lab = np.where(lab>0.8,1,0)
        lab = lab [0,0,:,0] 



        #print(count)

        #count = count +1
        #labf = np.append(labf,lab[0:winlen])
        #labfx[kq,:] = lab[0:winlen];
        if kq ==0:
            tmeplab0=lab[0:winlen][0:1000]
            tmeplab0all = lab[0:winlen]

        elif kq ==1:
            tmeplab1 =(tmeplab0all[1000:2000] +  lab[0:winlen][0:1000] )/2
            tmeplab1all = lab[0:winlen]

        else:
            tmeplab2 = (tmeplab0all[2000:3000] + tmeplab1all[1000:2000] + lab[0:winlen][0:1000])/3
            tmeplab0all = np.copy(tmeplab1all)
            tmeplab1all = lab[0:winlen]

            if np.mod(kq,3)==0:
                   
                if ReturnLabel==True:
                    #print(np.shape(labwin))
                    laball.append(labwin)
                #labwin = np.concatenate([tmeplab0,tmeplab1,tmeplab2])
                #print(kq,np.shape(labwin))
                #plt.plot(labwin)
                #plt.show()


                ind = np.where(labwin==1)[0]
                if (len(ind)>1) and (ind[0]>0):
                    eventime.append(astart + ci*((kq*1000)-1000)/st[0].stats.sampling_rate + (ind[0]/st[0].stats.sampling_rate))
                ci = ci+1
                labwin = []
                labwin = np.concatenate([labwin,tmeplab2])
                labwin = np.where(labwin>thre,1,0)

            else:
                if kq ==2:
                    labwin = np.concatenate([tmeplab0,tmeplab1,tmeplab2])
                    labwin = np.where(labwin>thre,1,0)
                else:
                    labwin = np.concatenate([labwin,tmeplab2])
                    labwin = np.where(labwin>thre,1,0)
    if ReturnLabel==True:
        #print(np.shape(labwin))
        laball.append(labwin)
    return laball, eventime