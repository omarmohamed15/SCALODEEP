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

def yc_patch(A,l1,l2,o1,o2):

    n1,n2=np.shape(A);
    tmp=np.mod(n1-l1,o1)
    if tmp!=0:
        print(np.shape(A), o1-tmp, n2)
        A=np.concatenate([A,np.zeros((o1-tmp,n2))],axis=0)

    tmp=np.mod(n2-l2,o2);
    if tmp!=0:
        A=np.concatenate([A,np.zeros((A.shape[0],o2-tmp))],axis=-1); 


    N1,N2 = np.shape(A)
    X=[]
    for i1 in range (0,N1-l1+1, o1):
        for i2 in range (0,N2-l2+1,o2):
            tmp=np.reshape(A[i1:i1+l1,i2:i2+l2],(l1*l2,1));
            X.append(tmp);  
    X = np.array(X)
    return X[:,:,0]

def yc_patch_inv(X1,n1,n2,l1,l2,o1,o2):
    
    tmp1=np.mod(n1-l1,o1)
    tmp2=np.mod(n2-l2,o2)
    if (tmp1!=0) and (tmp2!=0):
        A     = np.zeros((n1+o1-tmp1,n2+o2-tmp2))
        mask  = np.zeros((n1+o1-tmp1,n2+o2-tmp2)) 

    if (tmp1!=0) and (tmp2==0): 
        A   = np.zeros((n1+o1-tmp1,n2))
        mask= np.zeros((n1+o1-tmp1,n2))


    if (tmp1==0) and (tmp2!=0):
        A    = np.zeros((n1,n2+o2-tmp2))   
        mask = np.zeros((n1,n2+o2-tmp2))   


    if (tmp1==0) and (tmp2==0):
        A    = np.zeros((n1,n2))
        mask = np.zeros((n1,n2))

    N1,N2= np.shape(A)
    ids=0
    for i1 in range(0,N1-l1+1,o1):
        for i2 in range(0,N2-l2+1,o2):
            #print(i1,i2)
    #       [i1,i2,ids]
            A[i1:i1+l1,i2:i2+l2]=A[i1:i1+l1,i2:i2+l2]+np.reshape(X1[:,ids],(l1,l2))
            mask[i1:i1+l1,i2:i2+l2]=mask[i1:i1+l1,i2:i2+l2]+ np.ones((l1,l2))
            ids=ids+1


    A=A/mask;  
    A=A[0:n1,0:n2]

    return A

def Patching(Data,w1,w2,s1z,s2z):
    
    ach0 = np.reshape(Data[0].data, ((Data[0].data).shape[0],1))
    ach1 = np.reshape(Data[1].data, ((Data[1].data).shape[0],1))
    ach2 = np.reshape(Data[2].data, ((Data[2].data).shape[0],1))

    ch0 = yc_patch(ach0,w1,w2,s1z,s2z)
    ch1 = yc_patch(ach1,w1,w2,s1z,s2z)
    ch2 = yc_patch(ach2,w1,w2,s1z,s2z)
    
    return ch0,ch1,ch2

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



def SCALO_Continous_Windows_Overlapping_EvenTime(Data,ch0,ch1,ch2,model,n1,n2,w1,w2,s1z,s2z):
    astart = Data[0].stats.starttime
    eventime=[]
    labindx=[]
    evne=[]
    windindex=[]
    laball=[]
    ci=0

    i=0
    kk = 0
    dt = 1/Data[0].stats.sampling_rate
    f_min = 1
    f_max = 45*Data[0].stats.sampling_rate/100
    fs = Data[0].stats.sampling_rate
    winlen = 3000
    le =3000
    ccx = 6000
    temp0pall = []
    temp1pall = []
    temp2pall = []

    for kq in range(0,int(ch0.shape[0])):

            print(str(kq) + '/' + str(ch0.shape[0]))
            a0=np.zeros((ccx))
            a1=np.zeros((ccx))
            a2=np.zeros((ccx))
            a0[0:winlen] = ch0[kq]
            a1[0:winlen] = ch1[kq]
            a2[0:winlen] = ch2[kq]

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


            temp0 = np.reshape(temp0, (temp0.shape[0],temp0.shape[1] ,1 ))
            temp1 = np.reshape(temp1, (temp1.shape[0],temp1.shape[1] ,1 ))
            temp2 = np.reshape(temp2, (temp2.shape[0],temp2.shape[1] ,1 ))

            temp0p=np.zeros((7,3000,1))
            temp1p=np.zeros((7,3000,1))
            temp2p=np.zeros((7,3000,1))
            temp0p[:,0:le,:]= temp0[:,0:le,:]
            temp1p[:,0:le,:]= temp1[:,0:le,:]
            temp2p[:,0:le,:]= temp2[:,0:le,:]

            temp0pall.append(temp0p)
            temp1pall.append(temp1p)
            temp2pall.append(temp2p)
    lab = model.predict([temp0pall,temp1pall,temp2pall], batch_size=1024, verbose = 1)
    lab = lab[:,0,:,0]
    lab = np.transpose(lab)

   
    labf = yc_patch_inv(lab,n1,n2,w1,w2,s1z,s2z)
    
    return labf