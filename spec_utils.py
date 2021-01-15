import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from astropy.time import Time
from astropy import coordinates as coor
from astropy.constants import c,h, k_B, G, M_sun, au
import os.path
from astropy.io import fits
from scipy.signal import savgol_filter
from scipy import interpolate
import pandas as pd
import math
import matplotlib.pyplot as plt
import pdb
import json
from numpy.lib.recfunctions import append_fields
from .helpers import fwhm_to_sigma, sigma_to_fwhm, extract_hitran_data

def binspec(wave, flux, n):
#bin spectrum by # of bins n

#    if(np.remainder(n,2) == 0):
#        print("Even numbers not accepted by this program")
#        return 

    new_length=math.floor(np.size(wave)/n)
    newwave=np.zeros(new_length)
    newint=np.zeros(new_length)

    for i in np.arange(new_length-2):
         hw=math.floor(n/2)
         i_old=i*n+hw
         newwave[i]=np.mean(wave[i_old-hw:i_old+hw+1])
         newint[i]=np.sum(flux[i_old-hw:i_old+hw+1])

    mybool=(newint==0)
    newint[mybool]='NaN'
    return {'wave':newwave, 'flux':newint/n}

def make_line_profile(wave, flux, w0, vwidth=50, norm=None,v_dop=0):
    vel=(wave-w0)/w0*c.value*1e-3  #km/s
    flux=flux[(vel>(-vwidth)) & (vel<vwidth)]
    vel=vel[(vel>(-vwidth)) & (vel<vwidth)]
    
    if(norm=='Maxmin'):
        flux=(flux-np.nanmin(flux))/np.nanmax(flux-np.nanmin(flux))
    vel=vel-v_dop

    return (vel, flux)

def spec_combine(wavelist,fluxlist):
    n=len(wavelist)
    minlist=[np.min(mywavelist) for mywavelist in wavelist]
    maxlist=[np.max(mywavelist) for mywavelist in wavelist]
    minmin=np.min(minlist)
    maxmax=np.max(maxlist)
    whichmin=np.argmin(minlist)
    whichmax=np.argmax(maxlist)
    wave=np.concatenate([wavelist[whichmin][wavelist[whichmin]<np.min(wavelist[whichmax])],wavelist[whichmax]])

    nlist=np.zeros(np.size(wave))
    totflux=np.zeros(np.size(wave))
    for i,mywave in enumerate(wavelist):
        f=interp1d(mywave,fluxlist[i],bounds_error=False)   #Interpolate fluxes onto new grid
        myflux_interp=f(wave)
        fin=np.isfinite(myflux_interp)                           #Find where fluxes are finite
        nfin=np.logical_not(fin)                                    #Find where fluxes are NOT finite
        add=np.zeros(np.size(myflux_interp))                     #Create zero array
        add[fin]=1                                               #If flux is finite, set "add" to 1
        nlist+=add                                               #Add to sum 
        myflux_interp[nfin]=0                                    #Set non-finite fluxes to zero
        totflux+=myflux_interp                                   #Add all fluxes
    totflux[nlist!=0]=totflux[nlist!=0]/nlist[nlist!=0]    #Divide by number of spectra to find average
    totflux[nlist==0]='NaN'

    return (wave, totflux)

def line_overlap_mband(wave, flux, type='12co1-0', dv=3., voffset=0, p_list=[None], r_list=[None],norm=None):

#dv = resolution of final overlapped line in km/s
#type = '12co1-0', '12co2-1', '13co1-0'
#voffset = Doppler shift of source - can be adjusted based on data

#Get lines from HITRAN and select for appropriate line type
    if((type == '12co1-0' ) | (type == '12co2-1')):
        hitran_data=extract_hitran_data('CO',4.4,5.2,isotopologue_number=1, eupmax=10000., aupmin=0.005)
    if((type == '13co1-0' )):
        hitran_data=extract_hitran_data('CO',4.4,5.2,isotopologue_number=2, eupmax=10000., aupmin=0.005)

    w0=np.array(1e4/hitran_data['linecenter'])
    vup =np.array([json.loads(vp) for vp in hitran_data['Vp']])
    vlow =np.array([json.loads(vpp) for vpp in hitran_data['Vpp']])
    Qlow=hitran_data['Qpp']

    if((type == '12co1-0') | (type == '13co1-0')):
        vbool=(vup==1) & (vlow==0)
        w0=w0[vbool]
        Qlow=Qlow[vbool]
        vup=vup[vbool]
        vlow=vlow[vbool]

    if((type == '12co2-1')):
        vbool=[(vup==2) & (vlow==1)]
        w0=w0[vbool]
        Qlow=Qlow[vbool]
        vup=vup[vbool]
        vlow=vlow[vbool]

    if(any(p_list) or any(r_list)):
        keep=np.zeros(np.size(Qlow), dtype=bool)
        for i in np.arange(np.size(Qlow)):
            spl=Qlow.iloc[i].split()
            qtype=spl[0]
            qnum=np.int(spl[1])
            if(any(p_list)):
                if((qtype=='P') & (qnum in p_list)):
                    keep[i]=True
            if(any(r_list)):
                if((qtype=='R') & (qnum in r_list)):
                    keep[i]=True
        Qlow=Qlow.iloc[keep]
        w0=w0[keep]
        vup=vup[keep]
        vlow=vlow[keep]

#Select only lines that overlap with data
    w0=w0[ (w0 > np.min(wave)) & (w0 < np.max(wave))  ]
    nlines=np.size(w0)

#Offset w0 given Doppler shift
    w0*=(1+voffset*1e3/c.value)

#Make interpolation grid
    nvels=151
    nlines=np.size(w0)
    interpvel=np.arange(nvels)*dv-75.*dv
    interpind=np.zeros((nvels,nlines))+1  #keeps track of weighting for each velocity bin
    interpflux=np.zeros((nvels,nlines))

#Loop through all w0 values
    for i,my_w0 in enumerate(w0):
        mywave = wave[(wave > (my_w0-0.003)) & (wave < (my_w0+0.003))]
        myflux = flux[(wave > (my_w0-0.003)) & (wave < (my_w0+0.003))]
        myvel = c.value*1e-3*(mywave - my_w0)/my_w0
        f1=interp1d(myvel, myflux, kind='linear', bounds_error=False)
        interpflux[:,i]=f1(interpvel)
        w=np.where((interpvel > np.max(myvel)) | (interpvel < np.min(myvel)) | (np.isfinite(interpflux[:,i]) != 1 )  ) #remove fluxes beyond edges, NaNs
        if(np.size(w) > 0):
            interpind[w,i]=0
            interpflux[w,i]=0

    interpflux=np.sum(interpflux,1)/np.sum(interpind,1)
    if(norm=='Maxmin'):
        interpflux=(interpflux-np.nanmin(interpflux))/np.nanmax(interpflux-np.nanmin(interpflux))

    return (interpvel,interpflux)

def centers_to_corners(x):
    newx=(x[1:]+x[:np.size(x)-1])/2.
    xcorners=np.concatenate([[x[0]-(x[1]-x[0])/2.], newx])
    return xcorners

def frexp10(x):
    exponent=np.floor(np.log10(x))
    mantissa=10.**(np.log10(x)-exponent)
    return mantissa, exponent

def spec_convol(wave, flux, dv):

#Program assumes units of dv are km/s, and dv=FWHM

    dv=fwhm_to_sigma(dv)
    n=round(4.*dv/(c.value*1e-3)*np.median(wave)/(wave[1]-wave[0]))
    if (n < 10):
        n=10.

#Pad arrays to deal with edges
    dwave=wave[1]-wave[0]
    wave_low=np.arange(wave[0]-dwave*n, wave[0]-dwave, dwave)
    wave_high=np.arange(np.max(wave)+dwave, np.max(wave)+dwave*(n-1.), dwave)
    nlow=np.size(wave_low)
    nhigh=np.size(wave_high)
    flux_low=np.zeros(nlow)
    flux_high=np.zeros(nhigh)
    mask_low=np.zeros(nlow)
    mask_high=np.zeros(nhigh)
    mask_middle=np.ones(np.size(wave))
    wave=np.concatenate([wave_low, wave, wave_high])
    flux=np.concatenate([flux_low, flux, flux_high])
    mask=np.concatenate([mask_low, mask_middle, mask_high])

    newflux=np.copy(flux)

    if( n > (np.size(wave)-n)):
        print("Your wavelength range is too small for your kernel")
        print("Program will return an empty array")

    for i in np.arange(n, np.size(wave)-n+1):
        lwave=wave[np.int(i-n):np.int(i+n+1)]
        lflux=flux[np.int(i-n):np.int(i+n+1)]
        lvel=(lwave-wave[np.int(i)])/wave[np.int(i)]*c.value*1e-3 
        nvel=(np.max(lvel)-np.min(lvel))/(dv*.2) +3
        vel=np.arange(nvel)
        vel=.2*dv*(vel-np.median(vel))
        kernel=markgauss(vel,mean=0,sigma=dv,area=1.)
        f = interp1d(vel,kernel, bounds_error=False)
        wkernel=f(lvel)
        wkernel=wkernel/np.nansum(wkernel)
        newflux[np.int(i)]=np.nansum(lflux*wkernel)/np.nansum(wkernel[np.isfinite(lflux)])
        #Note: denominator is necessary to correctly account for NaN'd regions

#Remove NaN'd regions
    nanbool=np.invert(np.isfinite(flux))   #Places where flux is not finite
    newflux[nanbool]='NaN'

#Now remove padding
    newflux=newflux[mask==1]

    return newflux

def read_nirspec(filename):
    f=open(filename)
    lines=f.readlines()
    words=lines[0].split()
    headerlines=int(words[2])
    title=lines[headerlines-2].split()
    out=np.recfromtxt(filename, skip_header=headerlines, names=title)
    cont=fit_mband_continuum(out[title[0]], out[title[1]])
    out=append_fields(out, 'Continuum', data=cont, dtypes=None, fill_value=-1, usemask=True, asrecarray=False)
    return out

def rad_from_vel(vobs, i=90, mstar=1., iunits='degrees', vunits='km/s'):

    if(vunits == 'km/s'):
        vobs=vobs*1e3   #Convert from km/s to m/s
    else:
        print('Assuming velocity is in m/s')

    if(iunits == 'degrees'):
        i=i*np.pi/180.
    else:
        print('Assuming inclination is in radians')

    vactual=vobs/np.sin(i)

    R=(G.value*mstar*M_sun.value/vactual**2.)/au.value   #Returns R in AU

    return R

def vel_from_rad(rad, i=90, mstar=1., iunits='degrees', vunits='km/s'):

    if(iunits == 'degrees'):
        i=i*np.pi/180.
    else:
        print('Assuming inclination is in radians')

    vactual=np.sqrt(G.value*mstar*M_sun.value/(rad*au.value))
    vobs=vactual*np.sin(i)

    if(vunits == 'km/s'):
        vobs=vobs*1e-3   #Convert from m/s to km/s
    else:
        print('Assuming velocity is in m/s')

    return vobs


def fit_mband_continuum(wave, orig):

    #Define locations for continuum fitting
    xval0=np.array([4.664])
    xval1=4.678+np.arange(12)*0.0088 
    xval2=4.963+np.arange(12)*0.011
    xvals=np.concatenate([xval0,xval1,xval2])

    finwave=wave[np.isfinite(orig)]
    wspl=[]
    for idx, myx in enumerate(xvals):
        myw=np.where(np.abs(wave-myx) == np.min(np.abs(finwave-myx)))[0][0]
        wspl.append(myw)
    wspl=np.array(wspl)
    wspl=wspl.astype(np.int64)

    coeffs=np.polyfit(wave[wspl], orig[wspl], 1)
    p=np.poly1d(coeffs)
    spl=p(wave)
    return spl

def fit_irs_spline(wave, orig, sdfac=None):
  
  if(sdfac == None):
     sdfac=1.5

  x0=np.array([9.89,10.12,10.50, 10.73,11.07,11.34,11.50,11.59,11.75,11.84,11.98,12.30,12.47,12.67,12.92,13.34,13.55,
  14.07,14.23,14.59,14.80,15.22,15.40,15.85,16.16,16.43,17.27,17.42,17.88,18.06,18.40,18.78,19.49,
  20.13,20.52,21.03,21.67,22.29,22.46,23.26,23.70,24.18,24.73,25.43,25.65,25.83,26.30,26.82,
  27.73,28.30,28.48,28.84,29.21,29.72,30.08,30.58,31.58,31.82,
  32.17,32.36,32.87,33.38,34.08,36.42])

  x1=np.array([9.91,10.16, 10.53, 10.75, 11.09,11.37, 11.52, 11.62,11.79, 11.87, 12.11, 12.33, 12.49,12.72,
  12.95, 13.41,13.58, 14.11, 14.29, 14.69,14.87,15.28, 15.45,15.92, 16.19, 16.49, 17.31, 17.45,
  17.90, 18.14, 18.44, 18.95, 19.54, 20.18, 20.54, 21.08,21.74,22.31, 22.48, 23.28,23.74,
  24.21, 24.83, 25.46, 25.68, 25.89, 26.38, 26.87, 27.76, 28.34, 28.53,
  28.87, 29.26, 29.76, 30.12, 30.63, 31.62, 31.85, 32.24,
  32.38, 32.89, 33.41, 34.13, 36.50])

  wspl=np.array([])
  for idx, (myx0,myx1) in enumerate(zip(x0,x1)):
    w=np.where((wave >= myx0) & (wave <= myx1))[0]
    if(np.size(w) > 0):
      w2=np.where((orig == np.min(orig[w])) & (wave > myx0) & (wave < myx1))[0]  #Find index of min
      wmin=w[0]-5
      if(wmin < 0):
        wmin=0
      wmax=w[np.size(w)-1]+5
      if(wmax > (np.size(orig)-1)):
        wmax=np.size(orig)-1   
      mn=np.mean(orig[wmin:wmax])
      sd=np.std(orig[wmin:wmax])
      if(np.abs(orig[w2]-mn) < sdfac*sd):
        wspl=np.concatenate((wspl, w2))
      elif( (idx <= 4) or (np.size(x0)-idx <= 4) ):
        wspl=np.concatenate((wspl, w2))

  wspl=wspl.astype(np.int64)

  tck = interpolate.splrep(wave[wspl], orig[wspl], k=1)
  spl = interpolate.splev(wave, tck, der=0)

  return spl


def read_irs(src_name, orderc=True, shflag=False, lhflag=False, silicate=False, ice=False,
noerror=False):

  #Warning: This routine includes hardcoded fixes for some sources

  np.seterr(invalid='ignore')
  both= not (lhflag or shflag)

  #Hardcoded directory location
  #Eventually include an error catch here?
  dd='/Users/csalyk/DATA/Spitzer/All/REDUCED/'

  sherrflag=False
#Read in SH part of spectrum
  if(shflag or both):
    shfile=dd+src_name+'_SH_final.fits'
    file_exists_sh=os.path.isfile(shfile) 
    if (not file_exists_sh):
      print('File '+shfile+' not found')
    if (file_exists_sh==False and both==False):
      return
    if (file_exists_sh):
      data=fits.getdata(shfile,1)   #Read in FITS file for SH data
      spec=data['SPEC'][0]
      wave=data['WAVE'][0]
      wave=wave[1:]
      spec=spec[1:]

      if (noerror ==  False):
        error=data['ERROR'][0]
        error=error[1:]

#Get rid of nans
      w= np.isfinite(spec)

      if (src_name == 'COKUTAU4'):
        w= (np.isfinite(spec)) & ( (wave<10.43) | (wave>10.683) )

      if (src_name == 'SZ18'):
        w= (np.isfinite(spec)) & (wave < 11.94) & (wave > 11.98)

      orig=np.copy(spec[w])   #Original spectrum

      w2= (wave > 10.263) & (wave < 10.644) & w   #Define "quiescent" region

#      if(src_name == 'IRAS03245+3002'):
#        w2=(wave > 13.285) & (wave < 13.604) & w
#      if(src_name == 'ST34'):
#        w2=(wave > 10) & (wave < 10.5) & w
#      if(src_name == 'DLCHA'):
#        w2=(wave > 18.5) & (wave < 19) & w
#      if(src_name == 'VVSER'):  
#        w2=(wave > 15) & (wave < 16) & w

#Compute standard deviation of spectrum in "quiescent" region
      lspec=spec[w2]
      lwave=wave[w2]

      z=np.polyfit(lwave, lspec, 1)
      p=np.poly1d(z)
      yfit=p(lwave)
      sd=np.std(lspec-yfit)
      win=np.abs(lspec-yfit) < 4.*sd
      sd=np.std((lspec-yfit)[win])
      cont=savgol_filter(spec[w],61,2)-2.5*sd
        
      shsd=sd

      if(silicate == True):
        wpah=(wave > 10.9) & (wave < 11.7) & w
        cont[wpah]=(savgol_filter(spec[w],5,2)-2.5*sd)[wpah]
        wpah=(wave > 12.583) & (wave < 13.348)
        cont[wpah]=(savgol_filter(spec[w],5,2)-2.5*sd)[wpah]
     
      if(ice == True):
        wice=(wave > 14.59) & (wave < 15.70)
        cont[wice]=(savgol_filter(spec[w],5,2)-2.5*sd)[wice]
  
      wave=wave[w]
      flux=spec[w]-cont

      if(noerror == False):
         error=np.copy(error[w])
      if(both == True):
         wave_sh=wave
         flux_sh=flux
         orig_sh=orig
         cont_sh=cont
         if(noerror == False):
           error_sh=error

  if(lhflag==True or both==True):
    lhfile=dd+src_name+'_LH_final.fits'
    file_exists_lh=os.path.isfile(lhfile) 
    if(file_exists_lh == False):
      print('File '+lhfile+' not found')
    if(file_exists_lh == False and both == False):
      return
    if(file_exists_lh):
      data=fits.getdata(lhfile,1)
      spec=np.array(data['SPEC'][0])
      wave=np.array(data['WAVE'][0])
      spec=spec[1:]
      wave=wave[1:]
      if(src_name == 'IQTAU'):
          spec=spec[1:]
      orig=np.copy(spec)
 
      if (noerror == False):
        error=np.copy(data['ERROR'][0])
        error=error[1:]
        if(src_name == 'IQTAU'):
            error=error[1:]

      w=(spec == spec)   #Check for nan
      w2=(wave > 24) & (wave < 24.3)
      w3=(wave > 26.1) & (wave < 26.4)
      if(src_name == 'EXLUP'):
        w2=(wave > 22.5) & (wave < 23)
      if(src_name == 'GITAU'):
        w2=(wave > 24.3) & (wave < 24.7)
      if(src_name == 'HQTAU'):
        w2=(wave > 28) & (wave < 29)
      if(src_name == '04216+2603'):
        w2=(wave > 24.3) & (wave < 24.5)
      if(np.sum(w2) > 0):
        myw=(w & w2)
        lspec2=spec[myw] 
        lwave=wave[myw]
        z=np.polyfit(lwave, lspec2, 1)
        p=np.poly1d(z)
        yfit2=p(lwave)
        sd2=np.std(lspec2-yfit2)
      else:
        sd2=1000
      if(np.sum(w3) > 0):
        lspec3=spec[w3]
        lwave=wave[w3]
        z=np.polyfit(lwave, lspec3, 1)
        p=np.poly1d(z)
        yfit3=p(lwave)
        sd3=np.std(lspec3-yfit3)
      else:
        sd3=1000
      sd=sd2
      lspec=lspec2
      yfit=yfit2
      if((sd3 < sd2) and (sd2 != 1000) and (sd3 != 1000)):
        sd=sd3
        lspec=lspec3
        yfit=yfit3
      lhsd=sd

      win=(np.abs(lspec-yfit) < 4.*sd)
      sd=np.std((lspec-yfit)[win])

      cont=savgol_filter(spec[w],61,2)-2.5*sd
      wave=wave[w]
      flux=spec[w]-cont
      orig=orig[w]   #Original spectrum

      if(noerror == False):
        error=np.copy(error[w])
      if(both == True):
        wave_lh=wave
        flux_lh=flux
        orig_lh=orig
        cont_lh=cont
        if(noerror == False):
          error_lh=error

  if(both and file_exists_lh and file_exists_sh and orderc==False):
    w=(wave_lh > np.max(wave_sh))
    wave_lh=wave_lh[w]
    flux_lh=flux_lh[w]
    cont_lh=cont_lh[w]
    orig_lh=orig_lh[w]
    if(noerror == False):
      error_lh=error[w]

    wave=np.concatenate((wave_sh, wave_lh))
    flux=np.concatenate((flux_sh, flux_lh))
    cont=np.concatenate((cont_sh, cont_lh))
    orig=np.concatenate((orig_sh, orig_lh))

    if (noerror == False):
      error=np.concatenate((error_sh,error_lh))

  if(both and file_exists_lh and file_exists_sh and orderc==True):

    w=(wave_lh > np.max(wave_sh))
    wave_lh=wave_lh[w]
    flux_lh=flux_lh[w]
    cont_lh=cont_lh[w]
    orig_lh=orig_lh[w]
    error_lh=error_lh[w]
    wave=np.concatenate((wave_sh, wave_lh))

    ws=(wave_sh > 15) & (wave_sh < 19.4) & (cont_sh==cont_sh)
    p=np.poly1d(np.polyfit(wave_sh[ws], cont_sh[ws], 2))
    wl=(wave_lh > 19.7) & (wave_lh < 21)
    yfit = p(wave_lh[wl])   

    diff = np.mean( (cont_lh[wl]-yfit)[0:4] )
    flux_lh = flux_lh
    cont_lh = cont_lh - diff
    orig_lh = orig_lh - diff

    flux=np.concatenate((flux_sh, flux_lh))
    cont=np.concatenate((cont_sh, cont_lh))
    orig=np.concatenate((orig_sh, orig_lh))
    if(noerror == False):
      error=np.concatenate((error_sh, error_lh))

  spl=fit_irs_spline(wave, orig)
  if(src_name == "WAOPH6"):
    spl=fit_irs_spline(wave, orig, sdfac=2.5)

  cont=spl
  flux=orig-spl

#Need to figure out how to deal with NaN's here
#  wbad=(flux < -0.1) | np.isnan(flux)  Used to have this, but not sure it's a good idea
  wbad=np.isnan(flux)
  if(np.size(wbad) > 0):
    wgood=(flux > -0.1) & (np.isfinite(flux))
    f = interp1d(wave[wgood], flux[wgood], bounds_error=False)
    newfluxes=f(wave[wbad])
    flux[wbad]=newfluxes

  out={}
  out['wave']=wave
  out['flux']=flux
  out['cont']=cont
  out['orig']=orig
  out['error']=error

  out=pd.DataFrame.from_dict(out)

  return out

def contsub_irs_sh(wave,flux, error=None,silicate=False, ice=False):
    spec=flux
#Get rid of nans
    w=np.isfinite(spec)
    orig=np.copy(spec[w])   #Original spectrum

    w2= (wave > 10.263) & (wave < 10.644) & w   #Define "quiescent" region
#Compute standard deviation of spectrum in "quiescent" region
    lspec=spec[w2]
    lwave=wave[w2]
    z=np.polyfit(lwave, lspec, 1)
    p=np.poly1d(z)
    yfit=p(lwave)
    sd=np.std(lspec-yfit)
    win=np.abs(lspec-yfit) < 4.*sd
    sd=np.std((lspec-yfit)[win])
    cont=savgol_filter(spec[w],61,2)-2.5*sd
        
    if(silicate == True):
        wpah=(wave > 10.9) & (wave < 11.7) & w
        cont[wpah]=(savgol_filter(spec[w],5,2)-2.5*sd)[wpah]
        wpah=(wave > 12.583) & (wave < 13.348)
        cont[wpah]=(savgol_filter(spec[w],5,2)-2.5*sd)[wpah]
     
    if(ice == True):
        wice=(wave > 14.59) & (wave < 15.70)
        cont[wice]=(savgol_filter(spec[w],5,2)-2.5*sd)[wice]
  
    wave=wave[w]
    flux=spec[w]-cont

    out={}
    out['wave']=wave
    out['flux']=flux
    out['cont']=cont
    out['orig']=orig
    if(error is not None):    
        out['error']=error[w]

    out=pd.DataFrame.from_dict(out)

    return out

def wn_to_k(eup, units='cgs'):

      if(units == 'cgs'):
          eup=eup*1e2

      return eup*h.value*c.value/k_B.value

def sigma_to_fwhm(sigma):
    return  sigma*(2.*np.sqrt(2.*np.log(2.)))

def fwhm_to_sigma(fwhm):
    return fwhm/(2.*np.sqrt(2.*np.log(2.)))


def markgauss(x,mean=0, sigma=1., area=1):

    norm=area        
    u = ( (x-mean)/np.abs(sigma) )**2            
    norm = norm / (np.sqrt(2. * np.pi)*sigma)
    f=norm*np.exp(-0.5*u)

    return f

def cvsgauss(x,*args):
    a0=1.
    a1=0
    a2=1.
    a3=0
    a4=0
    a5=0
    if (np.size(args) >= 1):
        a0=args[0]
    if (np.size(args) >= 2):
        a1=args[1]
    if (np.size(args) >= 3):
        a2=args[2]
    if (np.size(args) >= 4):
        a3=args[3]
    if (np.size(args) >= 5):
        a4=args[4]

    z = (x -a1) / a2
    y = a0*np.exp(-z**2/2.) + a3 + a4 * x + a5 * x**2
    return y

def calc_draine(wave0, A_V):
    out=np.recfromtxt("/Users/csalyk/mypy/draine.txt", names='wave, AlamoverAIC', skip_header=1)
    wave = out['wave'][::-1]
    AlamoverAIC=out['AlamoverAIC'][::-1]
    AVoverAIC=AlamoverAIC[wave==0.547]

    f = interp1d(wave, AlamoverAIC, bounds_error=False)
    Awave0overAIC=f(wave0)
    Alam=Awave0overAIC/AVoverAIC*A_V

    return Alam

#Planck function for [wave]=microns and [T]=K, returning Jy
def planck(wave, temp):
    wave=np.array(wave)

    w = wave / 1.E6          #Convert microns to meters
    v = 2.99792458e8/w       #Convert to frequency

#Define SI constants
    c1 =  4.62e-50                 # =2*!PI*h/c^2       
    c2 =  4.8e-11                  # =h/k
    val =  c2*v/temp

    bbflux = c1*v**3./(np.exp(val)-1)

    return bbflux*1.e26  #convert from W/m^2/Hz to Jy

def gauss6(x, a0, a1, a2, a3, a4, a5):
   z = (x - a1) / a2
   y = a0 * np.exp(-z**2 / 2.) + a3 + a4 * x + a5 * x**2
   return y

def gauss5(x, a0, a1, a2, a3, a4):
   z = (x - a1) / a2
   y = a0 * np.exp(-z**2 / 2.) + a3 + a4 * x
   return y

def gauss4(x, a0, a1, a2, a3):
   z = (x - a1) / a2
   y = a0 * np.exp(-z**2 / 2.) + a3
   return y

def gauss3(x, a0, a1, a2):
   z = (x - a1) / a2
   y = a0 * np.exp(-z**2 / 2.)
   return y

def gaussfit(xdata,ydata,nterms=4,p0=None,bounds=None) :
    options={6:gauss6, 5:gauss5, 4:gauss4, 3:gauss3}
    fit_func=options[nterms]
    try:
        if(bounds is not None): 
            fitparameters, fitcovariance = curve_fit(fit_func, xdata, ydata, p0=p0,bounds=bounds)
        else:
            fitparameters, fitcovariance = curve_fit(fit_func, xdata, ydata, p0=p0)
    except RuntimeError:
        print("Error - curve_fit failed")
        return -1

    fitoutput={"yfit":fit_func(xdata,*fitparameters),"parameters":fitparameters,
               "covariance":fitcovariance}

#    class fitoutput:
#        def __init__(self, yfit, parameters, covariance):
#            self.yfit = yfit
#            self.parameters = parameters
#            self.covariance = covariance

#    out=fitoutput(fit_func(xdata, *fitparameters), fitparameters, fitcovariance)
    return fitoutput

def gridspec(x, y, xbin_center):
      dx=xbin_center[1]-xbin_center[0]
      xbin_edges=np.insert(xbin_center+dx/2.,0,xbin_center[0]-dx/2.)
      bin_means = (np.histogram(x, bins=xbin_edges, weights=y)[0]/np.histogram(x, xbin_edges)[0])

      return bin_means


def baryvel(dje, deq):
   
   #Define constants
   dc2pi = 2 * np.pi
   cc2pi = 2 * np.pi
   dc1 = 1.0e0
   dcto = 2415020.0e0
   dcjul = 36525.0e0                     #days in Julian year
   dcbes = 0.313e0
   dctrop = 365.24219572e0               #days in tropical year (...572 insig)
   dc1900 = 1900.0e0
   au = 1.4959787e8
   
   #Constants dcfel(i,k) of fast changing elements.
   dcfel = np.array([1.7400353e00, 6.2833195099091e02, 5.2796e-6, 6.2565836e00, 6.2830194572674e02, -2.6180e-6, 4.7199666e00, 
   8.3997091449254e03, -1.9780e-5, 1.9636505e-1, 8.4334662911720e03, -5.6044e-5, 4.1547339e00, 5.2993466764997e01, 5.8845e-6, 
   4.6524223e00, 2.1354275911213e01, 5.6797e-6, 4.2620486e00, 7.5025342197656e00, 5.5317e-6, 1.4740694e00, 3.8377331909193e00, 5.6093e-6])
   dcfel = np.reshape(dcfel, (8, 3))
    
   #constants dceps and ccsel(i,k) of slowly changing elements.
   dceps = np.array([4.093198e-1, -2.271110e-4, -2.860401e-8])
   ccsel = np.array([1.675104e-2, -4.179579e-5, -1.260516e-7, 2.220221e-1, 2.809917e-2, 1.852532e-5, 1.589963e00, 3.418075e-2, 1.430200e-5, 
   2.994089e00, 2.590824e-2, 4.155840e-6, 8.155457e-1, 2.486352e-2, 6.836840e-6, 1.735614e00, 1.763719e-2, 6.370440e-6, 1.968564e00, 
   1.524020e-2, -2.517152e-6, 1.282417e00, 8.703393e-3, 2.289292e-5, 2.280820e00, 1.918010e-2, 4.484520e-6, 4.833473e-2, 1.641773e-4, 
   -4.654200e-7, 5.589232e-2, -3.455092e-4, -7.388560e-7, 4.634443e-2, -2.658234e-5, 7.757000e-8, 8.997041e-3, 6.329728e-6, -1.939256e-9, 
   2.284178e-2, -9.941590e-5, 6.787400e-8, 4.350267e-2, -6.839749e-5, -2.714956e-7, 1.348204e-2, 1.091504e-5, 6.903760e-7, 3.106570e-2,
   -1.665665e-4, -1.590188e-7])
   ccsel = np.reshape(ccsel, (17, 3))
    
   #Constants of the arguments of the short-period perturbations.
   dcargs = np.array([5.0974222e0, -7.8604195454652e2, 3.9584962e0, -5.7533848094674e2, 1.6338070e0, -1.1506769618935e3, 2.5487111e0, 
   -3.9302097727326e2, 4.9255514e0, -5.8849265665348e2, 1.3363463e0, -5.5076098609303e2, 1.6072053e0, -5.2237501616674e2, 1.3629480e0, 
   -1.1790629318198e3, 5.5657014e0, -1.0977134971135e3, 5.0708205e0, -1.5774000881978e2, 3.9318944e0, 5.2963464780000e1, 4.8989497e0, 
   3.9809289073258e1, 1.3097446e0, 7.7540959633708e1, 3.5147141e0, 7.9618578146517e1, 3.5413158e0, -5.4868336758022e2])
   dcargs = np.reshape(dcargs, (15, 2))
    
   #Amplitudes ccamps(n,k) of the short-period perturbations.
   ccamps = np.array([-2.279594e-5, 1.407414e-5, 8.273188e-6, 1.340565e-5, -2.490817e-7, -3.494537e-5, 2.860401e-7, 1.289448e-7, 
   1.627237e-5, -1.823138e-7, 6.593466e-7, 1.322572e-5, 9.258695e-6, -4.674248e-7, -3.646275e-7, 1.140767e-5, -2.049792e-5, -4.747930e-6, 
   -2.638763e-6, -1.245408e-7, 9.516893e-6, -2.748894e-6, -1.319381e-6, -4.549908e-6, -1.864821e-7, 7.310990e-6, -1.924710e-6, -8.772849e-7,
   -3.334143e-6, -1.745256e-7, -2.603449e-6, 7.359472e-6, 3.168357e-6, 1.119056e-6, -1.655307e-7, -3.228859e-6, 1.308997e-7, 1.013137e-7, 
   2.403899e-6, -3.736225e-7, 3.442177e-7, 2.671323e-6, 1.832858e-6, -2.394688e-7, -3.478444e-7, 8.702406e-6, -8.421214e-6, -1.372341e-6, 
   -1.455234e-6, -4.998479e-8, -1.488378e-6, -1.251789e-5, 5.226868e-7, -2.049301e-7, 0.e0, -8.043059e-6, -2.991300e-6, 1.473654e-7, 
   -3.154542e-7, 0.e0, 3.699128e-6, -3.316126e-6, 2.901257e-7, 3.407826e-7, 0.e0, 2.550120e-6, -1.241123e-6, 9.901116e-8, 2.210482e-7, 0.e0, 
   -6.351059e-7, 2.341650e-6, 1.061492e-6, 2.878231e-7, 0.e0])
   ccamps = np.reshape(ccamps, (15, 5))
    
   #Constants csec3 and ccsec(n,k) of the secular perturbations in longitude.
   ccsec3 = -7.757020e-8
   ccsec = np.array([1.289600e-6, 5.550147e-1, 2.076942e00, 3.102810e-5, 4.035027e00, 3.525565e-1, 9.124190e-6, 9.990265e-1, 
   2.622706e00, 9.793240e-7, 5.508259e00, 1.559103e01])
   ccsec = np.reshape(ccsec, (4, 3))
    
   #Sidereal rates.
   dcsld = 1.990987e-7                   #sidereal rate in longitude
   ccsgd = 1.990969e-7                   #sidereal rate in mean anomaly

   #Constants used in the calculation of the lunar contribution.
   cckm = 3.122140e-5
   ccmld = 2.661699e-6
   ccfdi = 2.399485e-7
   
   #Constants dcargm(i,k) of the arguments of the perturbations of the motion
   # of the moon.
   dcargm = np.array([5.1679830e0, 8.3286911095275e3, 5.4913150e0, -7.2140632838100e3, 5.9598530e0, 1.5542754389685e4])
   dcargm = np.reshape(dcargm, (3, 2))
    
   #Amplitudes ccampm(n,k) of the perturbations of the moon.
   ccampm = np.array([1.097594e-1, 2.896773e-7, 5.450474e-2, 1.438491e-7, -2.223581e-2, 5.083103e-8, 1.002548e-2, -2.291823e-8, 
   1.148966e-2, 5.658888e-8, 8.249439e-3, 4.063015e-8])
   ccampm = np.reshape(ccampm, (3, 4))
   
   #ccpamv(k)=a*m*dl,dt (planets), dc1mme=1-mass(earth+moon)
   ccpamv = np.array([8.326827e-11, 1.843484e-11, 1.988712e-12, 1.881276e-12])
   dc1mme = 0.99999696e0
   
   #Time arguments.
   dt = (dje - dcto) / dcjul
   tvec = np.array([1e0, dt, dt * dt])
    
   #Values of all elements for the instant(aneous?) dje.
   temp = (np.transpose(np.dot(np.transpose(tvec), np.transpose(dcfel)))) % dc2pi
   dml = temp[0]
   forbel = temp[1:8]
   g = forbel[0]                         #old fortran equivalence
    
   deps = (tvec * dceps).sum() % dc2pi
   sorbel = (np.transpose(np.dot(np.transpose(tvec), np.transpose(ccsel)))) % dc2pi
   e = sorbel[0]                         #old fortran equivalence
  
   #Secular perturbations in longitude.
   dummy = np.cos(2.0)
   sn = np.sin((np.transpose(np.dot(np.transpose(tvec[0:2]), np.transpose(ccsec[:,1:3])))) % cc2pi)
   
   #Periodic perturbations of the emb (earth-moon barycenter).
   pertl = (ccsec[:,0] * sn).sum() + dt * ccsec3 * sn[2]
   pertld = 0.0
   pertr = 0.0
   pertrd = 0.0
   for k in np.arange(0, 15):
       a = (dcargs[k,0] + dt * dcargs[k,1]) % dc2pi
       cosa = np.cos(a)
       sina = np.sin(a)
       pertl = pertl + ccamps[k,0] * cosa + ccamps[k,1] * sina
       pertr = pertr + ccamps[k,2] * cosa + ccamps[k,3] * sina
       if k < 11:   
           pertld = pertld + (ccamps[k,1] * cosa - ccamps[k,0] * sina) * ccamps[k,4]
           pertrd = pertrd + (ccamps[k,3] * cosa - ccamps[k,2] * sina) * ccamps[k,4]
    
   #Elliptic part of the motion of the emb.
   phi = (e * e / 4e0) * (((8e0 / e) - e) * np.sin(g) + 5 * np.sin(2 * g) + (13 / 3e0) * e * np.sin(3 * g))
   f = g + phi
   sinf = np.sin(f)
   cosf = np.cos(f)
   dpsi = (dc1 - e * e) / (dc1 + e * cosf)
   phid = 2 * e * ccsgd * ((1 + 1.5 * e * e) * cosf + e * (1.25 - 0.5 * sinf * sinf))
   psid = ccsgd * e * sinf / np.sqrt(dc1 - e * e)
   
   #Perturbed heliocentric motion of the emb.
   d1pdro = dc1 + pertr
   drd = d1pdro * (psid + dpsi * pertrd)
   drld = d1pdro * dpsi * (dcsld + phid + pertld)
   dtl = (dml + phi + pertl) % dc2pi
   dsinls = np.sin(dtl)
   dcosls = np.cos(dtl)
   dxhd = drd * dcosls - drld * dsinls
   dyhd = drd * dsinls + drld * dcosls
   
   #Influence of eccentricity, evection and variation on the geocentric
   # motion of the moon.
   pertl = 0.0
   pertld = 0.0
   pertp = 0.0
   pertpd = 0.0
   for k in np.arange(0, 3):
      a = (dcargm[k,0] + dt * dcargm[k,1]) % dc2pi
      sina = np.sin(a)
      cosa = np.cos(a)
      pertl = pertl + ccampm[k,0] * sina
      pertld = pertld + ccampm[k,1] * cosa
      pertp = pertp + ccampm[k,2] * cosa
      pertpd = pertpd - ccampm[k,3] * sina
   
   #Heliocentric motion of the earth.
   tl = forbel[1] + pertl
   sinlm = np.sin(tl)
   coslm = np.cos(tl)
   sigma = cckm / (1.0 + pertp)
   a = sigma * (ccmld + pertld)
   b = sigma * pertpd
   dxhd = dxhd + a * sinlm + b * coslm
   dyhd = dyhd - a * coslm + b * sinlm
   dzhd = -sigma * ccfdi * np.cos(forbel[2])
   
   #Barycentric motion of the earth.
   dxbd = dxhd * dc1mme
   dybd = dyhd * dc1mme
   dzbd = dzhd * dc1mme
   for k in np.arange(0, 4):
       plon = forbel[k + 3]
       pomg = sorbel[k + 1]
       pecc = sorbel[k + 9]
       tl = (plon + 2.0 * pecc * np.sin(plon - pomg)) % cc2pi
       dxbd = dxbd + ccpamv[k] * (np.sin(tl) + pecc * np.sin(pomg))
       dybd = dybd - ccpamv[k] * (np.cos(tl) + pecc * np.cos(pomg))
       dzbd = dzbd - ccpamv[k] * sorbel[k + 13] * np.cos(plon - sorbel[k + 5])
   
   #Transition to mean equator of date.
   dcosep = np.cos(deps)
   dsinep = np.sin(deps)
   dyahd = dcosep * dyhd - dsinep * dzhd
   dzahd = dsinep * dyhd + dcosep * dzhd
   dyabd = dcosep * dybd - dsinep * dzbd
   dzabd = dsinep * dybd + dcosep * dzbd
   
   #Epoch of mean equinox (deq) of zero implies that we should use
   # Julian ephemeris date (dje) as epoch of mean equinox.
   if deq == 0:   
       dvelh = au * (np.array([dxhd, dyahd, dzahd]))
       dvelb = au * (np.array([dxbd, dyabd, dzabd]))
       return _ret()
   
   #General precession from epoch dje to deq.
   deqdat = (dje - dcto - dcbes) / dctrop + dc1900
   prema = premat(deqdat, deq, fk4=True)
   
   dvelh = au * (np.transpose(np.dot(np.transpose(prema), np.transpose(np.array([dxhd, dyahd, dzahd])))))
   dvelb = au * (np.transpose(np.dot(np.transpose(prema), np.transpose(np.array([dxbd, dyabd, dzabd])))))
   
   return (dvelh, dvelb)


def premat(equinox1, equinox2, fk4=False):


   deg_to_rad = np.pi / 180.0e0
   sec_to_rad = deg_to_rad / 3600.e0
   
   t = 0.001e0 * (equinox2 - equinox1)
   
   if not fk4:   
      st = 0.001e0 * (equinox1 - 2000.e0)
      #  Compute 3 rotation angles
      a = sec_to_rad * t * (23062.181e0 + st * (139.656e0 + 0.0139e0 * st) + t * (30.188e0 - 0.344e0 * st + 17.998e0 * t))
      
      b = sec_to_rad * t * t * (79.280e0 + 0.410e0 * st + 0.205e0 * t) + a
      
      c = sec_to_rad * t * (20043.109e0 - st * (85.33e0 + 0.217e0 * st) + t * (-42.665e0 - 0.217e0 * st - 41.833e0 * t))
      
   else:   
      
      st = 0.001e0 * (equinox1 - 1900.e0)
      #  Compute 3 rotation angles
      
      a = sec_to_rad * t * (23042.53e0 + st * (139.75e0 + 0.06e0 * st) + t * (30.23e0 - 0.27e0 * st + 18.0e0 * t))
      
      b = sec_to_rad * t * t * (79.27e0 + 0.66e0 * st + 0.32e0 * t) + a
      
      c = sec_to_rad * t * (20046.85e0 - st * (85.33e0 + 0.37e0 * st) + t * (-42.67e0 - 0.37e0 * st - 41.8e0 * t))
      
   
   sina = np.sin(a)
   sinb = np.sin(b)
   sinc = np.sin(c)
   cosa = np.cos(a)
   cosb = np.cos(b)
   cosc = np.cos(c)
   
   r = np.zeros((3, 3))
   r[0,:] = np.array([cosa * cosb * cosc - sina * sinb, sina * cosb + cosa * sinb * cosc, cosa * sinc])
   r[1,:] = np.array([-cosa * sinb - sina * cosb * cosc, cosa * cosb - sina * sinb * cosc, -sina * sinc])
   r[2,:] = np.array([-cosb * sinc, -sinb * sinc, cosc])
   
   return r

def vgeo(obstime, obscoord, vhel=0, epoch=2000):
    
#Convert to julian date from calendar return
    jd=obstime.jd

#Find heliocentric velocity of Earth for given date in km/s  
    vh, vb = baryvel(jd, epoch)

#Get RA and Dec in radians  
    ra_star=obscoord.ra.radian
    dec_star=obscoord.dec.radian
  
    vgeo = vh[0]*np.cos(dec_star)*np.cos(ra_star) +  vh[1]*np.cos(dec_star)*np.sin(ra_star) +   vh[2]*np.sin(dec_star) 

#Add radial heliocentric velocity of star to radial heliocentric velocity of the
#Earth on that date.  The sign of vgeo is negative!

    vgeo=vhel-vgeo

#Note that if vhel=0 this line essentially reverses the sign of what is returned.
#This program ultimately returns a + velocity if the source would appear redshifted,
#or a - velocity if the source would appear blueshifted.

    return vgeo
  

#a program to convert vlsr and vhelio
#Returns vlsr of sun projected onto source position.  
#Add vproj to vhelio to get vlsr of object
#Subtract vproj from vlsr of object to get vhelio
#vlsr_obj=vhel+vpro
#mycoord=SkyCoord(myra, mydec, frame='icrs')
def vproj(mycoord):
    vlsr=20.0*np.array([0.014498, -0.865863, 0.500071])  
    ra_rad=mycoord.ra.radian
    dec_rad=mycoord.dec.radian
    vproj = vlsr[0]*np.cos(dec_rad)*np.cos(ra_rad) + vlsr[1]*np.cos(dec_rad)*np.sin(ra_rad) + vlsr[2]*np.sin(dec_rad)
    return vproj

def vlsr_to_vhelio(mycoord, myvlsr):
    myvproj=vproj(mycoord)
    myvhelio=myvlsr-myvproj
    return myvhelio
  
def vhelio_to_vlsr(mycoord, myvhelio):
    myvproj=vproj(mycoord)
    myvlsr=myvhelio+vproj
    return myvlsr

#Add some error exceptions later
#  if ( N_elements( temp ) NE 1 ) then $
#      read,'Enter a blackbody temperature', temp



