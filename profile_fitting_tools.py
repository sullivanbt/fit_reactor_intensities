import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import libtiff
import sys
sys.path.append('/home/ntv/integrate/')
import BVGFitTools as BVGFT
import ICCAnalysisTools as ICAT
from scipy.optimize import curve_fit
import re
import os 
import pandas as pd

def loadRun(mtzFile, tifFile, ldmFile):
    #Parse the ldm to get the center pixels
    with open(ldmFile,'r') as f:
        for line in f:
            if line[:7] == 'X_CEN_F':
                rr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)
                y_c = float(rr[0])-1.
                x_c = float(rr[1])-13
            if line[:4] == 'CTOF':
                CTOF_mm = float(re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)[0])
            if line[:11] == 'RASTER_SIZE':
                RAST_mm = float(re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)[0])/1000.0
    CTOF_pxl = CTOF_mm / RAST_mm
    print 'Center of the image: %4.4f %4.4f'%(x_c, y_c)
    print 'CTOF (mm): %4.4f, Pixel size (mm): %4.4f, CTOF (pxl): %4.4f'%(CTOF_mm, RAST_mm, CTOF_pxl)
    #Now let's read the peaks file.
    #n.b. - we want to make this not dependent on my mtzdump location
    os.system(str('~/laue3/laue/laue_install/bin/mtzdump hklin '+ mtzFile + '< in.txt > /home/ntv/Dropbox\ \(ORNL\)/imagine_test/out.txt'))
    with open('out.txt','r') as f:
        peaksData = f.readlines()
    for i, line in enumerate(peaksData):
        if 'LIST OF REFLECTIONS' in line:
            startLine = i
        elif ' MTZDUMP:' in line:
            endLine = i
    peaksData = peaksData[startLine+2:endLine]
    newPeaksData = []
    for i in np.arange(0,len(peaksData),3):
        l1 = peaksData[i]
        l2 = peaksData[i+1]
        line = np.append(np.array(l1.split(),dtype=float),np.array(l2.split(),dtype=float))
        newPeaksData.append(line)
    newPeaksData = np.array(newPeaksData)
    
    newPeaksData[:,[5,6]] = np.array([y_c, x_c]) - np.array([-4,4*1.00245])*newPeaksData[:,[5,6]]
    peakLocations = newPeaksData[:,[5,6]]
    print 'Found %i peaks'%len(peakLocations)
    print 'Peaks are defined from %4.4f - %4.4f, %4.4f - %4.4f'%(peakLocations[:,0].min(),peakLocations[:,0].max(),
                                                                peakLocations[:,1].min(),peakLocations[:,1].max())
                                                                
    df = pd.DataFrame(newPeaksData, columns=['h','k','l','pack_id','plate',
                    'xf','yf','lambda','I','S','mult','minHarm','maxHarm','novpix','flags'])
                    
    df['sigX'] = np.nan
    df['sigY'] = np.nan
    df['sigP'] = np.nan
    df['scale'] = np.nan
    df['bgConst'] = np.nan
    df['muX'] = np.nan
    df['muY'] = np.nan
    df['IFit'] = np.nan
    df['SFit'] = np.nan
    df['redChiSq'] = np.nan
    imFile = libtiff.TIFF.open(tifFile)
    im = imFile.read_image()
    im = np.flipud(im)
    
    print 'Doing geometry calculations...'
    getGeometryColumns(df,im,y_c,x_c,CTOF_pxl)
    print 'Done!'
    
    return df, im, x_c, y_c
    
def displayPeaks(df, im,figNum=3,xlim=None, ylim=None, clim=None):
    plt.figure(figNum); plt.clf()
    if clim is None:
        plt.imshow(im,cmap='gray_r')
    else:
        plt.imshow(im,cmap='gray_r',vmin=np.min(clim), vmax=np.max(clim))
    x,y = df['xf'].values, df['yf'].values
    plt.plot(x,y,'*',ms=1)
    if xlim is not None and ylim is not None:
        plt.xlim([np.min(xlim), np.max(xlim)])
        plt.ylim([np.max(ylim), np.min(ylim)])

def getSinglePeakImage(df, im, peakID, dx=20, dy=20):
    py = int(df.iloc[peakID]['xf'])
    px = int(df.iloc[peakID]['yf'])
    ssim = im[px-dx:px+dx,py-dy:py+dy]
    return ssim
    
def fitSinglePeakImage(ssim,A=None, sigX=3.0, sigY=2.0, rho=0.0, bgConst=None):
    if bgConst is None:
        bgConst = np.median(ssim)
    if A is None:
        A = np.max(ssim) - bgConst; 

    mu = np.array(ssim.shape,dtype=int)/2
    sigma = np.array([[sigX**2,sigX*sigY*rho],[sigX*sigY*rho,sigY*sigY]])

    x = np.array(range(ssim.shape[0]))
    y = np.array(range(ssim.shape[1]))
    X,Y = np.meshgrid(x,y,indexing='ij')

    p0 = np.array([A, mu[0], mu[1], sigX, sigY, rho, bgConst])
    bounds = [  [0., p0[1]-3, p0[2]-5, 0.5*p0[3], 0.5*p0[4], -1., 0.],
              [np.inf, p0[1]+3, p0[2]+5, 1.5*p0[3], 1.5*p0[4], 1., np.inf] ]
    fitIDX = np.ones_like(X).astype(np.bool)
    
    params, cov = curve_fit(BVGFT.bvgFitFun, [X[fitIDX], Y[fitIDX]], ssim[fitIDX],p0=p0, bounds=bounds,sigma=np.sqrt(ssim[fitIDX]),maxfev=4000)
    
    A = params[0]
    mu = np.array([params[1], params[2]])
    sigX = params[3]; sigY = params[4]; rho = params[5]
    sigma = np.array([[sigX**2,sigX*sigY*rho],[sigX*sigY*rho,sigY*sigY]])
    bgConst = params[6]
    x = np.array(range(ssim.shape[0]))
    y = np.array(range(ssim.shape[1]))
    X,Y = np.meshgrid(x,y,indexing='ij')
    zFit = BVGFT.bvg(A, mu, sigma, X, Y, 0)
    paramsDict = {'scale':params[0],'muX':params[1], 'muY':params[2],
                'sigX':params[3], 'sigY':params[4], 'sigP':params[5], 'bgConst':params[6]}

    chiSq = np.sum( (zFit+bgConst-ssim)**2 / ssim)
    df = np.prod(ssim.size) - len(p0)
    redChiSq = chiSq/df
    return zFit, paramsDict, redChiSq
    
def integrateModel(zModel, bgConstant, ssim, threshold=0.05):
    goodIDX = zModel/zModel.max() > threshold
    I = np.sum(zModel[goodIDX])/10
    bgCounts = bgConstant*np.sum(goodIDX)
    nObs = ssim[goodIDX].sum()
    varFit = np.average(((zModel+bgConstant)[goodIDX]-ssim[goodIDX])**2, weights=ssim[goodIDX])
    S = np.sqrt(nObs + bgCounts + varFit )

    return I, S
    
def saveFitResults(df, peakID, params):
    for key in params.keys():
        df.loc[peakID,key] = params[key]
        #df.at(peakID)[key] = params[key]
        
def doPeakFit(df, im, peakID, plotResults=False):
    try:
        ssim = getSinglePeakImage(df, im, peakID)
        zModel, params, redChiSq = fitSinglePeakImage(ssim)
        zFit = zModel + params['bgConst']
        I,S = integrateModel(zModel, params['bgConst'], ssim)
        params['IFit'] = I
        params['SFit'] = S
        params['redChiSq'] = redChiSq
        saveFitResults(df,peakID,params)
        if plotResults:
            plt.clf()
            plt.subplot(1,2,1)
            plt.imshow(ssim,vmin=0, vmax=0.8*ssim.max())
            plt.subplot(1,2,2)
            plt.imshow(zFit,vmin=0, vmax=0.8*ssim.max())
            plt.figure()
            plt.plot(np.sum(ssim,axis=0),label='Image')
            plt.plot(np.sum(zFit,axis=0),label='Fit')
            plt.plot(np.sum(zModel,axis=0),label='Peak Model')
            plt.legend(loc='best')
    except KeyboardInterrupt: sys.exit(0)
    except:
        raise 
        pass
    
def plotFromParams(df, im, peakID,dx=20, dy=20):
    ssim = getSinglePeakImage(df, im, peakID, dx=dx, dy=dy)
    x = np.array(range(ssim.shape[0]))
    y = np.array(range(ssim.shape[1]))
    X,Y = np.meshgrid(x,y,indexing='ij')
    A = df.iloc[peakID]['scale']
    muX = df.iloc[peakID]['muX']
    muY = df.iloc[peakID]['muY']
    sigX = df.iloc[peakID]['sigX']
    sigY = df.iloc[peakID]['sigY']
    sigP = df.iloc[peakID]['sigP']
    bgConst = df.iloc[peakID]['bgConst']

    print A

    mu = [muX, muY]
    sigma = np.array([[sigX**2,sigX*sigY*sigP],[sigX*sigY*sigP,sigY*sigY]])

    zFit = BVGFT.bvg(A, mu, sigma, X, Y, bgConst)
    plt.figure()
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(ssim,vmin=0, vmax=0.8*ssim.max())
    plt.subplot(1,2,2)
    plt.imshow(zFit,vmin=0, vmax=0.8*ssim.max())
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(np.sum(ssim,axis=0),label='Image')
    plt.plot(np.sum(zFit,axis=0),label='Fit')
    plt.subplot(1,2,2)
    plt.plot(np.sum(ssim,axis=1),label='Image')
    plt.plot(np.sum(zFit,axis=1),label='Fit')
    plt.legend(loc='best')
    
    
def getNNIDX(df,intensCol,threshold, plotResults=False):
    strongIDX = df[intensCol]>threshold
    weakIDX = ~strongIDX
    a = np.array(df[strongIDX][['xf','yf']])
    c = np.array(df[strongIDX].index.values)
    nnIDX = -1*np.ones_like(np.array(df[intensCol]))
    dR = np.zeros_like(np.array(df[intensCol]))
    aW = np.array(df[['xf','yf']])
    for i, v in enumerate(aW):
        if weakIDX[i]:
            dR[i] = np.min(np.linalg.norm(a-v,axis=1))
            nnIDX[i] = c[np.argmin(np.linalg.norm(a-v,axis=1))]
    df['nnIDX'] = nnIDX.astype('int')
    df['dR'] = dR
    if plotResults:
        plt.figure(2); plt.clf()
        df[df['dR']>0]['dR'].hist(bins=100)
        plt.xlabel('Distance to nearest peak (pxl)')
        plt.ylabel('# Peaks')
    print 'Found nearest neighbors for %i peaks with %s < %4.4f'%(weakIDX.sum(), intensCol, threshold)
    

def fitWeakPeak(df, im, peakIDX, nnIDX, plotResults=False):
    #Let's start with just ampltidue and BG
    ssim = getSinglePeakImage(df, im, peakIDX)
    x = np.array(range(ssim.shape[0]))
    y = np.array(range(ssim.shape[1]))
    X,Y = np.meshgrid(x,y,indexing='ij')
    bg0 = np.median(ssim)
    A0 = np.max(ssim) - bg0
    p0 = np.array([A0, bg0])
    sigX, sigY, sigP, muX, muY = df.iloc[nnIDX][['sigX', 'sigY', 'sigP','muX', 'muY']]
    mu = np.array([muX, muY])
    sigma = np.array([[sigX**2,sigX*sigY*sigP],[sigX*sigY*sigP,sigY*sigY]])
    fitIDX = np.ones_like(ssim,dtype=np.bool)
    A = A0
    zFit = BVGFT.bvg(A0, mu, sigma, X, Y, 0.)

    def scalePeak(x,A,bg):
        return A*x+bg

    params, cov = curve_fit(scalePeak, zFit[fitIDX], ssim[fitIDX],p0=p0)
    paramsDict = {'scale':params[0]*A0,'muX':df.iloc[nnIDX]['muX'], 'muY':df.iloc[nnIDX]['muY'],
                'sigX':df.iloc[nnIDX]['sigX'], 'sigY':df.iloc[nnIDX]['sigY'], 'sigP':df.iloc[nnIDX]['sigP'],
                 'bgConst':params[1]}


    bgConst=params[1]
    zModel = params[0]*zFit
    zFit = zModel + bgConst
    if plotResults:
        plt.figure(4); plt.clf()
        plt.subplot(1,3,1)
        plt.imshow(ssim,vmin=0, vmax=ssim.max(),cmap='gray_r')
        plt.subplot(1,3,2)
        plt.imshow(zFit,vmin=0, vmax=ssim.max(),cmap='gray_r')
        plt.subplot(1,3,3)
        plt.imshow(getSinglePeakImage(df, im, nnIDX),vmin=0, vmax=ssim.max(),cmap='gray_r')
        plt.figure(5); plt.clf()
        plt.subplot(1,2,1)
        plt.plot(ssim.sum(axis=0))
        plt.plot(zFit.sum(axis=0))
        plt.subplot(1,2,2)
        plt.plot(ssim.sum(axis=1))
        plt.plot(zFit.sum(axis=1))

    chiSq = np.sum( (zModel+bgConst-ssim)**2 / ssim)
    df = np.prod(ssim.size) - len(p0)
    redChiSq = chiSq/df

    return zFit-params[1], paramsDict, redChiSq
    
def doWeakPeakFit(df, im, peakIDX, plotResults=False):
    ssim = getSinglePeakImage(df, im, peakIDX)
    nnIDX = int(df.iloc[peakIDX]['nnIDX'])
    zModel, params, redChiSq = fitWeakPeak(df, im, peakIDX, nnIDX, plotResults=plotResults)
    zFit = zModel + params['bgConst']
    I,S = integrateModel(zModel, params['bgConst'], ssim)
    params['IFit'] = I
    params['SFit'] = S
    params['redChiSq'] = redChiSq
    saveFitResults(df,peakIDX,params)

def getPlanarPhi(im, xc, returnDegrees=False):
    '''
    Given a detector image of size ny*nx from a cylindrical detector,
    this function will return a mapping between x (pixel number in the
    cylindrical direction) and phi.  
    Input: 
        im: 2D array of size [ny, nx]
        xc: pixel number for phi=0 (between 0 and nx-1)
    Output:
        phi: 1D array of length nx containing thevalue of phi
             (in radians unless otherwise specified) on range [0, 2*pi]
    '''
    nx = im.shape[1]
    phi = np.zeros(nx)
    tmp = np.linspace(360,0,nx)
    nRight = im.shape[1] - xc
    nLeft = im.shape[1]-nRight
    phi = np.append(tmp[int(nRight):],tmp[:int(nRight)])
    phi = phi
    if returnDegrees:
        return phi
    else: #Radians
        return phi/180.0*np.pi
        
def getPiecewisePhi(im, xc, returnDegrees=False):
    nx = im.shape[1]
    xMin = xc
    xMax = xc - nx//2
    phi = getPlanarPhi(im, xc, returnDegrees=returnDegrees)

    lowX = np.min([xMin, xMax])
    hiX = np.max([xMin, xMax])
    x = np.array(range(nx),dtype=float)

    p0 = np.polyfit(x[x<lowX], phi[x<lowX], 1)
    p1 = np.polyfit(x[np.logical_and(x>=lowX, x<=hiX)], phi[np.logical_and(x>=lowX, x<=hiX)], 1)
    p2 = np.polyfit(x[x>hiX], phi[x>hiX], 1)
    phiFun = lambda x: np.piecewise(x, [x<lowX, np.logical_and(x>=lowX, x<=hiX), x>hiX],
                    [lambda x: np.polyval(p0,x), lambda x: np.polyval(p1,x), lambda x: np.polyval(p2,x)])
    return phiFun
    
def getGeometryColumns(df, im, xc, yc, ctof):
    phiFun = getPiecewisePhi(im, xc)
    df['phiPlanar'] = df['xf'].apply(phiFun)
    df['r'] = ctof
    df['xReal'] = df['r']*np.sin(df['phiPlanar'])
    df['yReal'] = yc - df['yf']
    df['zReal'] = df['r']*np.cos(df['phiPlanar'])
    v1 = np.array(df[['xReal','yReal','zReal']])
    df['Scattering'] = np.arccos(np.dot(v1, [0,0,1])/np.linalg.norm(v1,axis=1))
    df['2theta'] = df['Scattering']
    df['phi'] = np.arctan2(df['zReal'],np.hypot(df['xReal'],df['yReal']))
    
def saveForLaueNorm(df,outFileName, hCol='h', kCol='k', lCol='l', lambdaCol='lambda', twoThCol='2theta', 
                    ICol = 'IFit', SCol = 'SFit'):
    outIDX = ~np.isnan(df['I'])
    print 'Saving lauenorm format to %s'%outFileName
    numnans = 0; numWritten = 0;
    with open(outFileName,'w') as f:
        for idd, srow in enumerate(df.iterrows()):
            row=srow[1]
            try:
                f.write('%5i%5i%5i%10.2f%10.2f%10i%10i\n'%(row[hCol], row[kCol], row[lCol], row[lambdaCol], 0.5*row[twoThCol],
                                                           row[ICol], row[SCol]))
                numWritten += 1
            except:
                if np.isnan(row['IFit']):
                    numnans += 1
                else:
                    raise
    print 'File saved correctly.  %i peaks written, %i peaks were NaN, %i peaks total'%(numWritten, numnans, len(df))
    
def analyzeRun(mtzFile, tifFile, ldmFile, ICutoff, outFileName='out.txt'):

    #Load the run
    df, im, yc, xc = loadRun(mtzFile, tifFile, ldmFile)
    peaksToFit = df[df['I']>ICutoff].index.values

    #Fit the strong peaks
    print 'This will fit %i strong peaks (%4.2f Percent)'%(len(peaksToFit), 100.0*len(peaksToFit)/len(df))
    for i, peakID in enumerate(peaksToFit):
        ICAT.print_progress(i,len(peaksToFit),prefix='Fitting Profiles: ',suffix='Complete')
        doPeakFit(df,im,peakID,plotResults=False)
    print '\nFit %i peaks with I>%4.4f'%(len(peaksToFit), ICutoff)

    # Get the NN index
    getNNIDX(df, 'I',ICutoff, plotResults=False)
    
    #Fit the weak peaks
    print 'Will fit %i peaks now using nearest neighbors:'%len(df)
    for i in range(len(df)):
        if df.iloc[i]['I']<ICutoff:
            ICAT.print_progress(i,len(df),prefix='Fitting Profiles: ',suffix='Complete')
            try:
                doWeakPeakFit(df, im, i)
            except KeyboardInterrupt:
                sys.exit(0)
            except: 
                raise 
                pass
    #Save the output
    saveForLaueNorm(df, outFileName=outFileName)
    return df, im, yc, xc
    
