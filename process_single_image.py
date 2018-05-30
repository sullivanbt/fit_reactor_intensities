import profile_fitting_tools as pft
import sys
sys.path.append('home/ntv/integrate/')
import ICCAnalysisTools as ICAT
import numpy as np
import argparse

def main(angle, ldmFile, tifFile, mtzFile):
    print '============ DOING ANGLE %i'%angle
    df, im, yc, xc = pft.analyzeRun(mtzFile, tifFile, ldmFile, 2000, outFileName='lnorm/out_lyso_%i'%angle)
    df.to_pickle('df_%i.pkl'%angle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Does a single TIF file profile fitting')
    parser.add_argument('-a','--angle', action='store', default=0,type=int,help='goniometer angle for image')
    parser.add_argument('-l','--ldmFile', action='store', default=0,type=str,help='path for ldm file')
    parser.add_argument('-t','--tifFile', action='store', default=0,type=str,help='path for tif file')
    parser.add_argument('-m','--mtzFile', action='store', default=0,type=str,help='path for mtz file')
    args = parser.parse_args()
    
    main(args.angle, args.ldmFile, args.tifFile, args.mtzFile)
