
from __future__ import (absolute_import, division, print_function)
import os
import sys
import threading
import time
import numpy as np
angles = np.arange(180,220+1,4)
ldmFiles = ['Lys_tetra-2h-12min_3_33-4A-%i.ldm'%i for i in angles]
tifFiles = ['Lysozyme_tetra-2h-12min-3_33-4A_%i.tif'%i for i in angles]
mtzFiles = ['Lysozyme_tetra-2h-12min-3_33-4A_%i.mtz1'%i for i in angles]



class ProcessThread ( threading.Thread ):
    command = ""

    def setCommand( self, command="" ):
        self.command = command

    def run ( self ):
        print('STARTING PROCESS: ' + self.command)
        os.system( self.command )


procList = []
index = 0
for angle, ldmFile, tifFile, mtzFile in zip(angles, ldmFiles, tifFiles, mtzFiles):
    procList.append( ProcessThread() )
    tComm = 'python process_single_image.py -a %i -l %s -t %s -m %s'%(angle, ldmFile, tifFile, mtzFile)
    print(tComm)
    procList[index].setCommand(tComm)
    index += 1


max_processes = 30
all_done = False
active_list=[]
while not all_done:
    if  len(procList) > 0 and len(active_list) < max_processes :
        thread = procList[0]
        procList.remove(thread)
        active_list.append( thread )
        thread.start()
    time.sleep(2)
    for thread in active_list:
        if not thread.isAlive():
            active_list.remove( thread )
    if len(procList) == 0 and len(active_list) == 0 :
        all_done = True



