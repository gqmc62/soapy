#Copyright Durham University and Andrew Reeves
#2014

# This file is part of soapy.

#     soapy is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     soapy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with soapy.  If not, see <http://www.gnu.org/licenses/>.

import traceback
import time

import numpy
from matplotlib import pyplot as plt
from astropy.io import fits

from . import logger

from . import interp
from scipy.signal import convolve2d
from scipy.optimize import curve_fit
import os

def my_unwrap(wrapped_phase, period=2*numpy.pi):
    
    # numpy unwrap start unwrap at 0 coordinate
    # but for circular aperture there is no corner!
    # if we unwrap by each quardrant there will be a corner! noice.
    # but we may have to stitch them back together nicely
    # so things should be de-piston to the center
    
    center = wrapped_phase.shape[-1]//2
    unwrapped_phase = numpy.zeros_like(wrapped_phase)
    
    # ++ quardrant
    wrapped_quardrant = wrapped_phase[...,center:,center:]
    unwrapped_quardrant = numpy.unwrap(numpy.unwrap(wrapped_quardrant,axis=0),axis=1)
      
    unwrapped_quardrant -= unwrapped_quardrant[0,0]
    unwrapped_phase[...,center:,center:] = unwrapped_quardrant
    
    # +- quardrant
    wrapped_quardrant = wrapped_phase[...,center:,:center][:,::-1]
    unwrapped_quardrant = numpy.unwrap(numpy.unwrap(wrapped_quardrant,axis=0),axis=1)
      
    unwrapped_quardrant -= unwrapped_quardrant[0,0]
    unwrapped_phase[...,center:,:center] = unwrapped_quardrant[:,::-1]
    
    # -+ quardrant
    wrapped_quardrant = wrapped_phase[...,:center,center:][::-1,:]
    unwrapped_quardrant = numpy.unwrap(numpy.unwrap(wrapped_quardrant,axis=0),axis=1)
      
    unwrapped_quardrant -= unwrapped_quardrant[0,0]
    unwrapped_phase[...,:center,center:] = unwrapped_quardrant[::-1,:]
      
    # -- quardrant
    wrapped_quardrant = wrapped_phase[...,:center,:center][::-1,::-1]
    unwrapped_quardrant = numpy.unwrap(numpy.unwrap(wrapped_quardrant,axis=0),axis=1)
      
    unwrapped_quardrant -= unwrapped_quardrant[0,0]
    unwrapped_phase[...,:center,:center] = unwrapped_quardrant[::-1,::-1]
    
    return unwrapped_phase

def crop_inscribed_square(circular_data):
    N = circular_data.shape[0]
    R = int(numpy.floor(N//2))
    r = int(numpy.floor(R/numpy.sqrt(2)))
    square_data = circular_data[N//2 - r
                  : N//2 + r,
                  N//2 - r
                  : N//2 + r]
    return square_data

# Use pyfits or astropy for fits file handling
try:
    from astropy.io import fits
except ImportError:
    try:
        import pyfits as fits
    except ImportError:
        raise ImportError("soapy requires either pyfits or astropy")

#xrange now just "range" in python3.
#Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range



def bin_this(A, w):
    # a = numpy.nanmean(A[:(A.shape[0]//w)*w,
    #       :(A.shape[1]//w)*w].reshape(
    #           (((A.shape[0]//w),w,((A.shape[1]//w)),w))
    #           ), axis=(1,3))
    a = numpy.mean(A[:(A.shape[0]//w)*w,
          :(A.shape[1]//w)*w].reshape(
              (((A.shape[0]//w),w,((A.shape[1]//w)),w))
              ), axis=(1,3))
    return a

class Reconstructor(object):
    """
    Reconstructor that will give DM commands required to correct an AO frame for a given set of WFS measurements
    """
    def __init__(self, soapy_config, dms, wfss, atmos, runWfsFunc=None):

        self.soapy_config = soapy_config

        self.dms = dms
        self.wfss = wfss
        self.sim_config = soapy_config.sim
        self.atmos = atmos
        self.config = soapy_config.recon

        self.n_dms = soapy_config.sim.nDM
        self.scrn_size = soapy_config.sim.scrnSize

        self.learnIters = self.sim_config.learnIters

        self.dmActs = []
        self.dmConds = []
        self.dmTypes = []
        for dm in xrange(self.sim_config.nDM):
            self.dmActs.append(self.dms[dm].dmConfig.nxActuators)
            self.dmConds.append(self.dms[dm].dmConfig.svdConditioning)
            self.dmTypes.append(self.dms[dm].dmConfig.type)

        self.dmConds = numpy.array(self.dmConds)
        self.dmActs = numpy.array(self.dmActs)

        n_acts = 0
        self.first_acts = []
        for i, dm in self.dms.items():
            self.first_acts.append(n_acts)
            n_acts += dm.n_acts

        n_wfs_measurements = 0
        self.first_measurements = []
        for i, wfs in self.wfss.items():
            self.first_measurements.append(n_wfs_measurements)
            n_wfs_measurements += wfs.n_measurements

        #2 functions used in case reconstructor requires more WFS data.
        #i.e. learn and apply
        self.runWfs = runWfsFunc
        if self.sim_config.learnAtmos == "random":
            self.moveScrns = atmos.randomScrns
        else:
            self.moveScrns = atmos.moveScrns
        self.wfss = wfss

        self.control_matrix = numpy.zeros(
            (self.sim_config.totalWfsData, self.sim_config.totalActs))
        self.controlShape = (
            self.sim_config.totalWfsData, self.sim_config.totalActs)

        self.Trecon = 0

        self.find_closed_actuators()

        self.actuator_values = None

    def find_closed_actuators(self):
        self.closed_actuators = numpy.zeros(self.sim_config.totalActs)
        n_act = 0
        for i_dm, dm in self.dms.items():
            if dm.dmConfig.closed:
                self.closed_actuators[n_act: n_act + dm.n_acts] = 1
            n_act += dm.n_acts

    def saveCMat(self):
        """
        Writes the current control Matrix to FITS file
        """
        filename = self.sim_config.simName+"/cMat.fits"

        # fits.writeto(
        #         filename, self.control_matrix,
        #         header=self.sim_config.saveHeader, overwrite=True)

    def loadCMat(self,cmat_to_load=None):
        """
        Loads a control matrix from file to the reconstructor

        Looks in the standard reconstructor directory for a control matrix and loads the file.
        Also looks at the FITS header and checks that the control matrix is compatible with the current simulation.
        """

        filename = self.sim_config.simName+"/cMat.fits"

        logger.info("Load Command Matrix")

        cMatHDU = fits.open(filename)[0]
        cMatHDU.verify("fix")
        header = cMatHDU.header
        
        if cmat_to_load is None:
        
            try:
                dmNo = int(header["NBDM"])
                exec("dmActs = numpy.array({})".format(
                        cMatHDU.header["DMACTS"]), globals())
                exec("dmTypes = %s" % header["DMTYPE"], globals())
                exec("dmConds = numpy.array({})".format(
                        cMatHDU.header["DMCOND"]), globals())
    
                if not numpy.allclose(dmConds, self.dmConds):
                    raise IOError("DM conditioning Parameter changed - will make new control matrix")
                if not numpy.all(dmActs==self.dmActs) or dmTypes!=self.dmTypes or dmNo!=dmNo:
                    logger.warning("loaded control matrix may not be compatibile with \
                                    the current simulation. Will try anyway....")
    
                cMat = cMatHDU.data
    
            except KeyError:
                logger.warning("loaded control matrix header has not created by this ao sim. Will load anyway.....")
                #cMat = cMatFile[1]
                cMat = cMatHDU.data
    
            if cMat.shape != self.controlShape:
                logger.warning("designated control matrix does not match the expected shape")
                raise IOError
            else:
                self.control_matrix = cMat
                
        elif cmat_to_load is not None:
            self.control_matrix = cmat_to_load

    def save_interaction_matrix(self):
        """
        Writes the current control Matrix to FITS file

        Writes the current interaction matrix to a FITS file in the simulation directory. Also
        writes the "valid actuators" as accompanying FITS files, and potentially premade DM
        influence functions.
        """
        imat_filename = self.sim_config.simName+"/iMat.fits"

        fits.writeto(
                imat_filename, self.interaction_matrix,
                header=self.sim_config.saveHeader, overwrite=True)

        for i in range(self.n_dms):
            valid_acts_filename =  self.sim_config.simName+"/active_acts_dm{}.fits".format(i)
            valid_acts = self.dms[i].valid_actuators
            fits.writeto(valid_acts_filename, valid_acts, header=self.sim_config.saveHeader, overwrite=True)

            # If DM has pre made influence funcs, save them too
            try:
                dm_shapes_filename = self.sim_config.simName + "/dmShapes_dm{}.fits".format(i)
                fits.writeto(
                        dm_shapes_filename, self.dms[i].iMatShapes,
                        header=self.simConfig.saveHeader, overwrite=True)
            # If not, don't worry about it! Must be a DM with no pre-made influence funcs
            except AttributeError:
                pass
                
    def load_interaction_matrix(self,imat_to_load=None):
        """
        Loads the interaction matrix from file

        AO interaction matrices can get very big, so its useful to be able to load it frmo file
        rather than make it new everytime. It is assumed that the iMat is saved as "FITS" in the
        simulation saved directory, with other accompanying FITS files that contain the indices
        of actuators which are "valid". Some DMs also use pre made influence functions, which are
        also loaded here.
        """
        
        if imat_to_load is None:
        
            filename = self.sim_config.simName+"/iMat.fits"
    
            imat_header = fits.getheader(filename)
            imat_data = fits.getdata(filename)
    
            imat_totalActs = imat_header['DMNACTU']
            imat_totalWfsData = imat_header['NSLOP']
            
            # Check iMat generated with same totalActs and totalWfsData as current sim
            # NOTE the actual shape of the loaded iMat can be different due to invalid acts
            if imat_totalActs != self.sim_config.totalActs or imat_totalWfsData != self.sim_config.totalWfsData:
                logger.warning(
                    "interaction matrix not generated with same number of actuators/wfs slopes"
                )
                raise IOError("interaction matrix does not match required required size.")
    
            # Load valid actuators
            n_total_valid_acts = 0
            for i in range(self.n_dms):
                valid_acts_filename =  self.sim_config.simName+"/active_acts_dm{}.fits".format(i)
                valid_acts = fits.getdata(valid_acts_filename)
                self.dms[i].valid_actuators = valid_acts
                n_total_valid_acts += self.dms[i].n_valid_actuators
    
                # DM may also have preloaded influence functions
                try:
                    dm_shapes_filename = self.sim_config.simName + "/dmShapes_dm{}.fits".format(i)
                    dm_shapes = fits.getdata(dm_shapes_filename)
                    self.dms[i].iMatShapes = dm_shapes
    
                except IOError:
                    # Found no DM influence funcs
                    logger.info("DM Influence functions not found. If the DM doesn't use them, this is ok. If not, set 'forceNew=True' when making IMat")
    
            # Final check of loaded iMat
            if imat_data.shape != (n_total_valid_acts, self.sim_config.totalWfsData):
                logger.warning(
                    "interaction matrix does not match required required size."
                )
                raise IOError("interaction matrix does not match required required size.")
    
            self.interaction_matrix = imat_data
            
        elif imat_to_load is not None:
            imat_data = imat_to_load
            imat_totalActs = imat_data.shape[0]
            imat_totalWfsData = imat_data.shape[1]
            # print(self.n_dms)
            for i in range(self.n_dms):
                # print(i)
                # print(self.soapy_config.dms[i].type)
                if not (self.soapy_config.dms[i].type == 'Aberration'):
                    # print('1')
                    # print(self.dms[i].n_acts)
                    # print('1.5')
                    self.dms[i].valid_actuators = numpy.ones((self.dms[i].n_acts), dtype="int")
                    # print('2')
                    self.interaction_matrix = imat_data

    def makeIMat(self, callback=None):

        self.interaction_matrix = numpy.zeros((self.sim_config.totalActs, self.sim_config.totalWfsData))

        n_acts = 0
        dm_imats = []
        total_valid_actuators = 0
        for dm_n, dm in self.dms.items():
            logger.info("Creating Interaction Matrix for DM %d " % (dm_n))
            if dm.config.type != 'Aberration':
                dm_imats.append(self.make_dm_iMat(dm, callback=callback))
    
                total_valid_actuators += dm_imats[dm_n].shape[0]

        self.interaction_matrix = numpy.zeros((total_valid_actuators, self.sim_config.totalWfsData))
        act_n = 0
        for imat in dm_imats:
            self.interaction_matrix[act_n: act_n + imat.shape[0]] = imat
            act_n += imat.shape[0]
            
        # plt.imshow(self.interaction_matrix)
        # plt.title('interaction matrix')
        # plt.show()


    def make_dm_iMat(self, dm, callback=None):
        """
        Makes an interaction matrix for a given DM with a given WFS

        Parameters:
            dm (DM): The Soapy DM for which an interaction matri is required.
            wfs (WFS): The Soapy WFS for which an interaction matrix is required
            callback (func, optional): A function to be called each iteration accepting no arguments
        """

        iMat = numpy.zeros(
            (dm.n_acts, self.sim_config.totalWfsData))

        # A vector of DM commands to use when making the iMat
        actCommands = numpy.zeros(dm.n_acts)

        phase = numpy.zeros((self.n_dms, self.scrn_size, self.scrn_size))
        
        
        ABERRATION = False
        
        # zero poke case
        
        zero_iMat = numpy.zeros((self.sim_config.totalWfsData))
        zero_iMat_intensity = numpy.zeros((self.sim_config.totalWfsData),dtype=float)
        zero_wfs_efield = []
        
        # Set vector of iMat commands and phase to 0
        actCommands[:] = 0
        # Now get a DM shape for that command
        phase[:] = 0
        phase[dm.n_dm] = dm.dmFrame(actCommands)
        for DM_N, DM in self.dms.items():
            if DM.config.type == 'Aberration':
                ABERRATION = True
                if DM.config.calibrate == False:
                    phase[DM_N] = DM.dmFrame('flat')
                else:
                    phase[DM_N] = DM.dmFrame('shape')
                    # print('yes')
        # Send the DM shape off to the relavent WFS. put result in iMat
        n_wfs_measurments = 0
        for wfs_n, wfs in self.wfss.items():
            # turn off wfs noise if set
            if self.config.imat_noise is False:
                wfs_pnoise = wfs.config.photonNoise
                wfs.config.photonNoise = False
                wfs_rnoise = wfs.config.eReadNoise
                wfs.config.eReadNoise = 0
            
            zero_iMat[n_wfs_measurments: n_wfs_measurments+wfs.n_measurements] = (
                    wfs.frame(None, phase_correction=phase, iMatFrame=True))# / (dm.dmConfig.iMatValue     *1e-9 )      
            zero_iMat_intensity[n_wfs_measurments: n_wfs_measurments+wfs.n_measurements] = (
                numpy.tile(wfs.centSubapArrays.sum(-1).sum(-1),2))
            if self.soapy_config.recon.analyseAberationCalib:
                zero_wfs_efield.append(numpy.copy(wfs.interp_efield))
                # plt.imshow(numpy.angle(zero_wfs_efield[wfs_n]))
                # plt.show()
                # plt.imshow(numpy.abs(zero_wfs_efield[wfs_n])**2)
                # plt.show()
            
        # plt.imshow(phase.sum(0))
        # plt.show()
        # plt.plot(zero_iMat)
        # plt.show()
        
        self.zero_iMat = numpy.copy(zero_iMat)
        self.zero_iMat_intensity = numpy.copy(zero_iMat_intensity)
        
        # poke each dms
        
        if self.soapy_config.recon.analyseAberationCalib:
        
            setattr(dm,'subapShift_influence', numpy.zeros((dm.n_acts,len(self.wfss),2)))
            dm.subapShift_influence[:] = numpy.nan
            setattr(dm,'subapShift_slope', numpy.zeros((dm.n_acts,len(self.wfss),2)))
            dm.subapShift_slope[:] = numpy.nan
            setattr(dm,'rytov', numpy.zeros((dm.n_acts,len(self.wfss))))
            dm.rytov[:] = numpy.nan
        
        # print(dm.n_acts)
        if dm.n_acts == 81:
            # FULL_ACTS = numpy.array([13,
            #                          20,21,22,23,24,
            #                          29,30,31,32,33,
            #                          37,38,39,40,41,42,43,
            #                          47,48,49,50,51,
            #                          56,57,58,59,60,
            #                          67],dtype=int)
            FULL_ACTS = numpy.array([22,
                                     30,31,32,
                                     38,39,40,41,42,
                                     48,49,50,
                                     58],dtype=int)
            
            position = numpy.array([[0, 3],[0, 4],
                                 [1, 1],[1, 2],[1, 3],[1, 4],[1, 5],[1, 6],
                                 [2, 1],[2, 2],[2, 3],[2, 4],[2, 5],[2, 6],
                                 [3, 0],[3, 1],[3, 2],[3, 3],[3, 4],[3, 5],[3, 6],[3, 7],
                                 [4, 0],[4, 1],[4, 2],[4, 3],[4, 4],[4, 5],[4, 6],[4, 7],
                                 [5, 1],[5, 2],[5, 3],[5, 4],[5, 5],[5, 6],
                                 [6, 1],[6, 2],[6, 3],[6, 4],[6, 5],[6, 6],
                                 [7, 3],[7, 4]], dtype=int) # wfs
            
            # ACT_LIST = numpy.array([24,
            #                      32,33,34,
            #                      40,41,42,43,44,
            #                      50,51,52,
            #                      60]) # dm act number
            
            ACT_LIST = numpy.array([13,
                                 20,21,22,23,24,
                                 29,30,31,32,33,
                                 37,38,39,40,41,42,43,
                                 47,48,49,50,51,
                                 56,57,58,59,60,
                                 67])
            
            DM_POS_TEMP = numpy.array([[0.5,3.5],
                                    [1.5,1.5],[1.5,2.5],[1.5,3.5],[1.5,4.5],[1.5,5.5],
                                    [2.5,1.5],[2.5,2.5],[2.5,3.5],[2.5,4.5],[2.5,5.5],
                                    [3.5,0.5],[3.5,1.5],[3.5,2.5],[3.5,3.5],[3.5,4.5],[3.5,5.5],[3.5,6.5],
                                    [4.5,1.5],[4.5,2.5],[4.5,3.5],[4.5,4.5],[4.5,5.5],
                                    [5.5,1.5],[5.5,2.5],[5.5,3.5],[5.5,4.5],[5.5,5.5],
                                    [6.5,3.5]])
        elif dm.n_acts == 25:
            # FULL_ACTS = numpy.array([7, 11,12,13, 17],dtype=int)
            FULL_ACTS = numpy.array([12],dtype=int) # use only dm with 4 active subaps next to it and another 8 next to them
            # xx
            #xxxx
            #xxxx
            # xx
            position = numpy.array([[0, 1],[0, 2],
                                 [1, 0],[1, 1],[1, 2],[1, 3],
                                 [2, 0],[2, 1],[2, 2],[2, 3],
                                 [3, 1],[3, 2]], dtype=int) # from wfs based on detector_cent_coords
            
            ACT_LIST = numpy.array([7,
                                 11,12,13,
                                 17]) # dm act number # count from all dm acts # list only those between 4 wfs subap
            
            DM_POS_TEMP = numpy.array([[0.5,1.5],
                                   [1.5,0.5],[1.5,1.5],[1.5,2.5],
                                   [2.5,1.5]]) # similar to the above buth with the coordinate
        elif dm.n_acts == 121:
            FULL_ACTS = numpy.array([28,
                                     37,38,39,40,41,
                                     48,49,50,51,52,
                                     58,59,60,61,62,63,64,
                                     70,71,72,73,74,
                                     81,82,83,84,85,
                                     94],dtype=int) - 1
            position = numpy.array([[0., 4.],[0., 5.],
                                     [1., 2.],[1., 3.],[1., 4.],[1., 5.],[1., 6.],[1., 7.],
                                     [2., 1.],[2., 2.],[2., 3.],[2., 4.],[2., 5.],[2., 6.],[2., 7.],[2., 8.],
                                     [3., 1.],[3., 2.],[3., 3.],[3., 4.],[3., 5.],[3., 6.],[3., 7.],[3., 8.],
                                     [4., 0.],[4., 1.],[4., 2.],[4., 3.],[4., 4.],[4., 5.],[4., 6.],[4., 7.],[4., 8.],[4., 9.],
                                     [5., 0.],[5., 1.],[5., 2.],[5., 3.],[5., 4.],[5., 5.],[5., 6.],[5., 7.],[5., 8.],[5., 9.],
                                     [6., 1.],[6., 2.],[6., 3.],[6., 4.],[6., 5.],[6., 6.],[6., 7.],[6., 8.],
                                     [7., 1.],[7., 2.],[7., 3.],[7., 4.],[7., 5.],[7., 6.],[7., 7.],[7., 8.],
                                     [8., 2.],[8., 3.],[8., 4.],[8., 5.],[8., 6.],[8., 7.],
                                     [9., 4.],[9., 5.]],dtype=int)
            ACT_LIST = numpy.array([17,
                                    26,27,28,29,30,
                                    36,37,38,39,40,41,42,
                                    47,48,49,50,51,52,53,
                                    57,58,59,60,61,62,63,64,65,
                                    69,70,71,72,73,74,75,
                                    80,81,82,83,84,85,86,
                                    92,93,94,95,96,
                                    116]) - 1
            DM_POS_TEMP = numpy.array([[0.5,4.5],
                                       [1.5,2.5],[1.5,3.5],[1.5,4.5],[1.5,5.5],[1.5,6.5],
                                       [2.5,1.5],[2.5,2.5],[2.5,3.5],[2.5,4.5],[2.5,5.5],[2.5,6.5],[2.5,7.5],
                                       [3.5,1.5],[3.5,2.5],[3.5,3.5],[3.5,4.5],[3.5,5.5],[3.5,6.5],[3.5,7.5],
                                       [4.5,0.5],[4.5,1.5],[4.5,2.5],[4.5,3.5],[4.5,4.5],[4.5,5.5],[4.5,6.5],[4.5,7.5],[4.5,8.5],
                                       [5.5,1.5],[5.5,2.5],[5.5,3.5],[5.5,4.5],[5.5,5.5],[5.5,6.5],[5.5,7.5],
                                       [6.5,1.5],[6.5,2.5],[6.5,3.5],[6.5,4.5],[6.5,5.5],[6.5,6.5],[6.5,7.5],
                                       [7.5,2.5],[7.5,3.5],[7.5,4.5],[7.5,5.5],[7.5,6.5],
                                       [8.5,4.5]])
            
        elif dm.n_acts == 49:
            FULL_ACTS = numpy.array([17,
                                     23,24,25,
                                     31],dtype=int)
            ACT_LIST = numpy.array([10,
                                     16,17,18,
                                     22,23,24,25,26,
                                     30,31,32,
                                     38],dtype=int)
            position = numpy.array([[0,2],[0,3],
                                    [1,1],[1,2],[1,3],[1,4],
                                    [2,0],[2,1],[2,2],[2,3],[2,4],[2,5],
                                    [3,0],[3,1],[3,2],[3,3],[3,4],[3,5],
                                    [4,1],[4,2],[4,3],[4,4],
                                    [5,2],[5,3]],
                                   dtype=int)
            DM_POS_TEMP = numpy.array([[0.5,2.5],
                                       [1.5,1.5],[1.5,2.5],[1.5,3.5],
                                       [2.5,0.5],[2.5,1.5],[2.5,2.5],[2.5,3.5],[2.5,4.5],
                                       [3.5,1.5],[3.5,2.5],[3.5,3.5],
                                       [4.5,2.5]])
            
        elif dm.n_acts == 169:
            # FULL_ACTS = numpy.array([10, 16,17,18, 22,23,24,25,26, 30,31,32, 38],dtype=int)
            FULL_ACTS = numpy.array([ 32,
                                      43, 44, 45, 46, 47,
                                      55, 56, 57, 58, 59, 60, 61,
                                      68, 69, 70, 71, 72, 73, 74,
                                      80, 81, 82, 83, 84, 85, 86, 87, 88,
                                      94, 95, 96, 97, 98, 99,100,
                                     107,108,109,110,111,112,113,
                                     121,122,123,124,125,
                                     136],dtype=int)
            position = numpy.array([[0,5],[0,6], # 12x12 subx
                                 [1,3],[1,4],[1,5],[1,6],[1,7],[1,8],
                                 [2,2],[2,3],[2,4],[2,5],[2,6],[2,7],[2,8],[2,9],
                                 [3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[3,10],
                                 [4,1],[4,2],[4,3],[4,4],[4,5],[4,6],[4,7],[4,8],[4,9],[4,10],
                                 [5,0],[5,1],[5,2],[5,3],[5,4],[5,5],[5,6],[5,7],[5,8],[5,9],[5,10],[5,11],
                                 [6,0],[6,1],[6,2],[6,3],[6,4],[6,5],[6,6],[6,7],[6,8],[6,9],[6,10],[6,11],
                                 [7,1],[7,2],[7,3],[7,4],[7,5],[7,6],[7,7],[7,8],[7,9],[7,10],
                                 [8,1],[8,2],[8,3],[8,4],[8,5],[8,6],[8,7],[8,8],[8,9],[4,10],
                                 [9,2],[9,3],[9,4],[9,5],[9,6],[9,7],[9,8],[9,9],
                                 [10,3],[10,4],[10,5],[10,6],[10,7],[10,8],
                                 [11,5],[11,6]],dtype=int)
            ACT_LIST = numpy.array([ 32,
                                  43, 44, 45, 46, 47,
                                  55, 56, 57, 58, 59, 60, 61,
                                  68, 69, 70, 71, 72, 73, 74,
                                  80, 81, 82, 83, 84, 85, 86, 87, 88,
                                  94, 95, 96, 97, 98, 99,100,
                                 107,108,109,110,111,112,113,
                                 121,122,123,124,125,
                                 136],dtype=int)
            DM_POS_TEMP = numpy.array([[1.5,5.5],
                                    [2.5,3.5],[2.5,4.5],[2.5,5.5],[2.5,6.5],[2.5,7.5],
                                    [3.5,2.5],[3.5,3.5],[3.5,4.5],[3.5,5.5],[3.5,6.5],[3.5,7.5],[3.5,8.5],
                                    [4.5,1.5],[4.5,2.5],[4.5,3.5],[4.5,4.5],[4.5,5.5],[4.5,6.5],[4.5,7.5],
                                    [5.5,1.5],[5.5,2.5],[5.5,3.5],[5.5,4.5],[5.5,5.5],[5.5,6.5],[5.5,7.5],[5.5,8.5],[5.5,9.5],
                                    [6.5,2.5],[6.5,3.5],[6.5,4.5],[6.5,5.5],[6.5,6.5],[6.5,7.5],[6.5,8.5],
                                    [7.5,1.5],[7.5,2.5],[7.5,3.5],[7.5,4.5],[7.5,5.5],[7.5,6.5],[7.5,7.5],
                                    [8.5,3.5],[8.5,4.5],[8.5,5.5],[8.5,6.5],[8.5,7.5],
                                    [9.5,5.5]])
        else:
            FULL_ACTS = numpy.array([])
        
        first_plot = False
        
        for i in range(dm.n_acts):
            
            
            
            # Set vector of iMat commands and phase to 0
            actCommands[:] = 0

            # Except the one we want to make an iMat for!
            actCommands[i] = 1#dm.dmConfig.iMatValue

            # Now get a DM shape for that command
            phase[:] = 0
            phase[dm.n_dm] = dm.dmFrame(actCommands)
            for DM_N, DM in self.dms.items():
                if DM.config.type == 'Aberration':
                    if DM.config.calibrate == False:
                        phase[DM_N] = DM.dmFrame('flat')
                    else:
                        phase[DM_N] = DM.dmFrame('shape')
                        # plt.imshow(phase[DM_N])
                        # plt.colorbar()
                        # plt.show()
            # Send the DM shape off to the relavent WFS. put result in iMat
            n_wfs_measurments = 0
            for wfs_n, wfs in self.wfss.items():
                # turn off wfs noise if set
                if self.config.imat_noise is False:
                    wfs_pnoise = wfs.config.photonNoise
                    wfs.config.photonNoise = False
                    wfs_rnoise = wfs.config.eReadNoise
                    wfs.config.eReadNoise = 0
                
                # plot = self.soapy_config.wfss[0].plot
                if self.soapy_config.wfss[0].plot == True:
                    if dm.n_acts == 121:
                        if i == 60:
                            plot = True
                            self.soapy_config.wfss[0].plot = False
                        else:
                            plot = False
                    elif dm.n_acts == 81:
                        if i == 40:
                            plot = True
                            self.soapy_config.wfss[0].plot = False
                        else:
                            plot = False
                    elif dm.n_acts == 49:
                        if i == 24:
                            plot = True
                            self.soapy_config.wfss[0].plot = False
                        else:
                            plot = False
                    elif dm.n_acts == 25:
                        if i == 12:
                            plot = True
                            self.soapy_config.wfss[0].plot = False
                        else:
                            plot = False
                    else:
                        plot = False
                else:
                    plot = False
                
                
                
                iMat[i, n_wfs_measurments: n_wfs_measurments+wfs.n_measurements] = -1 * (
                    wfs.frame(scrns=None, phase_correction=phase, iMatFrame=True)
                    - zero_iMat[n_wfs_measurments: n_wfs_measurments+wfs.n_measurements])# / (dm.dmConfig.iMatValue)
                # print(self.soapy_config.recon.analyseAberationCalib,plot,i,FULL_ACTS)
                if self.soapy_config.recon.analyseAberationCalib:# and DM.config.calibrate == True:
                    
                    if (i == FULL_ACTS).any():
                        # print(first_plot)
                        # if (first_plot == True) and (plot == True) :
                        #     first_plot = False
                        #     # print(first_plot)
                        #     if dm.n_acts == 81:
                        #         actCommands = numpy.zeros((81),dtype=float)
                        #         actCommands[20] = 1
                        #         actCommands[22] = 1
                        #         actCommands[24] = 1
                        #         actCommands[38] = 1
                        #         actCommands[40] = 1
                        #         actCommands[42] = 1
                        #         actCommands[56] = 1
                        #         actCommands[58] = 1
                        #         actCommands[60] = 1
                        #     elif dm.n_acts == 25:
                        #         actCommands = numpy.zeros((25),dtype=float)
                        #         actCommands[11] = 1
                        #         actCommands[13] = 1
                        #     elif dm.n_acts == 49:
                        #         actCommands = numpy.zeros((49),dtype=float)
                        #         actCommands[10] = 1
                        #         actCommands[22] = 1
                        #         actCommands[24] = 1
                        #         actCommands[26] = 1
                        #         actCommands[38] = 1
                        #     elif dm.n_acts == 169:
                        #         actCommands = numpy.zeros((169),dtype=float)
                        #         actCommands[32] = 1
                        #         actCommands[56] = 1
                        #         actCommands[58] = 1
                        #         actCommands[60] = 1
                        #         actCommands[80] = 1
                        #         actCommands[82] = 1
                        #         actCommands[84] = 1
                        #         actCommands[86] = 1
                        #         actCommands[88] = 1
                        #         actCommands[108] = 1
                        #         actCommands[110] = 1
                        #         actCommands[112] = 1
                        #         actCommands[136] = 1
                        #     # print(dm.n_acts,actCommands.shape)
                        #     phase[dm.n_dm] = dm.dmFrame(actCommands)
                        #     for nPhase in range(phase.shape[0]):
                        #         plt.imshow(phase[nPhase])
                        #         plt.colorbar()
                        #         plt.title('DM{:}'.format(nPhase))
                        #         plt.show()
                        #     wfs.frame(None, phase_correction=phase, iMatFrame=True,iMatFramePlot=True)
                            
                        #     xx = numpy.arange(numpy.array(zero_wfs_efield[wfs_n]).shape[0],dtype=float)
                        #     xx -= xx.max()/2.
                        #     xx /= wfs.nx_subap_interp
                            
                        #     zero_wfs_efield[wfs_n][wfs.scaledMask == 0] = numpy.nan
                        #     # plt.imshow(numpy.asarray(zero_wfs_efield[wfs_n]).real)
                        #     # plt.show()
                            
                        #     wfs.interp_efield[wfs.scaledMask == 0] = numpy.nan
                        #     # plt.imshow(wfs.interp_efield.real)
                        #     # plt.show()
                            
                        #     with_aberration_efield = (wfs.interp_efield
                        #                             / numpy.asarray(zero_wfs_efield[wfs_n]))
                        #     # plt.imshow(with_aberration_efield.real)
                        #     # plt.show()
                            
                            
                        #     to_plot = numpy.unwrap(numpy.unwrap(numpy.angle(numpy.nan_to_num(with_aberration_efield)),axis=0),axis=1)
                        #     to_plot[wfs.scaledMask == 0] = numpy.nan
                        #     # to_plot = my_unwrap(numpy.angle(with_aberration_efield))
                        #     # plt.imshow(to_plot)
                        #     # plt.show()
                        #     to_plot -= numpy.nanmedian(to_plot)
                        #     fig, ax1 = plt.subplots()
                        #     # print(to_plot[64,64])
                        #     poke_value = self.soapy_config.dms[1].iMatValue*1e-9 / self.soapy_config.wfss[0].wavelength * 2.0 * numpy.pi
                        #     c = ax1.pcolor(xx,xx,
                        #                 -to_plot,vmin=0,vmax=poke_value)
                            
                        #     ax1.hlines((-4,-3,-2,-1,0,1,2,3,4),xmin=-4,xmax=4,color='w')
                        #     ax1.vlines((-4,-3,-2,-1,0,1,2,3,4),ymin=-4,ymax=4,color='w')
                        #     # ax1.plot([0,2,-2,0,0,-2,-2,2,2],[0,0,0,2,-2,-2,2,-2,2],color='white',marker='o',ls='')
                        #     fig.colorbar(c,ax=ax1)
                        #     ax1.axis('square')
                            
                        #     frame1 = ax1
                        #     for xlabel_i in frame1.axes.get_xticklabels():
                        #         xlabel_i.set_visible(False)
                        #         xlabel_i.set_fontsize(0.0)
                        #     for xlabel_i in frame1.axes.get_yticklabels():
                        #         xlabel_i.set_fontsize(0.0)
                        #         xlabel_i.set_visible(False)
                                
                        #     ax1.tick_params(axis='both', which='both', length=0)
                            
                        #     plt.gcf().set_size_inches(6,6)
                            
                        #     plt.title('dm_influence')
                        #     plt.savefig('dm_influence-' + time.strftime("%Y-%m-%d-%H-%M-%S") + '.png',
                        #                 dpi=300,bbox_inches='tight',transparent=True)
                        
                        #     plt.show()
                        #     plt.clf()
                        #     # reset things to where they were
                        #     actCommands[:] = 0
    
                        #     # Except the one we want to make an iMat for!
                        #     actCommands[i] = dm.dmConfig.iMatValue*1e-9
                        #     phase[:] = 0
                        #     phase[dm.n_dm] = dm.dmFrame(actCommands)
                            
                        #     for DM_N, DM in self.dms.items():
                        #         if DM.config.type == 'Aberration':
                        #             if DM.config.calibrate == False:
                        #                 phase[DM_N] = DM.dmFrame('flat')
                        #             else:
                        #                 phase[DM_N] = DM.dmFrame('shape')
                                        
                        #     wfs.frame(None, phase_correction=phase, iMatFrame=True,iMatFramePlot=True)
                        # if plot == True:
                        #     plt.imshow(phase.sum(0))
                        #     plt.colorbar()
                        #     plt.show()
                        
                        xx = numpy.arange(numpy.array(zero_wfs_efield[wfs_n]).shape[0],dtype=float)
                        xx -= xx.max()/2.
                        xx /= wfs.nx_subap_interp
                        # print(wfs.nx_subap_interp)
                        
                        # zero_wfs_efield[wfs_n][wfs.scaledMask == 0] = numpy.nan
                        
                        wfs_size = wfs.interp_efield.shape[0]
                        
                        A = phase.shape[1]
                        B = self.soapy_config.sim.pupilSize
                        P = (A - B)//2
                        Q = (A + B)//2
                        
                        no_aberration_efield = (numpy.exp(1j*interp.zoom(phase[:-1,P:Q,P:Q].sum(0),wfs_size)/500.*2*numpy.pi))
                        no_aberration = numpy.unwrap(numpy.unwrap(numpy.angle(no_aberration_efield),axis=0),axis=1)
                        # no_aberration[wfs.scaledMask == 0] = numpy.nan
                        
                        # wfs.interp_efield[wfs.scaledMask == 0] = numpy.nan
                        
                        with_aberration_efield = (wfs.interp_efield
                                                / numpy.asarray(zero_wfs_efield[wfs_n]))
                        
                        # with_aberration = numpy.unwrap(numpy.unwrap(numpy.angle(with_aberration_efield),axis=0),axis=1)
                        with_aberration = numpy.angle(with_aberration_efield)
                        # with_aberration[wfs.scaledMask == 0] = numpy.nan
                        with_aberration[wfs.scaledMask == 0] = 0
        
                        wfs.interp_efield /= numpy.nanmean(numpy.abs(wfs.interp_efield)**2)**0.5
                        
                        dm.rytov[i,wfs_n] = numpy.nanvar(numpy.log(numpy.abs(
                            wfs.interp_efield[
                                numpy.asarray(wfs.scaledMask,dtype=bool)])))
                        
                        # if plot == True:
                        
                        #     no_aberration_phase_max_x,no_aberration_phase_max_y = find_centre(no_aberration,wfs.nx_subap_interp)
                        #     with_aberration_phase_max_x,with_aberration_phase_max_y = find_centre(-with_aberration,wfs.nx_subap_interp)
                        #     max_shift_x = with_aberration_phase_max_x - no_aberration_phase_max_x
                        #     max_shift_y = with_aberration_phase_max_y - no_aberration_phase_max_y
                            
                        #     # dm.subapShiftv2[i,wfs_n,0] = max_shift_x
                        #     # dm.subapShiftv2[i,wfs_n,1] = max_shift_y
                            
                        #     MEDIAN = numpy.nanmedian(no_aberration)
                        #     MAX = numpy.nanmax(no_aberration)
                        #     no_aberration -= MEDIAN
                        #     no_aberration /= MAX
                        #     with_aberration -= -MEDIAN
                        #     with_aberration /= MAX

                        #     plt.imshow(no_aberration)
                        #     plt.colorbar()
                        #     plt.show()              
                        #     plt.imshow(with_aberration)
                        #     plt.colorbar()
                        #     plt.show()       
                            
                        #     MAX = numpy.nanmax(-no_aberration)
                        #     MIN = numpy.nanmin(-no_aberration)
                        #     import time
                        #     s = time.gmtime(time.time())
                        #     time_stamp = time.strftime("%H-%M-%S", s)
                        #     # plt.figure(figsize=(6,4))
                        #     # plt.pcolor(xx,xx,numpy.angle(with_aberration_efield.T))
                        #     # plt.hlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
                        #     # plt.vlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
                        #     # plt.axis('square')
                        #     # plt.colorbar()
                        #     # plt.title('with aberration efield')
                        #     # plt.savefig(time_stamp + 'influ-with-ab{:d}.pdf'.format(i),dpi=600,bbox_inches='tight')
                        #     # plt.show()
                        #     mask = numpy.copy(numpy.asarray(wfs.scaledMask,dtype=float))
                        #     mask[mask==0] = numpy.nan
                        #     plt.figure(figsize=(6,4))
                        #     plt.pcolor(xx,xx,-with_aberration*mask,vmin=0,vmax=1)
                        #     plt.hlines(numpy.arange(-5,6),-5,5,ls=':',color='r')
                        #     plt.vlines(numpy.arange(-5,6),-5,5,ls=':',color='r')
                        #     plt.plot(no_aberration_phase_max_y,no_aberration_phase_max_x,marker='o',color='tab:pink',label='true location',markersize=10)
                        #     plt.plot(with_aberration_phase_max_y,with_aberration_phase_max_x,marker='X',color='tab:red',label='apparent position : influence',markersize=10)
                        #     plt.axis('square')
                        #     # plt.title('with aberration')
                        #     plt.colorbar()
                        #     plt.savefig(time_stamp + 'influ-with-ab{:d}.pdf'.format(i),dpi=600,bbox_inches='tight')
                        #     plt.show()
                        
                        size_A = with_aberration.shape[0]
                        
                        with_aberration[wfs.scaledMask == 0] = numpy.nan
                        no_aberration[wfs.scaledMask == 0] = numpy.nan
                        
                        # schmidt 2010 pg 45
                        
                        temp_corr = numpy.fft.fftshift(numpy.fft.ifft2(
                            numpy.fft.fft2(numpy.pad(
                                numpy.nan_to_num(with_aberration),
                                ((size_A//2,size_A//2),(size_A//2,size_A//2)),mode='constant'))
                            * numpy.conjugate(numpy.fft.fft2(numpy.pad(
                                numpy.nan_to_num(-no_aberration),
                                ((size_A//2,size_A//2),(size_A//2,size_A//2)),mode='constant')))
                            ))#.real[size_A//2:size_A*3//2,size_A//2:size_A*3//2]
                        
                        corr = temp_corr.real[size_A//2:size_A*3//2,size_A//2:size_A*3//2]

                        # mask = numpy.pad(numpy.ones((size_A,size_A),dtype=float),
                        #                  ((size_A//2,size_A//2),(size_A//2,size_A//2)),
                        #                  mode='constant')

                        # mask_corr = numpy.fft.fftshift(numpy.fft.ifft2(numpy.abs(
                        #     numpy.fft.fft2(mask))**2))#.real[size_A//2:size_A*3//2,size_A//2:size_A*3//2]
                        
                        # idx = (mask_corr != 0)
                        # c = numpy.zeros_like(temp_corr)
                        # c[idx] = temp_corr[idx] / mask_corr[idx] * mask[idx]
                        # corr = c.real[size_A//2:size_A*3//2,size_A//2:size_A*3//2]

                        # MAX = numpy.nanmax(-no_aberration)
                        # MIN = numpy.nanmin(-no_aberration)
                        
                        # if plot == True:
                        
                        #     plt.pcolor(xx,xx,-no_aberration.T,vmin=MIN,vmax=MAX)
                        #     plt.axis('square')
                        #     plt.title('original poke phase')
                        #     plt.colorbar()
                        #     plt.hlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
                        #     plt.vlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
                        #     # plt.savefig('poke{:d}.png'.format(i))
                        #     plt.show()
                            
                        #     plt.pcolor(xx,xx,with_aberration.T,vmin=MIN,vmax=MAX)
                        #     plt.hlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
                        #     plt.vlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
                        #     plt.title('real interaction')
                        #     plt.axis('square')
                        #     plt.colorbar()
                        #     # plt.savefig('im{:d}.png'.format(i))
                        #     plt.show()
                    
                        
                        
                        
                        true_locx, true_locy = find_centre(corr,wfs.nx_subap_interp)
                        
                        # if plot == True:
                        
                        #     x = numpy.linspace(-4,4,num=corr.shape[0])
                        #     plt.pcolor(x,x,corr.T)#,vmin=0,vmax=1000)
                        #     plt.hlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
                        #     plt.vlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
                        #     plt.plot(true_locx,true_locy,ls='',marker='o',color='white',label='true max')
                        #     plt.legend()
                        #     plt.colorbar()
                        #     plt.axis('square')
                        #     plt.title('true max at ({:.2f},{:.2f})'.format(true_locx,true_locy))
                        #     plt.show()
                        
                        # if ((numpy.abs(true_locx) < wfs.nx_subaps//2)
                        #     or (numpy.abs(true_locy) < wfs.nx_subaps//2)):
                            
                        dm.subapShift_influence[i,wfs_n,0] = true_locx
                        dm.subapShift_influence[i,wfs_n,1] = true_locy
                        
                        # else:
                        #     dm.subapShift[i,wfs_n,0] = numpy.nan
                        #     dm.subapShift[i,wfs_n,1] = numpy.nan
                        
                        if (plot == True):# and (i == 40):
                            POS = dm.valid_act_coords[i] - dm.valid_act_coords.max()//2
                            xx = numpy.arange(numpy.array(zero_wfs_efield[wfs_n]).shape[0],dtype=float)
                            xx -= xx.max()/2.
                            xx /= wfs.nx_subap_interp
                            center = numpy.nanmedian(no_aberration)
                            with_aberration += center
                            no_aberration -= center
                            MAX = numpy.nanmax(no_aberration)#0.006#
                            MIN = numpy.nanmin(no_aberration)#-0.006#
                            # plt.imshow(with_aberration);plt.show()
                            
                            
                            plt.figure(figsize=(6,4))
                            plt.pcolor(xx,xx,-with_aberration/MAX,vmin=MIN/MAX,vmax=1,cmap='viridis_r')
                            plt.axis('square')
                            plt.colorbar()
                            # plt.hlines(numpy.arange(-5,6),-5,5,ls=':',color='r')
                            # plt.vlines(numpy.arange(-5,6),-5,5,ls=':',color='r')
                            plt.plot(POS[1],POS[0],ls='',marker='.',color='white',label='true location',markersize=10)
                            # plt.hlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
                            # plt.vlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
                            plt.plot(true_locy+POS[1],true_locx + POS[0],ls='',marker='.',color='tab:orange',label='apparent position : influence',markersize=10)
                            import time
                            s = time.gmtime(time.time())
                            time_stamp = time.strftime("%H-%M-%S", s)
                            plt.savefig(time_stamp + 'im-influ{:d}.png'.format(i),bbox_inches='tight',dpi=600)
                            plt.show()
                            
                            plt.figure(figsize=(6,4))
                            plt.pcolor(xx,xx,-with_aberration/MAX,vmin=MIN/MAX,vmax=1,cmap='viridis_r')
                            plt.axis('square')
                            plt.colorbar()
                            # plt.hlines(numpy.arange(-5,6),-5,5,ls=':',color='r')
                            # plt.vlines(numpy.arange(-5,6),-5,5,ls=':',color='r')
                            plt.plot(POS[1],POS[0],ls='',marker='.',color='white',label='true location',markersize=10)
                            # plt.hlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
                            # plt.vlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
                            # plt.plot(POS[1],POS[0],ls='',marker='^',color='white',label='original max')
                            # plt.title('real interaction'
                            #           + '\nact_original at ({:.2f},{:.2f})'.format(POS[0],POS[1])
                            #           + '\nact_actual at ({:.2f},{:.2f})'.format(true_locx+POS[0],true_locy+POS[1])
                            #           + '\nshift = ({:.2f},{:.2f})'.format(true_locx,true_locy))
                            # plt.axis('square')
                            # plt.legend()
                            # plt.colorbar()
                            # plt.xlim(-position.max()/2-1,position.max()/2+1)
                            # plt.ylim(-position.max()/2-1,position.max()/2+1)
                            # plt.savefig('im{:d}.png'.format(i))
                        
                        if (i == ACT_LIST).any():
                            
                            # print(i,ACT_LIST)
                            # print(DM_POS_TEMP,DM_POS_TEMP[numpy.where(ACT_LIST == i),:])
                            # print(measure_AS_from_poke(
                            #     iMat[i, n_wfs_measurments: n_wfs_measurments+wfs.n_measurements],
                            #     position,DM_POS_TEMP[numpy.where(ACT_LIST == i),:],0.5))
                            
                            
                            dm.subapShift_slope[i,wfs_n] = measure_AS_from_poke(
                                iMat[i, n_wfs_measurments: n_wfs_measurments+wfs.n_measurements],
                                position,DM_POS_TEMP[numpy.where(ACT_LIST == i),:],0.25,debug=plot)
                        
                        
                        if (plot == True):# and (i == 40):
                            plt.plot(true_locy+POS[1],true_locx + POS[0],ls='',marker='.',color='tab:orange',label='apparent position : influence',markersize=10)
                            
                            # POS = dm.valid_act_coords[i] - dm.valid_act_coords.max()//2
                            # xx = numpy.arange(numpy.array(zero_wfs_efield[wfs_n]).shape[0],dtype=float)
                            # xx -= xx.max()/2.
                            # xx /= wfs.nx_subap_interp
                            # MAX = numpy.nanmax(-no_aberration)
                            # MIN = numpy.nanmin(-no_aberration)
                            # # plt.imshow(with_aberration);plt.show()
                            # plt.pcolor(xx,xx,with_aberration,vmin=MIN,vmax=MAX)
                            # plt.hlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
                            # plt.vlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
                            # plt.plot(true_locy+POS[1],true_locx + POS[0],ls='',marker='o',color='r',label='true max')
                            
                            # # plt.title('real interaction'
                            # #           + '\nact_original at ({:.2f},{:.2f})'.format(POS[0],POS[1])
                            # #           + '\nact_actual at ({:.2f},{:.2f})'.format(true_locx+POS[0],true_locy+POS[1])
                            # #           + '\nshift = ({:.2f},{:.2f})'.format(true_locx,true_locy))
                            # plt.axis('square')
                            # plt.legend()
                            # plt.colorbar()
                            # plt.xlim(-position.max()/2-0.5,position.max()/2+0.5)
                            # plt.ylim(-position.max()/2-0.5,position.max()/2+0.5)
                            # plt.legend()
                            
                            # plt.axis('square')
                            # plt.colorbar()
                            import time
                            s = time.gmtime(time.time())
                            time_stamp = time.strftime("%H-%M-%S", s)
                            plt.savefig(time_stamp + 'im{:d}.png'.format(i),bbox_inches='tight',dpi=600)
                            plt.show()
                            
                            nxsubap = wfs.soapy_config.wfss[0].nxSubaps
                            pxlsPerSubap = wfs.soapy_config.wfss[0].pxlsPerSubap
                            low_bound = nxsubap//2*pxlsPerSubap
                            up_bound = (nxsubap//2+1)*pxlsPerSubap
                            
                            ap = numpy.copy(wfs.wfsDetectorPlane / wfs.wfsDetectorPlane.max())
                            # ap -= self.soapy_config.wfss[0].centThreshold
                            # ap[ap < 0] = 0.0
                            # ap /= ap.max()
                            
                            plt.imshow(ap)
                            plt.colorbar()
                            # plt.title('sample wfs')
                            plt.show()
                            
                            subap = numpy.copy(wfs.wfsDetectorPlane[low_bound:up_bound,low_bound:up_bound])

                            plt.imshow(subap)
                            plt.colorbar()
                            # plt.title('sample wfs subap')
                            plt.show()
                    
                
                n_wfs_measurments += wfs.n_measurements

                # Turn noise back on again if it was turned off
                if self.config.imat_noise is False:
                    wfs.config.photonNoise = wfs_pnoise
                    wfs.config.eReadNoise = wfs_rnoise

            if callback != None:
                callback()

            
            logger.statusMessage(i, dm.n_acts,
                                "Generating {} Actuator DM iMat".format(dm.n_acts))
        
        if ABERRATION == True:
            logger.info("NOT Checking for redundant actuators...")
            valid_actuators = numpy.ones((dm.n_acts), dtype="int")
    
            dm.valid_actuators = valid_actuators
            n_valid_acts = valid_actuators.sum()
            logger.info("DM {} has {} valid actuators ({} dropped)".format(
                    dm.n_dm, n_valid_acts, dm.n_acts - n_valid_acts))
    
            # Can now make a final interaction matrix with only valid entries
            valid_iMat = numpy.zeros((n_valid_acts, self.sim_config.totalWfsData))
            i_valid_act = 0
            for i in range(dm.n_acts):
                if valid_actuators[i]:
                    valid_iMat[i_valid_act] = iMat[i]
                    i_valid_act += 1
    
            return valid_iMat
        
        else:
            
            logger.info("Checking for redundant actuators...")
            # Now check tath each actuator actually does something on a WFS.
            # If an act has a <0.1% effect then it will be removed
            # NOTE: THIS SHOULD REALLY BE DONE ON A PER WFS BASIS
            valid_actuators = numpy.zeros((dm.n_acts), dtype="int")
            act_threshold = abs(iMat).max() * 0.001
            for i in range(dm.n_acts):
                # plt.plot(i,abs(iMat[i]).max(),marker='o',ls='')
                if abs(iMat[i]).max() > act_threshold:
                    valid_actuators[i] = 1
                else:
                    valid_actuators[i] = 0
            # plt.hlines(act_threshold,0,dm.n_acts)
            # plt.show()
    
            dm.valid_actuators = valid_actuators
            n_valid_acts = valid_actuators.sum()
            logger.info("DM {} has {} valid actuators ({} dropped)".format(
                    dm.n_dm, n_valid_acts, dm.n_acts - n_valid_acts))
    
            # Can now make a final interaction matrix with only valid entries
            valid_iMat = numpy.zeros((n_valid_acts, self.sim_config.totalWfsData))
            i_valid_act = 0
            for i in range(dm.n_acts):
                if valid_actuators[i]:
                    valid_iMat[i_valid_act] = iMat[i]
                    i_valid_act += 1
    
            return valid_iMat


    def get_dm_imat(self, dm_index, wfs_index):
        """
        Slices and returns the interaction matrix between a given wfs and dm from teh main interaction matrix

        Parameters:
            dm_index (int): Index of required DM
            wfs_index (int): Index of required WFS

        Return:
             ndarray: interaction matrix
        """

        act_n1 = self.first_acts[dm_index]
        act_n2 = act_n1 + self.dms[dm_index].n_acts

        wfs_n1 = self.wfss[wfs_index].config.dataStart
        wfs_n2 = wfs_n1 + self.wfss[wfs_index].n_measurements
        return self.interaction_matrix[act_n1: act_n2, wfs_n1: wfs_n2]


    def makeCMat(
            self, loadIMat=True, loadCMat=True, callback=None,
            progressCallback=None,
            imat_to_load=None,cmat_to_load=None):
        if loadIMat:
            try:
                self.load_interaction_matrix(imat_to_load=imat_to_load)
                logger.info("Interaction Matrices loaded successfully")
            except:
                tc = traceback.format_exc()
                logger.info("Load Interaction Matrices failed with error: {} - will create new one...".format(tc))
                self.makeIMat(callback=callback)
                if self.sim_config.simName is not None:
                    self.save_interaction_matrix()
                logger.info("Interaction Matrices Done")

        else:
            self.makeIMat(callback=callback)
            # if self.sim_config.simName is not None:
            #         self.save_interaction_matrix()
            logger.info("Interaction Matrices Done")

        if loadCMat:
            try:
                self.loadCMat(cmat_to_load=cmat_to_load)
                logger.info("Command Matrix Loaded Successfully")
            except:
                tc = traceback.format_exc()
                logger.warning("Load Command Matrix failed qith error: {} - will create new one...".format(tc))

                self.calcCMat(callback, progressCallback)
                if self.sim_config.simName is not None:
                    self.saveCMat()
                logger.info("Command Matrix Generated!")
        else:
            logger.info("Creating Command Matrix")
            self.calcCMat(callback, progressCallback)
            if self.sim_config.simName is not None:
                    self.saveCMat()
            logger.info("Command Matrix Generated!")

    def apply_gain(self):
        """
        Applies the gains set for each DM to the DM actuator commands. 
        Also applies different control law if DM is in "closed" or "open" loop mode
        """
        # Loop through DMs and apply gain
        n_act1 = int(0)
        for dm_i, dm in self.dms.items():

            n_act2 = n_act1 + int(dm.n_valid_actuators)
            # If loop is closed, only add residual measurements onto old
            # actuator values
            if dm.dmConfig.closed:
                self.actuator_values[n_act1: n_act2] += (dm.dmConfig.gain * self.new_actuator_values[n_act1: n_act2])

            else:
                self.actuator_values[n_act1: n_act2] = ((dm.dmConfig.gain * self.new_actuator_values[n_act1: n_act2])
                                + ( (1. - dm.dmConfig.gain) * self.actuator_values[n_act1: n_act2]) )

            n_act1 += int(dm.n_valid_actuators)


    def reconstruct(self, wfs_measurements):
        t = time.time()

        if self.actuator_values is None:
            self.actuator_values = numpy.zeros((int(self.sim_config.totalActs)),dtype=float)

        self.new_actuator_values = self.control_matrix.T.dot(wfs_measurements)

        self.apply_gain()

        self.Trecon += time.time()-t
        return self.actuator_values

    def reset(self):
        if self.actuator_values is not None:
            self.actuator_values[:] = 0


class MVM(Reconstructor):
    """
    Re-constructor which combines all DM interaction matrices from all DMs and
    WFSs and inverts the resulting matrix to form a global interaction matrix.
    """

    def calcCMat(self, callback=None, progressCallback=None):
        '''
        Uses DM object makeIMat methods, then inverts each to create a
        control matrix
        '''
        
        
        
        # cumulative_actuators = 0
        # old_cumulative_actuators = 0
        # for i in self.dms:
        #     if self.dms[i].dmConfig.type == 'TT':
        #         cumulative_actuators += self.dms[i].n_acts
        #         old_iMat = self.dms[i].config.iMatValue
        #         new_iMat = old_iMat*self.interaction_matrix.max()/self.interaction_matrix[old_cumulative_actuators:cumulative_actuators].max()
        #         old_cumulative_actuators = cumulative_actuators
        #         print(self.dms[i],new_iMat)
        #     # if self.dms[i].dmConfig.type =='FastPiezo':
        #     #     cumulative_actuators += self.dms[i].n_acts
        #     #     old_iMat = self.dms[i].config.iMatValue
        #     #     new_iMat = old_iMat/numpy.max(svd[old_cumulative_actuators:cumulative_actuators])
        #     #     new_iMat *= (1 - 0.2*(len(self.dms) - i)/len(self.dms))
        #     #     old_cumulative_actuators = cumulative_actuators
        #     #     print(self.dms[i],new_iMat)
        
        if self.config.svdConditioning == 'adaptive':
            rcond = get_rcond_adaptive_threshold_rank(self.interaction_matrix)
            self.config.svdConditioning = rcond
        logger.info("Invert iMat with conditioning: {:.4f}".format(
                self.config.svdConditioning))
        self.control_matrix = numpy.linalg.pinv(
                self.interaction_matrix, self.config.svdConditioning
                )
        # plt.imshow(self.interaction_matrix)
        # plt.title('control matrix')
        # plt.show()
        # _,svd,_ = numpy.linalg.svd(self.interaction_matrix)
        # svd /= svd.max()
        # plt.plot(svd,label='old svd')
        # plt.hlines(self.config.svdConditioning,0,len(svd),label='old svd')
        # plt.legend()
        # plt.show()
        # plt.imshow(self.control_matrix)
        # plt.title('control matrix')
        # plt.show()


class MVM_SeparateDMs(Reconstructor):
    """
    Re-constructor which treats a each DM Separately.

    Similar to ``MVM`` re-constructor, except each DM has its own control matrix.
    Its is assumed that each DM is "associated" with a different WFS.
    """

    def calcCMat(self,callback=None, progressCallback=None):
        '''
        Uses DM object makeIMat methods, then inverts each to create a
        control matrix
        '''
        acts = 0
        for dm_index, dm in self.dms.items():

            n_wfs_measurements = 0
            for wfs in dm.wfss:
                n_wfs_measurements += wfs.n_measurements

            dm_interaction_matrix = numpy.zeros((dm.n_acts, n_wfs_measurements))
            # Get interaction matrices from main matrix
            n_wfs_measurement = 0
            for wfs_index in [dm.dmConfig.wfs]:
                wfs = self.wfss[wfs_index]
                wfs_imat = self.get_dm_imat(dm_index, wfs_index)
                print("DM: {}, WFS: {}".format(dm_index, wfs_index))
                dm_interaction_matrix[:, n_wfs_measurement:n_wfs_measurement + wfs.n_measurements] = wfs_imat
            
            if dm.dmConfig.svdConditioning == 'adaptive':
                rcond = get_rcond_adaptive_threshold_rank(dm_interaction_matrix)
                dm.dmConfig.svdConditioning = rcond
            dm_control_matrx = numpy.linalg.pinv(dm_interaction_matrix, dm.dmConfig.svdConditioning)

            # now put carefully back into one control matrix
            for wfs_index in [dm.dmConfig.wfs]:
                wfs = self.wfss[wfs_index]
                self.control_matrix[
                        wfs.config.dataStart:
                                wfs.config.dataStart + wfs.n_measurements,
                        acts:acts+dm.n_acts] = dm_control_matrx

            acts += dm.n_acts


class LearnAndApply(MVM):
    '''
    Class to perform a simply learn and apply algorithm, where
    "learn" slopes are recorded, and an interaction matrix between off-axis
    and on-axis WFS is computed from these slopes.

    Assumes that on-axis sensor is WFS 0
    '''

    def makeIMat(self, callback=None):
        super(LearnAndApply, self).makeIMat(callback=callback)

        # only truth sensor and DM(s) interaction matrix needed 
        self.interaction_matrix = self.interaction_matrix[:,:2*self.wfss[0].n_subaps]

    def saveCMat(self):
        cMatFilename = self.sim_config.simName+"/cMat.fits"
        tomoMatFilename = self.sim_config.simName+"/tomoMat.fits"

        fits.writeto(
                cMatFilename, self.control_matrix,
                header=self.sim_config.saveHeader, overwrite=True
                )

        fits.writeto(
                tomoMatFilename, self.tomoRecon,
                header=self.sim_config.saveHeader, overwrite=True
                )

    def loadCMat(self):

        super(LearnAndApply, self).loadCMat()

        #Load tomo reconstructor
        tomoFilename = self.sim_config.simName+"/tomoMat.fits"
        tomoMat = fits.getdata(tomoFilename)

        #And check its the right size
        if tomoMat.shape != (
                2*self.wfss[0].n_subaps,
                self.sim_config.totalWfsData - 2*self.wfss[0].n_subaps):
            logger.warning("Loaded Tomo matrix not the expected shape - gonna make a new one..." )
            raise Exception
        else:
            self.tomoRecon = tomoMat


    def initControlMatrix(self):

        self.controlShape = (2*self.wfss[0].n_subaps, self.sim_config.totalActs)
        self.control_matrix = numpy.zeros( self.controlShape )


    def learn(self, callback=None, progressCallback=None):
        '''
        Takes "self.learnFrames" WFS frames, and computes the tomographic
        reconstructor for the system. This method uses the "truth" sensor, and
        assumes that this is WFS0
        '''

        self.learnSlopes = numpy.zeros( (self.learnIters,self.sim_config.totalWfsData) )
        for i in xrange(self.learnIters):
            self.learnIter=i

            scrns = self.moveScrns()

            for j in range(len(self.wfss)):
                wfs = self.wfss[j]
                self.learnSlopes[i,j*wfs.n_measurements:(j+1)*wfs.n_measurements] = wfs.frame(scrns, read=True)


            logger.statusMessage(i+1, self.learnIters, "Performing Learn")
            if callback!=None:
                callback()
            if progressCallback!=None:
               progressCallback("Performing Learn", i, self.learnIters )

        if self.sim_config.saveLearn:
            #FITS.Write(self.learnSlopes,self.sim_config.simName+"/learn.fits")
            fits.writeto(
                    self.sim_config.simName+"/learn.fits",
                    self.learnSlopes, header=self.sim_config.saveHeader,
                    overwrite=True )


    def calcCMat(self,callback=None, progressCallback=None):
        '''
        Uses the slopes recorded in the "learn" and DM interaction matrices
        to create a CMat.
        '''

        logger.info("Performing Learn....")
        self.learn(callback, progressCallback)
        logger.info("Done. Creating Tomographic Reconstructor...")

        if progressCallback!=None:
            progressCallback(1,1, "Calculating Covariance Matrices")

        self.covMat = numpy.cov(self.learnSlopes.T)
        Conoff = self.covMat[   :2*self.wfss[0].n_subaps,
                                2*self.wfss[0].n_subaps:     ]
        Coffoff = self.covMat[  2*self.wfss[0].n_subaps:,
                                2*self.wfss[0].n_subaps:    ]

        logger.info("Inverting offoff Covariance Matrix")
        iCoffoff = numpy.linalg.pinv(Coffoff, rcond=1e-8)

        self.tomoRecon = Conoff.dot(iCoffoff)
        logger.info("Done. \nCreating full reconstructor....")

        #Same code as in "MVM" class to create dm-slopes reconstructor.

        super(LearnAndApply, self).calcCMat(callback, progressCallback)

        #Dont make global reconstructor. Will reconstruct on-axis slopes, then
        #dmcommands explicitly
        #self.controlMatrix = (self.controlMatrix.T.dot(self.tomoRecon)).T
        logger.info("Done.")


    def reconstruct(self, slopes):
        """
        Determine DM commands using previously made
        reconstructor from slopes.
        Args:
            slopes (ndarray): array of slopes to reconstruct from
        Returns:
            ndarray: array of commands to be sent to DM
        """

        #Retreive pseudo on-axis slopes from tomo reconstructor
        slopes = self.tomoRecon.dot(slopes[2*self.wfss[0].n_subaps:])

        if self.dms[0].dmConfig.type=="TT":
            ttMean = slopes.reshape(2, self.wfss[0].n_subaps).mean(1)
            ttCommands = self.control_matrix[:,:2].T.dot(slopes)
            slopes[:self.wfss[0].n_subaps] -= ttMean[0]
            slopes[self.wfss[0].n_subaps:] -= ttMean[1]

            #get dm commands for the calculated on axis slopes
            dmCommands = self.control_matrix[:,2:].T.dot(slopes)

            return numpy.append(ttCommands, dmCommands)

        #get dm commands for the calculated on axis slopes
        dmCommands = super(LearnAndApply, self).reconstruct(slopes)
        #dmCommands = self.control_matrix.T.dot(slopes)
        return dmCommands


class LearnAndApplyLTAO(LearnAndApply, MVM_SeparateDMs):
    '''
    Class to perform a simply learn and apply algorithm, where
    "learn" slopes are recorded, and an interaction matrix between off-axis
    and on-axis WFS is computed from these slopes.

    This is an ``
    Assumes that on-axis sensor is WFS 1
    '''

    def initcontrol_matrix(self):

        self.controlShape = (2*(self.wfss[0].activeSubaps+self.wfss[1].activeSubaps), self.sim_config.totalActs)
        self.control_matrix = numpy.zeros( self.controlShape )


    def calcCMat(self,callback=None, progressCallback=None):
        '''
        Uses the slopes recorded in the "learn" and DM interaction matrices
        to create a CMat.
        '''

        logger.info("Performing Learn....")
        self.learn(callback, progressCallback)
        logger.info("Done. Creating Tomographic Reconstructor...")

        if progressCallback!=None:
            progressCallback(1,1, "Calculating Covariance Matrices")

        self.covMat = numpy.cov(self.learnSlopes.T)
        Conoff = self.covMat[
                self.wfss[1].config.dataStart:
                        self.wfss[2].config.dataStart,
                self.wfss[2].config.dataStart:
                ]
        Coffoff = self.covMat[  self.wfss[2].config.dataStart:,
                                self.wfss[2].config.dataStart:    ]

        logger.info("Inverting offoff Covariance Matrix")
        iCoffoff = numpy.linalg.pinv(Coffoff)

        self.tomoRecon = Conoff.dot(iCoffoff)
        logger.info("Done. \nCreating full reconstructor....")

        #Same code as in "MVM" class to create dm-slopes reconstructor.

        MVM_SeparateDMs.calcCMat(self, callback, progressCallback)

        #Dont make global reconstructor. Will reconstruct on-axis slopes, then
        #dmcommands explicitly
        #self.control_matrix = (self.control_matrix.T.dot(self.tomoRecon)).T
        logger.info("Done.")

    def reconstruct(self, slopes):
        """
        Determine DM commands using previously made
        reconstructor from slopes.
        Args:
            slopes (ndarray): array of slopes to reconstruct from
        Returns:
            ndarray: array to comands to be sent to DM
        """

        #Retreive pseudo on-axis slopes from tomo reconstructor
        slopes_HO = self.tomoRecon.dot(
                slopes[self.wfss[2].config.dataStart:])

        # Probably should remove TT from these slopes?
        nSubaps = slopes_HO.shape[0]
        slopes_HO[:nSubaps] -= slopes_HO[:nSubaps].mean()
        slopes_HO[nSubaps:] -= slopes_HO[nSubaps:].mean()

        # Final slopes are TT slopes appended to the tomographic High order slopes
        onSlopes = numpy.append(
                slopes[:self.wfss[1].config.dataStart], slopes_HO)

        dmCommands = self.control_matrix.T.dot(onSlopes)

        #
        # ttCommands = self.control_matrix[
        #         :self.wfss[1].config.dataStart,:2].T.dot(slopes_TT)
        #
        # hoCommands = self.control_matrix[
        #         self.wfss[1].config.dataStart:,2:].T.dot(slopes_HO)
        #
        # #if self.dms[0].dmConfig.type=="TT":
        #    ttMean = slopes.reshape(2, self.wfss[0].activeSubaps).mean(1)
        #    ttCommands = self.control_matrix[:,:2].T.dot(slopes)
        #    slopes[:self.wfss[0].activeSubaps] -= ttMean[0]
        #    slopes[self.wfss[0].activeSubaps:] -= ttMean[1]

        #    #get dm commands for the calculated on axis slopes
        #    dmCommands = self.control_matrix[:,2:].T.dot(slopes)

        #    return numpy.append(ttCommands, dmCommands)

        #get dm commands for the calculated on axis slopes

       # dmCommands = self.control_matrix.T.dot(slopes)

        return dmCommands



#####################################
#Experimental....
#####################################
class GLAO_4LGS(MVM):
    """
    Reconstructor of LGS TT prediction algorithm.

    Uses one TT DM and a high order DM. The TT WFS controls the TT DM and
    the second WFS controls the high order DM. The TT WFS and DM are
    assumed to be the first in the system.
    """


    def initControlMatrix(self):

        self.controlShape = (2*self.wfss[0].activeSubaps+2*self.wfss[1].activeSubaps,
                             self.sim_config.totalActs)
        self.controlMatrix = numpy.zeros( self.controlShape )


    def reconstruct(self, slopes):
        """
        Determine DM commands using previously made
        reconstructor from slopes.
        Args:
            slopes (ndarray): array of slopes to reconstruct from
        Returns:
            ndarray: array to commands to be sent to DM
        """

        offSlopes = slopes[self.wfss[2].config.dataStart:]
        meanOffSlopes = offSlopes.reshape(4,self.wfss[2].activeSubaps*2).mean(0)

        meanOffSlopes = self.removeCommonTT(meanOffSlopes, [1])

        slopes = numpy.append(
                slopes[:self.wfss[1].config.dataStart], meanOffSlopes)

        return super(LgsTT, self).reconstruct(slopes)


    def removeCommonTT(self, slopes, wfsList):

        xSlopesShape = numpy.array(slopes.shape)
        xSlopesShape[-1] /= 2.
        xSlopes = numpy.zeros(xSlopesShape)
        ySlopes = numpy.zeros(xSlopesShape)

        for i in range(len(wfsList)):
            wfs = wfsList[i]
            wfsSubaps = self.wfss[wfs].activeSubaps
            xSlopes[..., i*wfsSubaps:(i+1)*wfsSubaps] = slopes[..., i*2*wfsSubaps:i*2*wfsSubaps+wfsSubaps]
            ySlopes[..., i*wfsSubaps:(i+1)*wfsSubaps] = slopes[..., i*2*wfsSubaps+wfsSubaps:i*2*wfsSubaps+2*wfsSubaps]

        xSlopes = (xSlopes.T - xSlopes.mean(-1)).T
        ySlopes = (ySlopes.T - ySlopes.mean(-1)).T

        for i in range(len(wfsList)):
            wfs = wfsList[i]
            wfsSubaps = self.wfss[wfs].activeSubaps

            slopes[..., i*2*wfsSubaps:i*2*wfsSubaps+wfsSubaps] = xSlopes[..., i*wfsSubaps:(i+1)*wfsSubaps]
            slopes[..., i*2*wfsSubaps+wfsSubaps:i*2*wfsSubaps+2*wfsSubaps] = ySlopes[..., i*wfsSubaps:(i+1)*wfsSubaps]

        return slopes

class WooferTweeter(Reconstructor):
    '''
    Reconstructs a 2 DM system, where 1 DM is of low order, high stroke
    and the other has a higher, but low stroke.

    Reconstructs dm commands for each DM, then removes the low order
    component from the high order commands by propagating back to the
    slopes corresponding to the lower order DM shape, and propagating
    to the high order DM shape.
    '''

    def calcCMat(self,callback=None, progressCallback=None):
        '''
        Creates control Matrix.
        Assumes that DM 0  is low order,
        and DM 1 is high order.
        '''

        if self.sim_config.nDM==1:
            logger.warning("Woofer Tweeter Reconstruction not valid for 1 dm.")
            return None
        acts = 0
        dmCMats = []
        for dm in xrange(self.sim_config.nDM):
            dmIMat = self.dms[dm].iMat
            
            if self.dms[dm].dmConfig.svdConditioning == 'adaptive':
                rcond = get_rcond_adaptive_threshold_rank(dmIMat)
                self.dms[dm].dmConfig.svdConditioning = rcond
            
            logger.info("Invert DM {} IMat with conditioning:{}".format(dm,self.dms[dm].dmConfig.svdConditioning))
            if dmIMat.shape[0]==dmIMat.shape[1]:
                dmCMat = numpy.linalg.pinv(dmIMat)
            else:
                dmCMat = numpy.linalg.pinv(
                                    dmIMat, self.dms[dm].dmConfig.svdConditioning)

            #if dm != self.sim_config.nDM-1:
            #    self.controlMatrix[:,acts:acts+self.dms[dm].n_acts] = dmCMat
            #    acts+=self.dms[dm].n_acts

            dmCMats.append(dmCMat)


        self.control_matrix[:, 0:self.dms[0].n_acts]
        acts = self.dms[0].n_acts
        for dm in range(1, self.sim_config.nDM):

            #This is the matrix which converts from Low order DM commands
            #to high order DM commands, via slopes
            lowToHighTransform = self.dms[dm-1].iMat.T.dot( dmCMats[dm-1] )

            highOrderCMat = dmCMats[dm].T.dot(
                    numpy.identity(self.sim_config.totalWfsData)-lowToHighTransform)

            dmCMats[dm] = highOrderCMat

            self.control_matrix[:, acts:acts + self.dms[dm].n_acts] = highOrderCMat.T
            acts += self.dms[dm].n_acts


class LgsTT(LearnAndApply):
    """
    Reconstructor of LGS TT prediction algorithm.

    Uses one TT DM and a high order DM. The TT WFS controls the TT DM and
    the second WFS controls the high order DM. The TT WFS and DM are
    assumed to be the first in the system.
    """

    def initControlMatrix(self):

        self.controlShape = (2*self.wfss[0].activeSubaps+2*self.wfss[1].activeSubaps,
                             self.sim_config.totalActs)
        self.controlMatrix = numpy.zeros( self.controlShape )


    def calcCMat(self,callback=None, progressCallback=None):
        '''
        Uses the slopes recorded in the "learn" and DM interaction matrices
        to create a CMat.
        '''

        logger.info("Performing Learn....")
        self.learn(callback, progressCallback)
        logger.info("Done. Creating Tomographic Reconstructor...")

        if progressCallback!=None:
            progressCallback(1,1, "Calculating Covariance Matrices")

        #Need to remove all *common* TT from off-axis learn slopes
        self.learnSlopes[:, 2*self.wfss[1].activeSubaps:] = self.removeCommonTT(
                self.learnSlopes[:, 2*self.wfss[1].activeSubaps:], [2,3,4,5])

        self.covMat = numpy.cov(self.learnSlopes.T)
        Conoff = self.covMat[   :2*self.wfss[1].activeSubaps,
                                2*self.wfss[1].activeSubaps:     ]
        Coffoff = self.covMat[  2*self.wfss[1].activeSubaps:,
                                2*self.wfss[1].activeSubaps:    ]

        logger.info("Inverting offoff Covariance Matrix")
        iCoffoff = numpy.linalg.pinv(Coffoff)

        self.tomoRecon = Conoff.dot(iCoffoff)
        logger.info("Done. \nCreating full reconstructor....")

        super(LgsTT, self).calcCMat(callback, progressCallback)


    def reconstruct(self, slopes):
        """
        Determine DM commands using previously made
        reconstructor from slopes.
        Args:
            slopes (ndarray): array of slopes to reconstruct from
        Returns:
            ndarray: array to commands to be sent to DM
        """

        #Get off axis slopes and remove *common* TT
        offSlopes = slopes[self.wfss[2].config.dataStart:]
        offSlopes = self.removeCommonTT(offSlopes,[2,3,4,5])

        #Use the tomo matrix to get pseudo on-axis slopes
        psuedoOnSlopes = self.tomoRecon.dot(offSlopes)

        #Combine on-axis slopes with TT measurements
        slopes = numpy.append(
                slopes[:self.wfss[1].config.dataStart], psuedoOnSlopes)

        #Send to command matrices to get dmCommands
        return super(LgsTT, self).reconstruct(slopes)


    def removeCommonTT(self, slopes, wfsList):

        xSlopesShape = numpy.array(slopes.shape)
        xSlopesShape[-1] /= 2.
        xSlopes = numpy.zeros(xSlopesShape)
        ySlopes = numpy.zeros(xSlopesShape)

        for i in range(len(wfsList)):
            wfs = wfsList[i]
            wfsSubaps = self.wfss[wfs].activeSubaps
            xSlopes[..., i*wfsSubaps:(i+1)*wfsSubaps] = slopes[..., i*2*wfsSubaps:i*2*wfsSubaps+wfsSubaps]
            ySlopes[..., i*wfsSubaps:(i+1)*wfsSubaps] = slopes[..., i*2*wfsSubaps+wfsSubaps:i*2*wfsSubaps+2*wfsSubaps]

        xSlopes = (xSlopes.T - xSlopes.mean(-1)).T
        ySlopes = (ySlopes.T - ySlopes.mean(-1)).T

        for i in range(len(wfsList)):
            wfs = wfsList[i]
            wfsSubaps = self.wfss[wfs].activeSubaps

            slopes[..., i*2*wfsSubaps:i*2*wfsSubaps+wfsSubaps] = xSlopes[..., i*wfsSubaps:(i+1)*wfsSubaps]
            slopes[..., i*2*wfsSubaps+wfsSubaps:i*2*wfsSubaps+2*wfsSubaps] = ySlopes[..., i*wfsSubaps:(i+1)*wfsSubaps]

        return slopes

class ANN(Reconstructor):
    """
    Reconstructs using a neural net
    Assumes on axis slopes are WFS 0

    Net must be set by setting ``sim.recon.net = net`` before loop is run
    net object must have a ``run`` method, which accepts slopes and returns
    on Axis slopes
    """

    def calcCMat(self, callback=None, progressCallback=None):

        nSlopes = self.wfss[0].activeSubaps*2

        self.controlShape = (nSlopes, self.sim_config.totalActs)
        self.controlMatrix = numpy.zeros((nSlopes, self.sim_config.totalActs))
        acts = 0
        for dm in xrange(self.sim_config.nDM):
            dmIMat = self.dms[dm].iMat

            if dmIMat.shape[0]==dmIMat.shape[1]:
                dmCMat = numpy.inv(dmIMat)
            else:
                dmCMat = numpy.linalg.pinv(dmIMat, self.dmConds[dm])

            self.controlMatrix[:,acts:acts+self.dms[dm].n_acts] = dmCMat
            acts += self.dms[dm].n_acts

    def reconstruct(self, slopes):
        """
        Determine DM commands using previously made
        reconstructor from slopes. Uses Artificial Neural Network.

        Slopes are normalised before being run through the network.

        Args:
            slopes (ndarray): array of slopes to reconstruct from
        Returns:
            ndarray: array to comands to be sent to DM
        """
        t=time.time()
        offSlopes = slopes[self.wfss[0].activeSubaps*2:]/7 # normalise
        onSlopes = self.net.run(offSlopes)*7 # un-normalise
        dmCommands = self.controlMatrix.T.dot(onSlopes)

        self.Trecon += time.time()-t
        return dmCommands

def get_rcond_adaptive_threshold_rank(A,plot=False,return_place=False,threshold_rank=0.5):
  _,s,_ = numpy.linalg.svd(A)
  s /= s.max()
  energy = s**2
  energy /= energy.sum()
  accumulated_energy = numpy.zeros_like(energy)
  threshold = threshold_rank/numpy.linalg.matrix_rank(A)

  for i in numpy.arange(energy.shape[0]):
    accumulated_energy[i] = energy[:i].sum()
  residual = 1 - accumulated_energy
  place = numpy.where(residual<=threshold)[0][0]
  
  if plot==True:
    # plt.plot(residual,label='residual')
    # plt.hlines(residual[place],0,s.shape[0])
    # plt.vlines(place,0,1)
    # plt.yscale('log')
    # plt.show()

    plt.plot(s)
    plt.hlines(s[place],0,s.shape[0])
    plt.vlines(place,0,1)
    plt.yscale('log')
    plt.show()
  rcond = s[place]
  if return_place == True:
    return rcond, place
  else:
    return rcond
#   return 0.05

def pinv_adaptive_threshold_rank(A,return_rcond=False):
    if return_rcond == False:
        rcond = get_rcond_adaptive_threshold_rank(A)
        pinvA = numpy.linalg.pinv(A,rcond=rcond)
        return pinvA
    else:
        rcond, place = get_rcond_adaptive_threshold_rank(A,return_place=True)
        pinvA = numpy.linalg.pinv(A,rcond=rcond)
        return pinvA, rcond, place

# def make_quiver_plot(wfs):
#     position = wfs.detector_cent_coords
#     N = position.shape[0]
    
#     step = (position[N//2 + 1,1]
#             - position[N//2,1])
#     position = position // step
    
#     slopex = wfs.slopes[:N]
#     slopey = wfs.slopes[N:]
#     plt.quiver(rearrange1(slopex, position),
#                 rearrange1(slopey, position),
#                 scale=5, scale_units='inches')
#     plt.axis('square')
#     plt.show()
#     return

def make_quiver_plot(position, slopes):
    #position = wfs.detector_cent_coords
    N = position.shape[0]
    
    step = (position[N//2 + 1,1]
            - position[N//2,1])
    position = position // step
    
    max_position = position.max()
    x = numpy.arange(max_position + 1.)
    x -= max_position/2.
    yy, xx = numpy.meshgrid(x,x)
    
    slopex = slopes[:N]#wfs.slopes[:N]
    slopey = slopes[N:]#wfs.slopes[N:]
    plt.quiver(xx,-yy,
               rearrange1(slopex, position).T,
               rearrange1(-slopey, position).T,
               scale=5, scale_units='inches')
    # plt.title('IM Slope')
    plt.axis('square')
    plt.show()
    return

def rearrange1(A, position):
    """
    rearrange soapy reported wfs value from 1d with skips into 2d

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    position : TYPE
        DESCRIPTION.

    Returns
    -------
    a : TYPE
        DESCRIPTION.

    """
    N = position.shape[0]
    size = numpy.max(position) - numpy.min(position) + 1
    DIM = position.shape[1]
    a = numpy.zeros((size,)*DIM, dtype=float)
    for n in numpy.arange(N):
        a[tuple(position[n])] = A[n]
    return a

def parabola2d(xy,x0,y0,ax,ay,c):
    x, y = xy
    z = ax*(x - x0)**2 + ay*(y - y0)**2 + c
    return z

def find_centre(corr,nx_subap_interp):
    # M. G. Löfdahl 2010
    try:
        x = numpy.arange(corr.shape[0],dtype=float)
        x -= x.max()/2.
        x -= 0.5
        x /= nx_subap_interp
        yy,xx = numpy.meshgrid(x,x)
        
        maxindex = numpy.array(numpy.where(corr==numpy.nanmax(corr)))[:,0]
        

        
        cropped_corr = corr[maxindex[0] - 1 
                            : maxindex[0] + 2,
                            maxindex[1] - 1 
                            : maxindex[1] + 2]
        
        locx = xx[maxindex[0],maxindex[1]]
        locy = yy[maxindex[0],maxindex[1]]
        
        a2 = (cropped_corr[1 + 1,:].mean() - cropped_corr[-1 + 1,:].mean())/2.
        a3 = (cropped_corr[1 + 1,:].mean() - 2.*cropped_corr[0 + 1,:].mean() + cropped_corr[-1 + 1,:].mean())/2.
        a4 = (cropped_corr[:,1 + 1].mean() - cropped_corr[:,-1 + 1].mean())/2.
        a5 = (cropped_corr[:,1 + 1].mean() - 2.*cropped_corr[:,0 + 1].mean() + cropped_corr[:,-1 + 1].mean())/2.
        a6 = (cropped_corr[1 + 1,1 + 1] - cropped_corr[-1 + 1,1 + 1] - cropped_corr[1 + 1,-1 + 1] + cropped_corr[-1 + 1,-1 + 1])/4.
        
        true_locx = ((2.*a2*a5 - a4*a6)/(a6**2 - 4.*a3*a5))
        true_locy = ((2.*a3*a4 - a2*a6)/(a6**2 - 4.*a3*a5))
        
        # plt.pcolor([-1,0,1],[-1,0,1],cropped_corr.T)
        # plt.plot(true_locx,true_locy,marker='o',color='r')
        # plt.axis('square')
        # plt.show()
        
        true_locx /= nx_subap_interp
        true_locy /= nx_subap_interp
        
        true_locx += locx
        true_locy += locy
    
    except:
        return numpy.nan, numpy.nan
    
    return true_locx, true_locy


# def rearrange1(A, position):
#     """
#     rearrange soapy reported wfs value from 1d with skips into 2d

#     Parameters
#     ----------
#     A : TYPE
#         DESCRIPTION.
#     position : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     a : TYPE
#         DESCRIPTION.

#     """
#     N = position.shape[0]
#     size = numpy.max(position) - numpy.min(position) + 1
#     DIM = position.shape[1]
#     a = numpy.zeros((size,)*DIM, dtype=float)
#     for n in numpy.arange(N):
#         a[tuple(position[n])] = A[n]
#     a[a==0] = numpy.nan
#     return a

def measure_AS_from_IM(IM,SHWFS_position,ACT_list,ACT_position,threshold, debug=False):
    # debug= False
    position = SHWFS_position

    ACT_LIST = ACT_list

    DM_POS_TEMP = ACT_position

    nACT = IM.shape[0]
    nSubap = IM.shape[1]

    DM_POS = numpy.zeros([nACT, 2],dtype=float)
    
    j = 0
    for i in numpy.arange(nACT):
        if (i==ACT_LIST).any():
            DM_POS[i,0] = DM_POS_TEMP[j,1]
            DM_POS[i,1] = DM_POS_TEMP[j,0]
            j += 1
        else:
            DM_POS[i,:] = numpy.nan
            
    # DM_GRID = numpy.arange(DM_POS_TEMP.min(),DM_POS_TEMP.max() + 1)
    WFS_GRID = numpy.arange(position.min(),position.max() + 1)
    WFS_GRID_X,WFS_GRID_Y = numpy.meshgrid(WFS_GRID,WFS_GRID)

    CEN = numpy.zeros((nACT,2),dtype=float)
    CEN[:] = numpy.nan

    for act in ACT_LIST:#numpy.arange(83):

        
        IM /= numpy.abs(numpy.nanmax(IM))
        
        #for act in ACT_LIST:
    
        # act = 42
        WFS_MAP_X = rearrange1(IM[act][:nSubap//2], position)
        WFS_MAP_Y = rearrange1(IM[act][nSubap//2:], position)
        
        # cen_x = numpy.nansum(WFS_GRID_X*WFS_MAP_X**2)/numpy.nansum(WFS_GRID_X*WFS_MAP_X**2/WFS_GRID_X)
        # cen_y = numpy.nansum(WFS_GRID_Y*WFS_MAP_Y**2)/numpy.nansum(WFS_GRID_Y*WFS_MAP_Y**2/WFS_GRID_Y)
        
        # plt.clf()
        # fig,axes = plt.subplots(1,2,sharey=True)
        # axes[0].imshow(WFS_MAP_X,vmin=-1,vmax=1,cmap='RdBu')
        # CS = axes[1].imshow(WFS_MAP_Y,vmin=-1,vmax=1,cmap='RdBu')
        # fig.tight_layout()
        # axes[0].set_title('x_slope')
        # axes[1].set_title('y_slope')
        # cbar_ax = fig.add_axes([1,0.17,0.05,0.65])
        # axes[0].plot(DM_POS[act,0],DM_POS[act,1],marker='o',color='white')
        # axes[1].plot(DM_POS[act,0],DM_POS[act,1],marker='o',color='white')
        # fig.colorbar(CS,cax=cbar_ax)
        # fig.suptitle('act#{}'.format(act))
        # plt.show()
        
        # plt.clf()
        # # plt.plot(DM_POS[act,0],DM_POS[act,1],color='r',marker='o')
        # plt.quiver(WFS_GRID_X,WFS_GRID_Y,
        #            WFS_MAP_X,WFS_MAP_Y,scale=5)
        
        WFS_MAP = (WFS_MAP_X**2 + WFS_MAP_Y**2)**0.5
        
        WFS_MAP[WFS_MAP < threshold] = numpy.nan
        WFS_MAP_X[WFS_MAP < threshold] = numpy.nan
        WFS_MAP_Y[WFS_MAP < threshold] = numpy.nan
        
        MAX4 = numpy.sort(WFS_MAP[~numpy.isnan(WFS_MAP)].flatten())[-4:]
        
        WFS_MAP_X_MAX4 = numpy.zeros_like(WFS_MAP)
        WFS_MAP_Y_MAX4 = numpy.zeros_like(WFS_MAP)
        WFS_MAP_X_MAX4[:] = numpy.nan
        WFS_MAP_Y_MAX4[:] = numpy.nan
        
        
        
        MAX4_POS = numpy.zeros((MAX4.shape[0],2),dtype=int)
        v_x = numpy.zeros((MAX4.shape[0]),dtype=float)
        v_y = numpy.zeros((MAX4.shape[0]),dtype=float)
        
        for i in numpy.arange(MAX4.shape[0]):
            loc = numpy.asarray(numpy.where(WFS_MAP == MAX4[i]))[:,0]
            MAX4_POS[i] = loc
            WFS_MAP_X_MAX4[tuple(MAX4_POS[i])] = WFS_MAP_X[tuple(MAX4_POS[i])]
            WFS_MAP_Y_MAX4[tuple(MAX4_POS[i])] = WFS_MAP_Y[tuple(MAX4_POS[i])]
            v_x[i] = WFS_MAP_X_MAX4[tuple(MAX4_POS[i])]
            v_y[i] = WFS_MAP_Y_MAX4[tuple(MAX4_POS[i])]
            
        if debug:
            plt.quiver(WFS_GRID_X,WFS_GRID_Y,
                       WFS_MAP_X_MAX4,WFS_MAP_Y_MAX4,scale=5,color='r')
            
            x_standard = numpy.linspace(WFS_GRID.min(),WFS_GRID.max())
            x = numpy.zeros((MAX4.shape[0],x_standard.shape[0]))
            y = numpy.zeros((MAX4.shape[0],x_standard.shape[0]),dtype=float)
            for i in numpy.arange(MAX4.shape[0]):
                m = v_x[i]/v_y[i]
                x[i] = x_standard
                y[i] = m*(x[i] - MAX4_POS[i][0]) + MAX4_POS[i][1]
                
                # y[i] = (x_standard*v_x[i] + MAX4_POS[i][1])
                # x[i] = (x_standard*v_y[i] + MAX4_POS[i][0])
            
                # plt.plot(y[i],x[i],ls=':',color='b',marker='',alpha=0.5)
        
        cross = numpy.zeros((MAX4.shape[0],MAX4.shape[0],2),dtype=float)
        cross[:] = numpy.nan
        trustability = numpy.zeros((MAX4.shape[0],MAX4.shape[0]),dtype=float)
        
        for i in numpy.arange(MAX4.shape[0]):
            for j in numpy.arange(i + 1,MAX4.shape[0]):
                mi = (v_y[i]/v_x[i])**-1
                mj = (v_y[j]/v_x[j])**-1
                xi = MAX4_POS[i][0]#x[i]
                xj = MAX4_POS[j][0]#x[j]
                yi = MAX4_POS[i][1]#y[i]
                yj = MAX4_POS[j][1]#y[j]
                
                cross[i,j,1] = ((yi - yj) + (mj*xj - mi*xi))/(mj - mi)
                cross[i,j,0] = yi + mi*(cross[i,j,1] - xi)
                
                if (
                        
                        (not (((yi - cross[i,j,0])/v_x[i] >= 0)
                              and ((xi - cross[i,j,1])/v_y[i] >= 0)
                              and ((yj - cross[i,j,0])/v_x[j] >= 0)
                              and ((xj - cross[i,j,1])/v_y[j] >= 0)))
                        
                        or
                        
                        ((cross[i,j,0] < (WFS_GRID.min() - 0.5))
                         or (cross[i,j,0] > (WFS_GRID.max() + 0.5))
                         or (cross[i,j,1] < (WFS_GRID.min() - 0.5))
                         or (cross[i,j,1] > (WFS_GRID.max() + 0.5)))
                        
                        ):

                    cross[i,j,0] = numpy.nan
                    cross[i,j,1] = numpy.nan
                    
                else:
                    if debug:
                        plt.plot(cross[i,j,0],cross[i,j,1],marker='o',color='g')
                        
                        plt.plot(y[i],x[i],ls=':',color='b',marker='')
                        plt.plot(y[j],x[j],ls=':',color='b',marker='')
                        plt.quiver(WFS_GRID_X,WFS_GRID_Y,
                                   WFS_MAP_X_MAX4,WFS_MAP_Y_MAX4,scale=5)
                        plt.plot(DM_POS[act,0],DM_POS[act,1],color='r',marker='o')
                        # plt.show()
                        # plt.plot(cen_x,cen_y,marker='x',color='b')
                        # plt.axis('square')
                        # plt.xlim(-0.5,7.5)
                        # plt.ylim(-0.5,7.5)
                        # plt.title(name_list[i_aber_z_list] + '\n'
                        #     'act#{}'.format(act)
                        #     +'\n{}{}'.format(i,j))
                        # plt.show()
                    size = ((WFS_MAP_X[tuple(MAX4_POS[i])]**2
                             + WFS_MAP_Y[tuple(MAX4_POS[i])]**2)**0.5
                            * (WFS_MAP_X[tuple(MAX4_POS[j])]**2
                               + WFS_MAP_Y[tuple(MAX4_POS[j])]**2)**0.5)
                    trustability[i,j] = size#**2
        
        cen_x = numpy.nansum(cross[...,0]*trustability)/numpy.nansum(trustability)
        cen_y = numpy.nansum(cross[...,1]*trustability)/numpy.nansum(trustability)
        
        CEN[act] = [cen_x - DM_POS[act,0], cen_y - DM_POS[act,1]]
        if debug:
            plt.plot(cen_x,cen_y,marker='o',color='b')
            plt.axis('square')
            plt.xlim(-0.5,7.5)
            plt.ylim(-0.5,7.5)
            # plt.title(name_list[i_aber_z_list] + '\n'
            #     'act#{}'.format(act))
            plt.show()

    return CEN

def measure_AS_from_poke(slope,SHWFS_position,ACT_position,threshold,debug=False):
    # debug= True
    position = SHWFS_position

    nSubap = slope.shape[0]

    DM_POS = ACT_position[0][0][::-1]
    
    
            
    # DM_GRID = numpy.arange(DM_POS_TEMP.min(),DM_POS_TEMP.max() + 1)
    WFS_GRID = numpy.arange(position.min(),position.max() + 1)
    WFS_GRID_X,WFS_GRID_Y = numpy.meshgrid(WFS_GRID,WFS_GRID)

    CEN = numpy.zeros((2),dtype=float)
    CEN[:] = numpy.nan

    

        
    # slope /= numpy.abs(numpy.nanmax(slope))
    # print(position.shape)
    # print(slope.shape)

    WFS_MAP_X = rearrange1(slope[:nSubap//2], position)
    WFS_MAP_Y = rearrange1(slope[nSubap//2:], position)
    

    
    WFS_MAP = (WFS_MAP_X**2 + WFS_MAP_Y**2)**0.5
    
    
    WFS_MAP_X /= numpy.nanmax(WFS_MAP)
    WFS_MAP_Y /= numpy.nanmax(WFS_MAP)
    WFS_MAP /= numpy.nanmax(WFS_MAP)
    
    
    WFS_MAP[WFS_MAP < threshold] = numpy.nan
    WFS_MAP_X[WFS_MAP < threshold] = numpy.nan
    WFS_MAP_Y[WFS_MAP < threshold] = numpy.nan
    
    MAX4 = numpy.sort(WFS_MAP[~numpy.isnan(WFS_MAP)].flatten())[-4:]
    
    WFS_MAP_X_MAX4 = numpy.zeros_like(WFS_MAP)
    WFS_MAP_Y_MAX4 = numpy.zeros_like(WFS_MAP)
    WFS_MAP_X_MAX4[:] = numpy.nan
    WFS_MAP_Y_MAX4[:] = numpy.nan
    
    if debug:
        recenter = SHWFS_position.max()/2
    
    MAX4_POS = numpy.zeros((MAX4.shape[0],2),dtype=int)
    v_x = numpy.zeros((MAX4.shape[0]),dtype=float)
    v_y = numpy.zeros((MAX4.shape[0]),dtype=float)
    
    for i in numpy.arange(MAX4.shape[0]):
        loc = numpy.asarray(numpy.where(WFS_MAP == MAX4[i]))[:,0]
        MAX4_POS[i] = loc
        WFS_MAP_X_MAX4[tuple(MAX4_POS[i])] = WFS_MAP_X[tuple(MAX4_POS[i])]
        WFS_MAP_Y_MAX4[tuple(MAX4_POS[i])] = WFS_MAP_Y[tuple(MAX4_POS[i])]
        v_x[i] = WFS_MAP_X_MAX4[tuple(MAX4_POS[i])]
        v_y[i] = WFS_MAP_Y_MAX4[tuple(MAX4_POS[i])]
        
    if debug:
        plt.quiver(WFS_GRID_X - recenter,WFS_GRID_Y - recenter,
                   WFS_MAP_X,WFS_MAP_Y,scale=5,color='k',headwidth=5,width=0.01)#
        
        x_standard = numpy.linspace(WFS_GRID.min(),WFS_GRID.max())
        x = numpy.zeros((MAX4.shape[0],x_standard.shape[0]))
        y = numpy.zeros((MAX4.shape[0],x_standard.shape[0]),dtype=float)
        for i in numpy.arange(MAX4.shape[0]):
            m = v_x[i]/v_y[i]
            x[i] = x_standard
            y[i] = m*(x[i] - MAX4_POS[i][0]) + MAX4_POS[i][1]
            
            # y[i] = (x_standard*v_x[i] + MAX4_POS[i][1])
            # x[i] = (x_standard*v_y[i] + MAX4_POS[i][0])
        
            # plt.plot(y[i],x[i],ls=':',color='b',marker='',alpha=0.5)
    
    cross = numpy.zeros((MAX4.shape[0],MAX4.shape[0],2),dtype=float)
    cross[:] = numpy.nan
    trustability = numpy.zeros((MAX4.shape[0],MAX4.shape[0]),dtype=float)
    
    for i in numpy.arange(MAX4.shape[0]):
        for j in numpy.arange(i + 1,MAX4.shape[0]):
            mi = (v_y[i]/v_x[i])**-1
            mj = (v_y[j]/v_x[j])**-1
            xi = MAX4_POS[i][0]#x[i]
            xj = MAX4_POS[j][0]#x[j]
            yi = MAX4_POS[i][1]#y[i]
            yj = MAX4_POS[j][1]#y[j]
            
            cross[i,j,1] = ((yi - yj) + (mj*xj - mi*xi))/(mj - mi)
            cross[i,j,0] = yi + mi*(cross[i,j,1] - xi)
            
            if (
                    
                    (not (((yi - cross[i,j,0])/v_x[i] >= 0)
                          and ((xi - cross[i,j,1])/v_y[i] >= 0)
                          and ((yj - cross[i,j,0])/v_x[j] >= 0)
                          and ((xj - cross[i,j,1])/v_y[j] >= 0)))
                    
                    or
                    
                    ((cross[i,j,0] < (WFS_GRID.min() - 0.5))
                     or (cross[i,j,0] > (WFS_GRID.max() + 0.5))
                     or (cross[i,j,1] < (WFS_GRID.min() - 0.5))
                     or (cross[i,j,1] > (WFS_GRID.max() + 0.5)))
                    
                    ):

                cross[i,j,0] = numpy.nan
                cross[i,j,1] = numpy.nan
                
            else:
                if debug:
                    # plt.plot(cross[i,j,0] - recenter,cross[i,j,1] - recenter,marker='s',color='tab:orange',label='cross spots',markersize=5)
                    
                    # plt.plot(y[i] - recenter,x[i] - recenter,ls=':',color='b',marker='')
                    # plt.plot(y[j] - recenter,x[j] - recenter,ls=':',color='b',marker='')
                    plt.quiver(WFS_GRID_X - recenter,WFS_GRID_Y - recenter,
                               WFS_MAP_X_MAX4,WFS_MAP_Y_MAX4,scale=5,color='tab:red',headwidth=5,width=0.01)
                    # print(DM_POS)
                    
                    # plt.show()
                    # plt.plot(cen_x,cen_y,marker='x',color='b')
                    # plt.axis('square')
                    # plt.xlim(-0.5,7.5)
                    # plt.ylim(-0.5,7.5)
                    # plt.title(name_list[i_aber_z_list] + '\n'
                    #     'act#{}'.format(act)
                    #     +'\n{}{}'.format(i,j))
                    # plt.show()
                size = ((WFS_MAP_X[tuple(MAX4_POS[i])]**2
                         + WFS_MAP_Y[tuple(MAX4_POS[i])]**2)**0.5
                        * (WFS_MAP_X[tuple(MAX4_POS[j])]**2
                           + WFS_MAP_Y[tuple(MAX4_POS[j])]**2)**0.5)
                trustability[i,j] = size#**2
    
    cen_x = numpy.nansum(cross[...,0]*trustability)/numpy.nansum(trustability)
    cen_y = numpy.nansum(cross[...,1]*trustability)/numpy.nansum(trustability)
    
    CEN = [cen_x - DM_POS[0], cen_y - DM_POS[1]]
    if debug:
        plt.plot(cen_x - recenter,cen_y - recenter,marker='.',color='tab:red',ls='',label='apparent position : slope',markersize=10)
        # plt.plot(DM_POS[0] - recenter,DM_POS[1] - recenter,color='white',marker='o',label='actuator true location')
        # plt.axis('square')
        # plt.xlim(-0.5 - recenter,7.5 - recenter)
        # plt.ylim(-0.5 - recenter,7.5 - recenter)
        # plt.title(name_list[i_aber_z_list] + '\n'
        #     'act#{}'.format(act))
        # plt.show()

    return CEN