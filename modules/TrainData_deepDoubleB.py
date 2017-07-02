
from TrainData import TrainData,fileTimeOut
from TrainData_deepAK8 import TrainData_AK8Jet
import numpy

        
class TrainData_DoubleB_init(TrainData_AK8Jet):
    
    def __init__(self):
        '''
        This is an example data format description for FatJet studies
        '''
        TrainData_AK8Jet.__init__(self)
        
        #example of how to register global branches (for characterizing output)
        self.addBranches(['fj_pt',
                          'fj_eta',
                          'fj_sdmass',
                          'fj_n_sdsubjets',
                          'fj_doubleb',
                          'fj_tau21',
                          'fj_tau32',
                          'npv',
                          'npfcands',
                          'ntracks',
                          'nsv',
                          ])
                          
        #example of how to register global branches (for training inputs)
        self.addBranches(['fj_jetNTracks',
                          'fj_nSV',
                          'fj_tau0_trackEtaRel_0',
                          'fj_tau0_trackEtaRel_1',
                          'fj_tau0_trackEtaRel_2',
                          'fj_tau1_trackEtaRel_0',
                          'fj_tau1_trackEtaRel_1',
                          'fj_tau1_trackEtaRel_2',
                          'fj_tau_flightDistance2dSig_0',
                          'fj_tau_flightDistance2dSig_1',
                          'fj_tau_vertexDeltaR_0',
                          'fj_tau_vertexEnergyRatio_0',
                          'fj_tau_vertexEnergyRatio_1',
                          'fj_tau_vertexMass_0',
                          'fj_tau_vertexMass_1',
                          'fj_trackSip2dSigAboveBottom_0',
                          'fj_trackSip2dSigAboveBottom_1',
                          'fj_trackSip2dSigAboveCharm_0',
                          'fj_trackSipdSig_0',
                          'fj_trackSipdSig_0_0',
                          'fj_trackSipdSig_0_1',
                          'fj_trackSipdSig_1',
                          'fj_trackSipdSig_1_0',
                          'fj_trackSipdSig_1_1',
                          'fj_trackSipdSig_2',
                          'fj_trackSipdSig_3',
                          'fj_z_ratio'
                          ])

        
        #branches that are used directly in the following function 'readFromRootFile'
        #this is a technical trick to speed up the conversion
        #self.registerBranches(['Cpfcan_erel','Cpfcan_eta','Cpfcan_phi',
        #                       'Npfcan_erel','Npfcan_eta','Npfcan_phi',
        #                       'nCpfcand','nNpfcand',
        #                       'jet_eta','jet_phi'])
        
        
    #this function describes how the branches are converted
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        #the first part is standard, no changes needed
        from preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        import ROOT
        
        fileTimeOut(filename,120) #give eos 2 minutes to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples=tree.GetEntries()
        
        #the definition of what to do with the branches
        
        # those are the global branches (jet pt etc)
        # they should be just glued to each other in one vector
        # and zero padded (and mean subtracted and normalised)
        #x_global = MeanNormZeroPad(filename,TupleMeanStd,
        #                           [self.branches[0]],
        #                           [self.branchcutoffs[0]],self.nsamples)
        
        # the second part (the pf candidates) should be treated particle wise
        # an array with (njets, nparticles, nproperties) is created
    
        x_glb  = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                          self.branches[0],
                                          self.branchcutoffs[0],self.nsamples)
        
        x_input  = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                          self.branches[1],
                                          self.branchcutoffs[1],self.nsamples)

        # maybe also an image of the energy density of charged particles 
        # should be added
        #x_chmap = createDensityMap(filename,TupleMeanStd,
        #                           'Cpfcan_erel', #use the energy to create the image
        #                           self.nsamples,
        #                           # 7 bins in eta with a total width of 2*0.9
        #                           ['Cpfcan_eta','jet_eta',7,0.9], 
        #                           # 7 bins in phi with a total width of 2*0.9
        #                           ['Cpfcan_phi','jet_phi',7,0.9],
        #                           'nCpfcand',
                                   # the last is an offset because the relative energy as 
                                   # can be found in the ntuples is shifted by 1
        #                           -1)
        
        
        # now, some jets are removed to avoid pt and eta biases
        
        Tuple = self.readTreeFromRootToTuple(filename)
        if self.remove:
            # jets are removed until the shapes in eta and pt are the same as
            # the truth class 'fj_isLight'
            notremoves=weighter.createNotRemoveIndices(Tuple)
            #undef=Tuple[self.undefTruth]
            #notremoves-=undef
        
        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
            
            
        # create all collections:
        truthtuple =  Tuple[self.truthclasses]
        alltruth=self.reduceTruth(truthtuple)
        
        # remove the entries to get same jet shapes
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_glb=x_glb[notremoves > 0]            
            x_input=x_input[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            #x_global=x_global[notremoves > 0]
            #x_chmap=x_chmap[notremoves > 0]        
        
        #newnsamp=x_global.shape[0]
        newnsamp=x_glb.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        # fill everything
        self.w=[weights]
        self.x=[x_input]
        self.y=[alltruth]
        

    ## categories to use for training     
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses=['fj_isLight','fj_isBB']
        if tuple_in is not None:
            q = tuple_in['fj_isLight'].view(numpy.ndarray)
            h = tuple_in['fj_isH'].view(numpy.ndarray) 
            
            return numpy.vstack((q,h)).transpose()  
