
from TrainData import TrainData,fileTimeOut
import numpy


class TrainData_deepDoubleB(TrainData):
    
    def __init__(self):
        '''
        This class is meant as a base class for the FatJet studies
        You will not need to edit it for trying out things
        '''
        TrainData.__init__(self)
        
        #define truth:
        self.undefTruth=['isUndefined']
        self.truthclasses=['fj_isNonBB','fj_isBB']
        self.referenceclass='fj_isNonBB' ## used for pt reshaping

        self.registerBranches(self.truthclasses)
        self.registerBranches(['fj_pt','fj_sdmass'])

        self.weightbranchX='fj_pt'
        self.weightbranchY='fj_sdmass'

        #self.weight_binX = numpy.array([
        #        300,400,500,
        #        600,700,800,1000,2500],dtype=float)

        self.weight_binX = numpy.array([
                300,2500],dtype=float)

        self.weight_binY = numpy.array(
            [40,200],
            dtype=float
            )


        
        #this is only needed because the truth definitions are different from deepFlavour
        self.allbranchestoberead=[]
        self.registerBranches(self.undefTruth)
        self.registerBranches(self.truthclasses)
        
        
    ## categories to use for training     
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses=['fj_isNonBB','fj_isBB']
        if tuple_in is not None:
            q = tuple_in['fj_isNonBB'].view(numpy.ndarray)
            #w = tuple_in['fj_isTop'].view(numpy.ndarray)
            #z = tuple_in['fj_isZ'].view(numpy.ndarray)
            h = tuple_in['fj_isBB'].view(numpy.ndarray)
            #t = tuple_in['fj_isW'].view(numpy.ndarray)
            
            return numpy.vstack((q,h)).transpose()  
        
        
        
#######################################
        
        
class TrainData_deepDoubleB_init(TrainData_deepDoubleB):
    
    def __init__(self):
        '''
        This is an example data format description for FatJet studies
        '''
        TrainData_deepDoubleB.__init__(self)
        
        #example of how to register global branches
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
                          'nsv'
                      ])
        
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
                          'fj_z_ratio',
                          ]) 

        self.addBranches(['sv_ptrel',
                          'sv_erel',
                          'sv_phirel',
                          'sv_etarel',
                          'sv_deltaR',
                          'sv_pt',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_normchi2',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3d',
                          'sv_d3dsig',
                          'sv_costhetasvpv'
                         ],
                         5)

        #branches that are used directly in the following function 'readFromRootFile'
        #this is a technical trick to speed up the conversion
        #self.registerBranches(['Cpfcan_erel','Cpfcan_eta','Cpfcan_phi',
        #                       'Npfcan_erel','Npfcan_eta','Npfcan_phi',
        #                       'nCpfcand','nNpfcand',
        #                       'jet_eta','jet_phi'])
        
        
    #this function describes how the branches are converted
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        #the first part is standard, no changes needed
        from preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles, ZeroPadParticles
        import numpy
        import ROOT
        
        fileTimeOut(filename,120) #give eos 2 minutes to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples=tree.GetEntries()
    
        x_glb  = ZeroPadParticles(filename,TupleMeanStd,
                                          self.branches[0],
                                          self.branchcutoffs[0],self.nsamples)

        x_db  = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                         self.branches[1],
                                         self.branchcutoffs[1],self.nsamples)

        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                        self.branches[2],
                                        self.branchcutoffs[2],self.nsamples)
        
        # now, some jets are removed to avoid pt and eta biases
        
        Tuple = self.readTreeFromRootToTuple(filename)
        if self.remove:
            # jets are removed until the shapes in eta and pt are the same as
            # the truth class 'fj_isNonBB'
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
	print(alltruth)
        
        # remove the entries to get same jet shapes
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_glb=x_glb[notremoves > 0]
            x_db=x_db[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]  
        
        newnsamp=x_glb.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        # fill everything
        self.w=[weights]
        self.x=[x_db,x_sv]
        self.z=[x_glb]
        self.y=[alltruth]
	print("self:")
	print(self.y)

    
        
class TrainData_deepDoubleB_simple(TrainData_deepDoubleB):
    
    def __init__(self):
        '''
        This is an example data format description for FatJet studies
        '''
        TrainData_deepDoubleB.__init__(self)

        self.remove = True
        self.weight = False
        #example of how to register global branches
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
                          'nsv'
                      ])
        
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
                          'fj_z_ratio',
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
        from preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles, ZeroPadParticles
        import numpy
        import ROOT
        
        fileTimeOut(filename,120) #give eos 2 minutes to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples=tree.GetEntries()
    
        x_glb  = ZeroPadParticles(filename,TupleMeanStd,
                                          self.branches[0],
                                          self.branchcutoffs[0],self.nsamples)

        x_db  = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                         self.branches[1],
                                         self.branchcutoffs[1],self.nsamples)
        
        # now, some jets are removed to avoid pt and eta biases
        
        Tuple = self.readTreeFromRootToTuple(filename)
        if self.remove:
            # jets are removed until the shapes in eta and pt are the same as
            # the truth class 'fj_isNonBB'
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
	print(alltruth)
        
        # remove the entries to get same jet shapes
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_glb=x_glb[notremoves > 0]
            x_db=x_db[notremoves > 0]
            alltruth=alltruth[notremoves > 0]  
        
        newnsamp=x_glb.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        # fill everything
        self.w=[weights]
        self.x=[x_db]
        self.z=[x_glb]
        self.y=[alltruth]
	print("self:")
	print(self.y)
        
    
class TrainData_deepDoubleB_full(TrainData_deepDoubleB):
    
    def __init__(self):
        '''
        This is an example data format description for FatJet studies
        '''
        TrainData_deepDoubleB.__init__(self)
        
        #example of how to register global branches
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
                          'nsv'
                      ])
        
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
        
        #example of pf candidate branches
        self.addBranches(['pfcand_ptrel',
                          'pfcand_erel',
                          'pfcand_phirel',
                          'pfcand_etarel',
                          'pfcand_deltaR',
                          'pfcand_puppiw',
                          'pfcand_drminsv',
                          'pfcand_drsubjet1',
                          'pfcand_drsubjet2',
                          'pfcand_hcalFrac'
                         ],
                         100) 

        self.addBranches(['track_ptrel',     
                          'track_erel',     
                          'track_phirel',     
                          'track_etarel',     
                          'track_deltaR',
                          'track_drminsv',     
                          'track_drsubjet1',     
                          'track_drsubjet2',
                          'track_dz',     
                          'track_dzsig',     
                          'track_dxy',     
                          'track_dxysig',     
                          'track_normchi2',     
                          'track_quality',     
                          'track_dptdpt',     
                          'track_detadeta',     
                          'track_dphidphi',     
                          'track_dxydxy',     
                          'track_dzdz',     
                          'track_dxydz',     
                          'track_dphidxy',     
                          'track_dlambdadz',     
                          'trackBTag_EtaRel',     
                          'trackBTag_PtRatio',     
                          'trackBTag_PParRatio',     
                          'trackBTag_Sip2dVal',     
                          'trackBTag_Sip2dSig',     
                          'trackBTag_Sip3dVal',     
                          'trackBTag_Sip3dSig',     
                          'trackBTag_JetDistVal'
                         ],
                         60) 
        
        self.addBranches(['sv_ptrel',
                          'sv_erel',
                          'sv_phirel',
                          'sv_etarel',
                          'sv_deltaR',
                          'sv_pt',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_normchi2',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3d',
                          'sv_d3dsig',
                          'sv_costhetasvpv'
                         ],
                         5)

        #branches that are used directly in the following function 'readFromRootFile'
        #this is a technical trick to speed up the conversion
        #self.registerBranches(['Cpfcan_erel','Cpfcan_eta','Cpfcan_phi',
        #                       'Npfcan_erel','Npfcan_eta','Npfcan_phi',
        #                       'nCpfcand','nNpfcand',
        #                       'jet_eta','jet_phi'])
        
        
    #this function describes how the branches are converted
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        #the first part is standard, no changes needed
        from preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles, ZeroPadParticles
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
    
        x_glb  = ZeroPadParticles(filename,TupleMeanStd,
                                          self.branches[0],
                                          self.branchcutoffs[0],self.nsamples)

        x_db  = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                         self.branches[1],
                                         self.branchcutoffs[1],self.nsamples)
        
        x_pf  = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                         self.branches[2],
                                         self.branchcutoffs[2],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                         self.branches[3],
                                         self.branchcutoffs[3],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                        self.branches[4],
                                        self.branchcutoffs[4],self.nsamples)
        
        
        # now, some jets are removed to avoid pt and eta biases
        
        Tuple = self.readTreeFromRootToTuple(filename)
        if self.remove:
            # jets are removed until the shapes in eta and pt are the same as
            # the truth class 'fj_isNonBB'
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
	print(alltruth)
        
        # remove the entries to get same jet shapes
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_glb=x_glb[notremoves > 0]
            x_db=x_db[notremoves > 0]
            x_pf=x_pf[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            #x_global=x_global[notremoves > 0]
            #x_chmap=x_chmap[notremoves > 0]        
        
        #newnsamp=x_global.shape[0]
        newnsamp=x_glb.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        # fill everything
        self.w=[weights]
        self.x=[x_db,x_pf,x_cpf,x_sv]
        self.z=[x_glb]
        self.y=[alltruth]
	print("self:")
	print(self.y)

        
class TrainData_deepDoubleB_pf(TrainData_deepDoubleB):
    
    def __init__(self):
        '''
        This is an example data format description for FatJet studies
        '''
        TrainData_deepDoubleB.__init__(self)
        
        #example of how to register global branches
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
                          'nsv'
                      ])
        
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
        
        #example of pf candidate branches
        self.addBranches(['pfcand_ptrel',
                          'pfcand_erel',
                          'pfcand_phirel',
                          'pfcand_etarel',
                          'pfcand_deltaR',
                          'pfcand_puppiw',
                          'pfcand_drminsv',
                          'pfcand_drsubjet1',
                          'pfcand_drsubjet2',
                          'pfcand_hcalFrac'
                         ],
                         100) 

        #branches that are used directly in the following function 'readFromRootFile'
        #this is a technical trick to speed up the conversion
        #self.registerBranches(['Cpfcan_erel','Cpfcan_eta','Cpfcan_phi',
        #                       'Npfcan_erel','Npfcan_eta','Npfcan_phi',
        #                       'nCpfcand','nNpfcand',
        #                       'jet_eta','jet_phi'])
        
        
    #this function describes how the branches are converted
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        #the first part is standard, no changes needed
        from preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles, ZeroPadParticles
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
    
        x_glb  = ZeroPadParticles(filename,TupleMeanStd,
                                          self.branches[0],
                                          self.branchcutoffs[0],self.nsamples)

        x_db  = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                         self.branches[1],
                                         self.branchcutoffs[1],self.nsamples)
        
        x_pf  = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                         self.branches[2],
                                         self.branchcutoffs[2],self.nsamples)
        
        # now, some jets are removed to avoid pt and eta biases
        
        Tuple = self.readTreeFromRootToTuple(filename)
        if self.remove:
            # jets are removed until the shapes in eta and pt are the same as
            # the truth class 'fj_isNonBB'
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
	print(alltruth)
        
        # remove the entries to get same jet shapes
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_glb=x_glb[notremoves > 0]
            x_db=x_db[notremoves > 0]
            x_pf=x_pf[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            #x_global=x_global[notremoves > 0]
            #x_chmap=x_chmap[notremoves > 0]        
        
        #newnsamp=x_global.shape[0]
        newnsamp=x_glb.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        # fill everything
        self.w=[weights]
        self.x=[x_db,x_pf]
        self.z=[x_glb]
        self.y=[alltruth]
	print("self:")
	print(self.y)
        

    
        
class TrainData_deepDoubleB_simple10(TrainData_deepDoubleB):
    
    def __init__(self):
        '''
        This is an example data format description for FatJet studies
        '''
        TrainData_deepDoubleB.__init__(self)

        self.remove = True
        self.weight = False
        #example of how to register global branches
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
                          'nsv'
                      ])
        
        self.addBranches(['fj_jetNTracks',
                          'fj_nSV',
                          'fj_tau_flightDistance2dSig_0',
                          'fj_tau_flightDistance2dSig_1',
                          'fj_tau_vertexEnergyRatio_0',
                          'fj_tau_vertexMass_0',
                          'fj_trackSipdSig_0',
                          'fj_trackSipdSig_1',
                          'fj_trackSip2dSigAboveBottom_0',
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
        from preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles, ZeroPadParticles
        import numpy
        import ROOT
        
        fileTimeOut(filename,120) #give eos 2 minutes to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples=tree.GetEntries()
    
        x_glb  = ZeroPadParticles(filename,TupleMeanStd,
                                          self.branches[0],
                                          self.branchcutoffs[0],self.nsamples)

        x_db  = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                         self.branches[1],
                                         self.branchcutoffs[1],self.nsamples)
        
        # now, some jets are removed to avoid pt and eta biases
        
        Tuple = self.readTreeFromRootToTuple(filename)
        if self.remove:
            # jets are removed until the shapes in eta and pt are the same as
            # the truth class 'fj_isNonBB'
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
	print(alltruth)
        
        # remove the entries to get same jet shapes
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_glb=x_glb[notremoves > 0]
            x_db=x_db[notremoves > 0]
            alltruth=alltruth[notremoves > 0]  
        
        newnsamp=x_glb.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        # fill everything
        self.w=[weights]
        self.x=[x_db]
        self.z=[x_glb]
        self.y=[alltruth]
	print("self:")
	print(self.y)
        
