#Bogdan Bintu
#Copyright Presidents and Fellows of Harvard College, 2017.

# This is intended as a suppliment (mostly wrappers) to the storm_analysis repository Hazen Babcock wrote
# Add windows_dll to enviromental variables on windows in order for the STORM fitting to work

import numpy as np
import sys,os,glob


#Add paths
storm_analysis_path = os.path.dirname(os.path.abspath(__file__))+os.sep+'storm-analysis'
daostorm_3d_path = storm_analysis_path+os.sep+'3d_daostorm'
windows_dll_path = storm_analysis_path+os.sep+'windows_dll'
paths = [storm_analysis_path,daostorm_3d_path,windows_dll_path]
#Remember to add windows_dll to enviromental variables.

for path_ in paths:
    sys.path.append(path_)

import sa_library.datareader as datareader
import sa_library.parameters as params


class EmptyParamaters:
    def __init__(self):
        None
def loadParms(xml_file):
    parms=params.Parameters(xml_file)
    return parms

def writeDicParms(dic,xmlFile):
    import collections, dicttoxml, inspect
    from xml.dom.minidom import parseString
    dic_ = collections.OrderedDict(dic)
    xml_pars = dicttoxml.dicttoxml(dic_,custom_root='settings') 
    dom = parseString(xml_pars)
    save_pars = dom.toprettyxml()
    f=open(xmlFile,'w')
    f.write(save_pars)
    f.close()
def writeParms(parms,xmlFile):
    """Save paramaters"""
    print "If necessary, run in terminal: python -m pip install dicttoxml" 
    import collections, dicttoxml, inspect
    from xml.dom.minidom import parseString
    dic = collections.OrderedDict(inspect.getmembers(parms)[3:])
    xml_pars = dicttoxml.dicttoxml(dic,custom_root='settings') 
    dom = parseString(xml_pars)
    save_pars = dom.toprettyxml()
    f=open(xmlFile,'w')
    f.write(save_pars)
    f.close()

def getParms(threshold_=280.0,baseline_=120.0,radius_=1.0,cutoff_=1.0,sigma_=1.0,pixel_size_=158.0,descriptor_=u'1',iterations_=20,model_=u'Z',orientation_=u'normal',start_frame_=0,max_frame_=-1,drift_correction_=0,d_scale_=2,frame_step_=8000,do_zfit_=0,max_z_=0.5,min_z_=-0.5,wxA_=-7.1131,wxB_=19.9998,wxC_=0.0,wxD_=0.0,wx_c_=415.5645,wx_d_=958.792,wx_wo_=238.3076,wyA_=0.53549,wyB_=-0.099514,wyC_=0.0,wyD_=0.0,wy_c_=-310.7737,wy_d_=268.0425,wy_wo_=218.9904,x_start_=0,x_stop_=512,y_start_=0,y_stop_=512):
    """Returns a mock set of paramaters"""
    parms = EmptyParamaters()
    #standard paramaters
    parms.threshold=threshold_
    parms.baseline=baseline_ #ccd background
    parms.radius=radius_
    parms.cutoff=cutoff_
    parms.sigma=sigma_
    parms.pixel_size=pixel_size_
    parms.descriptor=descriptor_
    parms.iterations=iterations_
    parms.model=model_
    parms.orientation=orientation_
    #parms.parameters_file=xml_file
    # time frames
    parms.start_frame=start_frame_
    parms.max_frame=max_frame_
    # drift correction
    parms.drift_correction=drift_correction_

    parms.d_scale=d_scale_
    parms.frame_step=frame_step_
    #z-fit
    parms.do_zfit=do_zfit_

    parms.max_z=max_z_
    parms.min_z=min_z_

    parms.wxA=wxA_
    parms.wxB=wxB_
    parms.wxC=wxC_
    parms.wxD=wxD_
    parms.wx_c=wx_c_
    parms.wx_d=wx_d_
    parms.wx_wo=wx_wo_
    parms.wyA=wyA_
    parms.wyB=wyB_
    parms.wyC=wyC_
    parms.wyD=wyD_
    parms.wy_c=wy_c_
    parms.wy_d=wy_d_
    parms.wy_wo=wy_wo_

    #ROI
    parms.x_start=x_start_
    parms.x_stop=x_stop_
    parms.y_start=y_start_
    parms.y_stop=y_stop_
    return parms
def fitFrame(dax_file,bin_file,parms,over_write=True,write_parms=True):
    """
    Constructs a bin_file
    Use getParms function to generate parms for this function
    getParms(xml_file,max_frame_=<frame>+1,start_frame_=<frame>) 
    """
    import find_peaks
    import sa_utilities.std_analysis as std_analysis
    mlist_file = bin_file
    if not over_write:
        if os.path.exists(bin_file):
            return parms
    if os.path.exists(bin_file):
        os.remove(bin_file)
    alist_file=bin_file.replace('_mlist','_alist')
    if os.path.exists(alist_file):
        os.remove(alist_file)
    if write_parms:
        xmlFile = bin_file.replace('_mlist.bin','_parms.xml')
        writeParms(parms,xmlFile)
    finder = find_peaks.initFindAndFit(parms)
    std_analysis.standardAnalysis(finder,dax_file,mlist_file,parms)
    return parms
def fitImage(im,parms,rescale=False,plt_val=False):
    """
    fits an image using writeDax to tempfile and fits using fitFrame with parms
    Returns mlist
    """
    import tempfile
    (fd_dax,dax_file) = tempfile.mkstemp(suffix='.dax')
   
    im_=np.array(im)
    if len(im_.shape)==2: im_=[im_]
    writeDax(dax_file,im_,rescale=rescale)
    inf_file=dax_file.replace('.dax','.inf')
    bin_file=dax_file.replace('.dax','_mlist.bin')
    alist_file=dax_file.replace('.dax','_alist.bin')
    fitFrame(dax_file,bin_file,parms,over_write=True,write_parms=False)
    mlist = readMasterMoleculeList(bin_file)
    
    tfile = os.fdopen(fd_dax, "w")
    tfile.close()
    for fl in [dax_file,bin_file,alist_file,inf_file]:
        os.remove(fl)
    if plt_val:
        import matplotlib.pylab as plt
        plt.figure()
        plt.plot(mlist['y']-1,mlist['x']-1,'o')
        plt.imshow(im,interpolation='nearest')
        plt.show()
    return mlist

def batch_command(str_runs,batch_size=8,max_time=np.inf,verbose=True):
    """str_runs is a list of commands you want to bach in the terminal
    batch_size is the number of commands you run at once
    max_time is the maximum execution time in seconds of each command
    """
    from timeit import default_timer as timer
    import subprocess
    str_inds=range(len(str_runs))
    popens=[] # list of the running processes
    commands=[] # list of the running comands (strings)
    starts=[] # list of timers for the running processes
    #initial jobs
    for i in range(batch_size):
        if i<len(str_inds):
            popens.append(subprocess.Popen(str_runs[str_inds[0]], shell=True))
            commands.append(str_runs[str_inds[0]])
            if verbose:
                print "initial_job: "+str_runs[str_inds[0]]
            str_inds=np.setdiff1d(str_inds,str_inds[0])
            starts.append(timer())
    starts=np.array(starts)
    #checks status
    while len(str_inds):
        for i in range(batch_size):
            if i<len(str_inds):
                #check if process finished, if so, open a new one
                if popens[i].poll()==0:
                    if verbose:
                        print "finished job: "+commands[i]
                    popens[i]=subprocess.Popen(str_runs[str_inds[0]], shell=True)
                    commands[i]=str_runs[str_inds[0]]
                    if verbose:
                        print "started_new_job: "+commands[i]
                    str_inds=np.setdiff1d(str_inds,str_inds[0])
                    starts[i]=timer()
                #check if process maxed out on time, if so, kill it and open a new one
                end_timer = timer()
                if end_timer-starts[i]>max_time:
                    popens[i].kill()
                    if verbose:
                        print "killed job - timed out: "+commands[i]
                    popens[i]=subprocess.Popen(str_runs[str_inds[0]], shell=True)
                    commands[i]=str_runs[str_inds[0]]
                    if verbose:
                        print "started_new_job: "+commands[i]
                    str_inds=np.setdiff1d(str_inds,str_inds[0])
                    starts[i]=timer()
    while(len(popens)):
        for i in range(len(popens)):
            end_timer = timer()
            if end_timer-starts[i]>max_time:
                popens[i].kill()
                if verbose:
                    print "killed job - timed out: "+commands[i]
                popens.pop(i)
            if i<len(popens):
                if popens[i].poll()==0:
                    if verbose:
                        print "finished job: "+commands[i]
                    popens.pop(i)    
def fitFrameBatch(dax_files,bin_files,parms_files,over_write=1,mock=False,batch_size_=20,verbose=False,python_path=None):
    """This batches the fits."""
    import GeneralTools as gt
    folder_functions = os.path.dirname(os.path.abspath(__file__))
    script = folder_functions+os.sep+r'BatchFit.py'
    if not os.path.exists(script):
        print "Script file not found in: "+folder_functions
        return None
    if python_path is None:
        python_path=r""
    else:
        python_path = "path="+python_path+";&&"
    str_run_base = python_path+r'python '+script+' '
    str_runs = [str_run_base+'"'+dax_file+'" "'+bin_file+'" "'+parms_file+'" '+str(int(over_write)) 
                for dax_file,bin_file,parms_file in zip(dax_files,bin_files,parms_files)]
    if mock:
        return str_runs
    batch_command(str_runs,batch_size=batch_size_,verbose=verbose)
    
def readMasterMoleculeList(bin_file,verbose=True):
    """
    This reads the molecule list 3dDaoSTORM computes and returns it as a multifield numpy.array
    """
    import sa_library.readinsight3 as readinsight3
    i3_block = readinsight3.loadI3FileNumpy(bin_file, verbose = verbose)
    return i3_block
    
    
def readDax(dax_file,frames=None):
    """
    This imports dax files as numpy arrays. By default if frames==None it imports all frames.
    You can also specify a list of frames to import.
    """
    import sa_library.datareader as datareader
    DaxReader_=datareader.DaxReader(dax_file)
    sz_x,sz_y,sz_t=DaxReader_.filmSize()
    if frames==None:
        frames=range(sz_t)
    im_dax = np.array([DaxReader_.loadAFrame(frame) for frame in frames])
    return im_dax

def writeDax(dax_file,dax_data,rescale=False):
    """This saves dax files given numpy arrays. It automatically converts to uint16"""
    import sa_library.daxwriter as daxwriter
    dax_data_ = np.array(dax_data)
    if rescale:
        dax_data_=np.array(dax_data,dtype=float)
        min_=np.min(dax_data_)
        max_=np.max(dax_data_)
        if max_-min_!=0:
            dax_data_ = (dax_data_-min_)/(max_-min_)
        else:
            dax_data_ = dax_data_-min_
        dax_data_=dax_data_*(2**16-1)
    dax_data_ = np.array(dax_data_,dtype=np.uint16)
    w,h = dax_data_[0].shape
    DaxWriter_ = daxwriter.DaxWriter(dax_file,w,h)
    for f in dax_data_:
        DaxWriter_.addFrame(f)
    DaxWriter_.close()
    
    
def readInfoFile(info_file):
    """Transform dax info file to dictionary format"""
    lines = [ln for ln in open(info_file,'r')]
    parse_lines = [ln for ln in lines if ln.count('=')]
    parse_lines = [[ln.split(' = ')[0],ln.split(' = ')[1][:-1]] for ln in parse_lines]
    import re
    dic={}
    for ln in parse_lines:
        list_=re.findall(r"[-+]?\d*\.\d+|\d+", ln[-1])
        if len(list_)!=1:
            dic[ln[0]]=ln[1]
        else:
            dic[ln[0]]=float(list_[0])
    return dic
    
def readOffsetFile(offset_file):
    """Transform offset file to dictionary format"""
    import csv
    with open(offset_file,'r') as f:
        reader=csv.reader(f,delimiter=' ')
        data = list(reader)
    fields = data[0] # get fields from header
    datavalues = np.array(data[1:],dtype=float).T
    dic={field:data for field,data in zip(fields,datavalues)}
    return dic
