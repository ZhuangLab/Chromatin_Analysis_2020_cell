#Bogdan Bintu
#Copyright Presidents and Fellows of Harvard College, 2017.

#external packages
import numpy as np
import re
import glob,os

##Reader classes and functions
# Classes that handles reading STORM movie files.
# Currently this is limited to the dax format.
# It will be extended to tiff files in future releases.

#
# The superclass containing those functions that 
# are common to reading a (STORM/diffraction-limited) movie file.
# This was originally develloped by Hazen Babcok and extended by Bogdan Bintu

class Reader:
    
    # Close the file on cleanup.
    def __del__(self):
        if self.fileptr:
            self.fileptr.close()

    def __enter__(self):
        return self

    def __exit__(self, etype, value, traceback):
        if self.fileptr:
            self.fileptr.close()

    # Average multiple frames in a movie.
    def averageFrames(self, start = False, end = False, verbose = False):
        if (not start):
            start = 0
        if (not end):
            end = self.number_frames 

        length = end - start
        average = np.zeros((self.image_width, self.image_height), np.float)
        for i in range(length):
            if verbose and ((i%10)==0):
                print " processing frame:", i, " of", self.number_frames
            average += self.loadAFrame(i + start)
            
        average = average/float(length)
        return average

    # returns the film name
    def filmFilename(self):
        return self.filename

    # returns the film size
    def filmSize(self):
        return [self.image_width, self.image_height, self.number_frames]

    # returns the picture x,y location, if available
    def filmLocation(self):
        if hasattr(self, "stage_x"):
            return [self.stage_x, self.stage_y]
        else:
            return [0.0, 0.0]

    # returns the film focus lock target
    def lockTarget(self):
        if hasattr(self, "lock_target"):
            return self.lock_target
        else:
            return 0.0

    # returns the scale used to display the film when
    # the picture was taken.
    def filmScale(self):
        if hasattr(self, "scalemin") and hasattr(self, "scalemax"):
            return [self.scalemin, self.scalemax]
        else:
            return [100, 2000]


#
# Dax reader class. This is the Zhuang lab custom format.
#
class DaxReader(Reader):
    # dax specific initialization
    def __init__(self, filename, verbose = 0):
        # save the filenames
        self.filename = filename
        dirname = os.path.dirname(filename)
        if (len(dirname) > 0):
            dirname = dirname + "/"
        self.inf_filename = dirname + os.path.splitext(os.path.basename(filename))[0] + ".inf"

        # defaults
        self.image_height = None
        self.image_width = None

        # extract the movie information from the associated inf file
        size_re = re.compile(r'frame dimensions = ([\d]+) x ([\d]+)')
        length_re = re.compile(r'number of frames = ([\d]+)')
        endian_re = re.compile(r' (big|little) endian')
        stagex_re = re.compile(r'Stage X = ([\d\.\-]+)')
        stagey_re = re.compile(r'Stage Y = ([\d\.\-]+)')
        lock_target_re = re.compile(r'Lock Target = ([\d\.\-]+)')
        scalemax_re = re.compile(r'scalemax = ([\d\.\-]+)')
        scalemin_re = re.compile(r'scalemin = ([\d\.\-]+)')

        inf_file = open(self.inf_filename, "r")
        while 1:
            line = inf_file.readline()
            if not line: break
            m = size_re.match(line)
            if m:
                self.image_height = int(m.group(1))
                self.image_width = int(m.group(2))
            m = length_re.match(line)
            if m:
                self.number_frames = int(m.group(1))
            m = endian_re.search(line)
            if m:
                if m.group(1) == "big":
                    self.bigendian = 1
                else:
                    self.bigendian = 0
            m = stagex_re.match(line)
            if m:
                self.stage_x = float(m.group(1))
            m = stagey_re.match(line)
            if m:
                self.stage_y = float(m.group(1))
            m = lock_target_re.match(line)
            if m:
                self.lock_target = float(m.group(1))
            m = scalemax_re.match(line)
            if m:
                self.scalemax = int(m.group(1))
            m = scalemin_re.match(line)
            if m:
                self.scalemin = int(m.group(1))

        inf_file.close()

        # set defaults, probably correct, but warn the user 
        # that they couldn't be determined from the inf file.
        if not self.image_height:
            print "Could not determine image size, assuming 256x256."
            self.image_height = 256
            self.image_width = 256

        # open the dax file
        if os.path.exists(filename):
            self.fileptr = open(filename, "rb")
        else:
            self.fileptr = 0
            if verbose:
                print "dax data not found", filename
                
    # Create and return a memory map the dax file
    def loadMap(self):
        if os.path.exists(self.filename):
            if self.bigendian:
                self.image_map = np.memmap(self.filename, dtype='>u2', mode='r', shape=(self.number_frames,self.image_width, self.image_height))
            else:
                self.image_map = np.memmap(self.filename, dtype='uint16', mode='r', shape=(self.number_frames,self.image_width, self.image_height))
        return self.image_map
        
    # load a frame & return it as a np array
    def loadAFrame(self, frame_number):
        if self.fileptr:
            assert frame_number >= 0, "frame_number must be greater than or equal to 0"
            assert frame_number < self.number_frames, "frame number must be less than " + str(self.number_frames)
            self.fileptr.seek(frame_number * self.image_height * self.image_width * 2)
            image_data = np.fromfile(self.fileptr, dtype='uint16', count = self.image_height * self.image_width)
            image_data = np.transpose(np.reshape(image_data, [self.image_width, self.image_height]))
            if self.bigendian:
                image_data.byteswap(True)
            return image_data
    # load full movie and retun it as a np array        
    def loadAll(self):
        image_data = np.fromfile(self.fileptr, dtype='uint16', count = -1)
        image_data = np.swapaxes(np.reshape(image_data, [self.number_frames,self.image_width, self.image_height]),1,2)
        if self.bigendian:
            image_data.byteswap(True)
        return image_data
import re
import numpy as np
import os
def slice_file(fl,sx,sy,sz,minx=0,maxx=np.inf,miny=0,maxy=np.inf,minz=0,maxz=np.inf,stridex=1,stridey=1,stridez=1):
    """
    Given a file <fl> with the binary output of some np.uint16 data 
    (i.e. saved via: data.tofile("temp.bin") where data is np.array of size sx,sy,sz)
    This returns a sliced array: data[minx:maxx,miny:maxy,minz:maxz]
    """
    sx,sy,sz = int(sx),int(sy),int(sz)
    if maxx>sx: maxx=sx
    if maxy>sy: maxy=sy
    if maxz>sz: maxz=sz
    if minx<0: minx=0
    if miny<0: miny=0
    if minz<0: minz=0
    minx,maxx,miny,maxy,minz,maxz = int(minx),int(maxx),int(miny),int(maxy),int(minz),int(maxz)
    dx = maxx-minx
    dy = maxy-miny
    dz = maxz-minz
    
    if dx<=0: dx = 0
    if dy<=0: dy = 0
    if dz<=0: dz = 0
    data = np.zeros([dx,dy,dz],dtype=np.uint16)
    if np.prod(data.shape)==0:
        return data
    #print minx,maxx,miny,maxy,minz,maxz,sx,sy,sz
    f = open(fl, "rb")
    start = sy*sz*minx+miny*sz+minz
    f.seek(start*2)

    if dx<=0 or dy<=0 or dz<=0:
        #return np.array([],dtype=np.uint16)
        pass
    dims = [int(np.ceil(float(dx)/stridex)),int(np.ceil(float(dy)/stridey)),int(np.ceil(float(dz)/stridez))]
    data = np.zeros(dims,dtype=np.uint16)
    f = open(fl, "rb")
    start = sy*sz*minx+miny*sz+minz
    f.seek(start*2)


    chunk = np.fromfile(f, dtype=np.uint16,count=dz)
    data[0,0]=chunk[::stridez]
    county = 0
    for i in range(dy-1):
        if (i+1)%stridey==0:
            county+=1
            f.seek((sz-dz)*2,os.SEEK_CUR)
            chunk = np.fromfile(f, dtype=np.uint16,count=dz)
            data[0,county]=chunk[::stridez]
        else:
            f.seek(sz*2,os.SEEK_CUR)
    countx=0
    for i in range(dx-1):
        if (i+1)%stridex==0:
            countx+=1
            start = (sy-dy)*sz+sz-dz
            f.seek(start*2,os.SEEK_CUR)
            chunk = np.fromfile(f, dtype=np.uint16,count=dz)
            data[countx,0]=chunk[::stridez]
            
            county = 0
            for j in range(dy-1):
                if (j+1)%stridey==0:
                    county+=1
                    f.seek((sz-dz)*2,os.SEEK_CUR)
                    chunk = np.fromfile(f, dtype=np.uint16,count=dz)
                    data[countx,county]=chunk[::stridez]
                else:
                    f.seek(sz*2,os.SEEK_CUR)
        else:
            f.seek(sy*sz*2,os.SEEK_CUR)
        
    f.close()
    return data
class dax_im():
    def __init__(self,dax_fl,num_col=None,start_cutoff=12,end_cutoff=10,bead_col=None,color=0):
        #internalize
        self.color = color
        self.start_cutoff,self.end_cutoff = start_cutoff,end_cutoff
        self.dax_fl = dax_fl
        self.hybe = os.path.basename(os.path.dirname(self.dax_fl))
        self.num_col = num_col
        self.bead_col = bead_col
        if self.num_col is None:
            self.num_col = self.hybe.count(',')+2
        self.read_info_file()
        self.shape = (self.number_frames,self.image_height,self.image_width)
        
        start = self.start(self.color)
        end = self.number_frames-self.end_cutoff
        self.shape = (int((end-start)/self.num_col)+1,self.image_height,self.image_width)
    def read_info_file(self):
        self.inf_filename = self.dax_fl.replace('.dax','.inf')
        
        inf_file = open(self.inf_filename, "r")
        
        size_re = re.compile(r'frame dimensions = ([\d]+) x ([\d]+)')
        length_re = re.compile(r'number of frames = ([\d]+)')
        endian_re = re.compile(r' (big|little) endian')
        stagex_re = re.compile(r'Stage X = ([\d\.\-]+)')
        stagey_re = re.compile(r'Stage Y = ([\d\.\-]+)')
        lock_target_re = re.compile(r'Lock Target = ([\d\.\-]+)')
        scalemax_re = re.compile(r'scalemax = ([\d\.\-]+)')
        scalemin_re = re.compile(r'scalemin = ([\d\.\-]+)')
        
        while 1:
            line = inf_file.readline()
            if not line: break
            m = size_re.match(line)
            if m:
                self.image_height = int(m.group(1))
                self.image_width = int(m.group(2))
            m = length_re.match(line)
            if m:
                self.number_frames = int(m.group(1))
            m = endian_re.search(line)
            if m:
                if m.group(1) == "big":
                    self.bigendian = 1
                else:
                    self.bigendian = 0
            m = stagex_re.match(line)
            if m:
                self.stage_x = float(m.group(1))
            m = stagey_re.match(line)
            if m:
                self.stage_y = float(m.group(1))
            m = lock_target_re.match(line)
            if m:
                self.lock_target = float(m.group(1))
            m = scalemax_re.match(line)
            if m:
                self.scalemax = int(m.group(1))
            m = scalemin_re.match(line)
            if m:
                self.scalemin = int(m.group(1))
                    # extract the movie information from the associated inf file
        inf_file.close()

    def get_slice(self,
               minx=0,maxx=np.inf,miny=0,maxy=np.inf,minz=0,maxz=np.inf,
               stridex=1,stridey=1,stridez=1):
        fl = self.dax_fl
        sx,sy,sz = self.number_frames,self.image_height,self.image_width
        return slice_file(fl,sx,sy,sz,
                   minx=minx,maxx=maxx,miny=miny,maxy=maxy,minz=minz,maxz=maxz,
                   stridex=stridex,stridey=stridey,stridez=stridez)
    def start(self,ind_col):
        """Given the color index <ind_col> this returns the first frame z-step frame 
        given information on the number of colors and padding"""
        num_col,start_cutoff,end_cutoff = self.num_col,self.start_cutoff,self.end_cutoff
        return int(np.ceil((start_cutoff-1)/float(num_col))*num_col+1+ind_col%num_col)
    def get_ims(self,minx=0,maxx=np.inf,miny=0,maxy=np.inf):
        """
        Given the optional x,y crop values, this returns num_col arrays 
        """
        num_col = self.num_col
        end = self.number_frames-self.end_cutoff
        im_block = self.get_slice(minx=0,maxx=end,miny=miny,maxy=maxy,minz=minx,maxz=maxx,
               stridex=1,stridey=1,stridez=1)
        im_block = np.swapaxes(im_block,1,2)
        
        im_blocks = [im_block[self.start(ind_col)::num_col] for ind_col in range(num_col)]
        len_ = np.min(map(len,im_blocks))
        im_blocks = np.array([im[:len_] for im in im_blocks],dtype=np.uint16)
        return im_blocks
    def get_mids(self,tag='mid',minx=0,maxx=np.inf,miny=0,maxy=np.inf):
        """
        Given the optional x,y crop values, this returns num_col arrays of 
        """
        num_col = self.num_col
        end = self.number_frames-self.end_cutoff
        ims = []
        for ind_col in range(num_col):
            frames = range(self.start(ind_col),end,num_col)
            if tag=='all':
                im_block = self.get_slice(minx=0,maxx=end,miny=miny,maxy=maxy,minz=minx,maxz=maxx,
                       stridex=1,stridey=1,stridez=1)
            elif tag=='mid':
                fr = frames[int(len(frames)/2)]
                im_block = self.get_slice(minx=fr,maxx=fr+1,miny=miny,maxy=maxy,minz=minx,maxz=maxx,
                       stridex=1,stridey=1,stridez=1)
            elif tag=='start':
                fr = frames[0]
                im_block = self.get_slice(minx=fr,maxx=fr+1,miny=miny,maxy=maxy,minz=minx,maxz=maxx,
                       stridex=1,stridey=1,stridez=1)
            im_block = np.mean(np.swapaxes(im_block,1,2),axis=0)
            ims.append(im_block)
        return np.array(ims)
    def get_im(self,ind_col=None,minx=0,maxx=np.inf,miny=0,maxy=np.inf):
        """
        Given the optional x,y crop values, this returns the <ind_col> indexed image
        """
        if ind_col is None:
            ind_col = self.color
        num_col = self.num_col
        start = self.start(ind_col)
        end = self.number_frames-self.end_cutoff
        im_block = self.get_slice(minx=start,maxx=end,miny=miny,maxy=maxy,minz=minx,maxz=maxx,
               stridex=num_col,stridey=1,stridez=1)
        im_block = np.swapaxes(im_block,1,2)
        return im_block
    def get_im_beads(self,bead_col=None,minx=0,maxx=np.inf,miny=0,maxy=np.inf):
        if bead_col is not None:
            self.bead_col = bead_col
        if self.bead_col is None:
            self.bead_col = self.num_col
        return self.get_im(self.bead_col,minx=minx,maxx=maxx,miny=miny,maxy=maxy)  
# function for quickly reading fasta files
def fastaread(fl,force_upper=False):
    """
    Given a .fasta file <fl> this returns names,sequences
    """
    fid = open(fl,'r')
    names = []
    seqs = []
    lines = []
    while True:
        line = fid.readline()
        if not line:
            seq = "".join(lines)
            if force_upper:
                seq=seq.upper()
            seqs.append(seq)
            break
        if line[0]=='>':
            name = line[1:-1]
            names.append(name)
            seq = "".join(lines)
            if force_upper:
                seq=seq.upper()
            seqs.append(seq)
            lines = []
        else:
            lines.append(line[:-1])
    fid.close()
    return [names,seqs[1:]]

def batch_command(str_runs,batch_size=8,max_time=np.inf,verbose=True):
    """str_runs is a list of commands you want to bach in the terminal
    batch_size is the number of commands you run at once
    max_time is the maximum execution time in seconds of each command
    """
    from timeit import default_timer as timer
    import subprocess
    str_inds=range(len(str_runs))
    ninds = len(str_inds)
    popens=[] # list of the running processes
    commands=[] # list of the running comands (strings)
    starts=[] # list of timers for the running processes
    #initial jobs
    for i in range(batch_size):
        if i<ninds:
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

#Stand alone functions
def getH(folder):
    nm = os.path.basename(folder)
    let_ = ''
    for let in nm[1:]:
        if not let.isdigit():
            break
        let_+=let
    return int(let_)
def flatten(l):
    return [item for sublist in l for item in sublist]
    
def readInfoFile(info_file):
    """Transform dax info file to dictionary format"""
    #ensure appropriate extension
    ext = info_file.split('.')[-1]
    if ext=='dax':
        info_file=info_file.replace('.dax','.inf')
    
    lines = [ln for ln in open(info_file,'r')]
    dic={}
    for ln in lines:
        splits = ln[:-1].split(' =')
        if len(splits)>1:
            try:
                dic[splits[0]]=float(splits[1])
            except:
                dic[splits[0]]=splits[1]
    
    return dic
def loadMap(filename):
    dic_info = readInfoFile(filename)
    bigendian = not ('little' in dic_info['data type'])
    dtype= '>u2' if bigendian else 'uint16'
    number_frames = int(dic_info['number of frames'])
    image_width,image_height = list(map(int,dic_info['frame dimensions'].split('x')))
    image_map = np.memmap(filename, dtype=dtype, mode='r', shape=(number_frames,image_width, image_height))
    return image_map
def mem_map_ims(folders,fov,starts = None):
    """Given a list of <folders>,a <fov>, and a list of start positions this returns the images """
    ims=[]
    ims_names=[]
    ims_beads = []
    #num_cols = np.max([os.path.basename(folder).count(',')for folder in folders])+1
    #num_cols = len(starts)
    for folder in folders:
        if starts is not None:
            starts_ = starts
            num_cols = len(starts_)
        else:
            num_cols = os.path.basename(folder).count(',')+1
            starts_ = [0]*num_cols
        im = np.swapaxes(loadMap(folder+os.sep+fov),1,2)
        #names = nm_to_nms(folder)
        names = [os.path.basename(folder)+'_col'+str(i) for i in range(num_cols)]
        rmax=np.max(starts_)
        #st = int(np.ceil(10./(num_cols+1))*(num_cols+1))
        st=0
        for i in range(num_cols):
            start = starts_[i]
            end = -(rmax-start) if (rmax-start)!=0 else None
            ims += [im[st+i+1::(num_cols+1)][start:end]]#
            ims_names += [names[i%len(names)]]
        ims_beads += [im[st+num_cols+1::(num_cols+1)]]#
    beads_names = map(os.path.basename,folders)
    return ims,ims_names,ims_beads,beads_names
def get_ims_fov_daxmap(folders,fov,col_tags = None,pad = 10):
    """Given folders and a fov, this returns ims_signal,ims_beads,names"""
    daxs_signal,names_signal,daxs_beads,names_beads=[],[],[],[]
    col_tags_ = col_tags
    for folder in folders:
        dax_fl= folder+os.sep+fov
        tag=os.path.basename(folder)
        if os.path.exists(dax_fl):
            if col_tags is None:
                num_dyes = tag.count(',')+1
                if num_dyes==1: num_dyes=2
                col_tags_= ['750','647','561','488'][:num_dyes]+['beads']
            num_cols = len(col_tags_)
            for icol,col_tag in enumerate(col_tags_):
                if col_tag=='beads':
                    names_beads+= [tag+'_'+col_tag]
                    dax_obj =dax_im(dax_fl, num_col=num_cols, start_cutoff=pad, end_cutoff=pad, bead_col=num_cols-1,color=icol)
                    daxs_beads.append(dax_obj)
                else:
                    dax_obj =dax_im(dax_fl, num_col=num_cols, start_cutoff=pad, end_cutoff=pad, bead_col=num_cols-1,color=icol)
                    dax_obj.index_beads = len(daxs_beads)
                    daxs_signal.append(dax_obj)
                    names_signal+= [tag+'_'+col_tag]
    return  daxs_signal,names_signal,daxs_beads,names_beads
def hybe_number(hybe_folder):
    """Give a folder of the type path\H3R9, this returns the hybe number 3"""
    hybe_tag = os.path.basename(hybe_folder)
    is_letter = [char.isalpha() for char in hybe_tag]
    pos = np.where(is_letter)[0]
    if len(pos)==1:
        pos=list(pos)+[len(hybe_tag)]
    return int(hybe_tag[pos[0]+1:pos[1]])
def get_valid_dax(spots_folder,ifov=0):
    files_folders = glob.glob(spots_folder+os.sep+'*')
    folders = [fl for fl in files_folders if os.path.isdir(fl)]
    valid_folders = [folder for folder in folders if os.path.basename(folder)[0]=='H']
    hybe_tags = [os.path.basename(folder) for folder in valid_folders]
    #order hybe tags
    
    hybe_tags = np.array(hybe_tags)[np.argsort(map(hybe_number,hybe_tags))]
    fov_tags=[]
    for hybe_tag in hybe_tags:
        fov_tags.extend(map(os.path.basename,glob.glob(spots_folder+os.sep+hybe_tag+os.sep+'*.dax')))
    fov_tags = np.unique(fov_tags)
    fov_tag = fov_tags[ifov]
    daxs = [spots_folder+os.sep+tag+os.sep+fov_tag for tag in hybe_tags]
    daxs = [dax for dax in daxs if os.path.exists(dax)]
    return daxs
### Writing tools
class DaxWriter():
    def __init__(self, name):
        """
        # Writes dax files. Designed for use by analysis & simulation
        # not so much for use by data taking code.
        #
        # Hazen 4/09, modified by Bogdan Bintu
        #
        """
        self.name = name
        if len(os.path.dirname(name)) > 0:
            self.root_name = os.path.dirname(name) + "/" + os.path.splitext(os.path.basename(name))[0]
        else:
            self.root_name = os.path.splitext(os.path.basename(name))[0]
        self.w,self.h,self.l = 0,0,0
        self.fp = open(self.name, "wb")

    def addFrame(self, frame):
        frame = frame.copy()
        frame = np.clip(frame,0,2**16-1)
        image16 = frame.astype(np.uint16)
        image16 = np.transpose(image16)
        image16.byteswap(False)
        image16.tofile(self.fp)
        self.l += 1
        [self.w, self.h] = frame.shape
    def close(self):
        self.fp.close()
        inf_fp = open(self.root_name + ".inf", "w")
        inf_fp.write("binning = 1 x 1\n")
        inf_fp.write("data type = 16 bit integers (binary, little endian)\n")
        inf_fp.write("frame dimensions = " + str(self.w) + " x " + str(self.h) + "\n")
        inf_fp.write("number of frames = " + str(self.l) + "\n")
        inf_fp.write("Lock Target = 0.0\n")
        if 1:
            inf_fp.write("x_start = 1\n")
            inf_fp.write("x_end = " + str(self.w) + "\n")
            inf_fp.write("y_start = 1\n")
            inf_fp.write("y_end = " + str(self.h) + "\n")
        inf_fp.close()
def writeDax(im3d,dax_file,rescale=True,min_=None,max_=None):
    im3d_ = im3d
    if rescale:
        im3d_ = cast_uint16(im3d,min_=min_,max_=max_)
    DaxWriter_ = DaxWriter(dax_file)
    for f in im3d_:
        DaxWriter_.addFrame(f)
    DaxWriter_.close()
def cast_uint16(im,min_=None,max_=None):
    im_ = np.array(im,dtype=np.float32)
    if min_ is None: min_ = np.min(im_)
    if max_ is None: max_ = np.max(im_)
    delta = max_-min_
    if delta==0: delta =1
    im_ = (im_-min_)/delta
    im_ = (np.clip(im_,0,1)*(2**16-1)).astype(np.uint16)
    return im_
def cast_uint8(im,min_=None,max_=None):
    im_ = np.array(im,dtype=np.float32)
    if min_ is None: min_ = np.min(im_)
    if max_ is None: max_ = np.max(im_)
    delta = max_-min_
    if delta==0: delta =1
    im_ = (im_-min_)/delta
    im_ = (np.clip(im_,0,1)*255).astype(np.uint8)
    return im_