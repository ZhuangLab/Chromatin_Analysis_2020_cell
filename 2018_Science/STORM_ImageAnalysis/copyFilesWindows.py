#Author: Bogdan Bintu
#bbintu@fas.harvard.edu
#Copyright Presidents and Fellows of Harvard College, 2018.

#This has been addapted from: https://github.com/frmdstryr/pywinutils/blob/master/winutils.py
#This is intended to handle copying on windows.

import pythoncom
from win32com.shell import shell,shellcon
import os

def _file_operation(src,dst=None,operation='copy',flags=shellcon.FOF_NOCONFIRMATION):
    # @see IFileOperation
    pfo = pythoncom.CoCreateInstance(shell.CLSID_FileOperation,None,pythoncom.CLSCTX_ALL,shell.IID_IFileOperation)

    # Respond with Yes to All for any dialog
    # @see http://msdn.microsoft.com/en-us/library/bb775799(v=vs.85).aspx
    pfo.SetOperationFlags(flags)

    if not isinstance(src,(tuple,list)):
        src = (src,)

    for f in src:
        item = shell.SHCreateItemFromParsingName(f,None,shell.IID_IShellItem)
        op = operation.strip().lower()
        if op=='copy':
            # Set the destionation folder
            if type(dst) is str:
                dst_ = shell.SHCreateItemFromParsingName(dst,None,shell.IID_IShellItem)
            else:
                dst_str = os.path.dirname(dst[src.index(f)])
                dst_ = shell.SHCreateItemFromParsingName(dst_str,None,shell.IID_IShellItem)
            pfo.CopyItem(item,dst_) # Schedule an operation to be performed
        elif op=='move':
            # Set the destionation folder
            if type(dst) is str:
                dst_ = shell.SHCreateItemFromParsingName(dst,None,shell.IID_IShellItem)
            else:
                dst_str = dst[src.index(f)]
                dst_ = shell.SHCreateItemFromParsingName(dst_str,None,shell.IID_IShellItem)
            pfo.MoveItem(item,dst_)
        elif op=='delete':
            pfo.DeleteItem(item)
        else:
            raise ValueError("Invalid operation {}".format(operation))

    # @see http://msdn.microsoft.com/en-us/library/bb775780(v=vs.85).aspx
    success = pfo.PerformOperations()

    # @see sdn.microsoft.com/en-us/library/bb775769(v=vs.85).aspx
    aborted = pfo.GetAnyOperationsAborted()
    return success is None and not aborted
    

def copy(src,dst,flags=shellcon.FOF_NOCONFIRMATION):
    """ Copy files using the built in Windows File operations dialog
    
    Requires absolute paths. Does NOT create root destination folder if it doesn't exist.
    
    Overwrites and is recursive by default 
    
    @see http://msdn.microsoft.com/en-us/library/bb775799(v=vs.85).aspx for flags available
    
    """
    return _file_operation(src,dst,'copy',flags)

def move(src,dst,flags=shellcon.FOF_NOCONFIRMATION):
    """ Move files using the built in Windows File operations dialog
    
    Requires absolute paths. Does NOT create root destination folder if it doesn't exist.
    
    @see http://msdn.microsoft.com/en-us/library/bb775799(v=vs.85).aspx for flags available
    
    """
    return _file_operation(src,dst,'move',flags)

def delete(path,flags=shellcon.FOF_NOCONFIRMATION):
    """ Delete files using the built in Windows File operations dialog
    
    Requires absolute paths.
    
    @see http://msdn.microsoft.com/en-us/library/bb775799(v=vs.85).aspx for flags available
    
    """
    return _file_operation(path,None,'delete',flags)

def discrepant_files(source_path,target_path):
    list_source = set(os.listdir(source_path))
    list_target = set(os.listdir(target_path))
    list_intersection = list_source.intersection(list_target)
    for item in list_intersection:
        if os.path.getsize(source_path+os.sep+item)==os.path.getsize(source_path+os.sep+item):
            list_source.discard(item)
    return list_source
def sync(source_path,target_path,delete_source,force):
    if os.path.exists(source_path):
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        list_source = set(os.listdir(source_path))
        if force is False:
            list_source = discrepant_files(source_path,target_path)
        source_list = [source_path+os.sep+item for item in list_source]
        #target_list = [target_path+os.sep+item for item in list_source]
        if len(source_list)!=0:
            success = copy(source_list,target_path)
        else:
            success = True
        if success:
            list_source = discrepant_files(source_path,target_path)
            success = len(list_source)==0
        if success and delete_source:
            delete(source_path)
    
    
if __name__ == "__main__":
    import sys
    source_path = sys.argv[1]
    target_path = sys.argv[2]
    delete_source = eval(sys.argv[3])
    force = False
    if len(sys.argv)>4:
        force = eval(sys.argv[4])
    sync(source_path,target_path,delete_source,force)
