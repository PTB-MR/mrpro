from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Union, Tuple, Sequence, Optional

import ismrmrd
import numpy as np
import torch

import re, os

ACQ_FLAGS = ('ACQ_NO_FLAG',
    'ACQ_FIRST_IN_ENCODE_STEP1',                
    'ACQ_LAST_IN_ENCODE_STEP1',                        
    'ACQ_FIRST_IN_ENCODE_STEP2',                          
    'ACQ_LAST_IN_ENCODE_STEP2',                          
    'ACQ_FIRST_IN_AVERAGE',                       
    'ACQ_LAST_IN_AVERAGE',
    'ACQ_FIRST_IN_SLICE',
    'ACQ_LAST_IN_SLICE', 
    'ACQ_FIRST_IN_CONTRAST',
    'ACQ_LAST_IN_CONTRAST', 
    'ACQ_FIRST_IN_PHASE',   
    'ACQ_LAST_IN_PHASE',    
    'ACQ_FIRST_IN_REPETITION',
    'ACQ_LAST_IN_REPETITION', 
    'ACQ_FIRST_IN_SET',     
    'ACQ_LAST_IN_SET',      
    'ACQ_FIRST_IN_SEGMENT', 
    'ACQ_LAST_IN_SEGMENT',  
    'ACQ_IS_NOISE_MEASUREMENT',  
    'ACQ_IS_PARALLEL_CALIBRATION', 
    'ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING', 
    'ACQ_IS_REVERSE',        
    'ACQ_IS_NAVIGATION_DATA',  
    'ACQ_IS_PHASECORR_DATA',      
    'ACQ_LAST_IN_MEASUREMENT',  
    'ACQ_IS_HPFEEDBACK_DATA',  
    'ACQ_IS_DUMMYSCAN_DATA', 
    'ACQ_IS_RTFEEDBACK_DATA',  
    'ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA', 
    'ACQ_USER1',                               
    'ACQ_USER2',                               
    'ACQ_USER3',                               
    'ACQ_USER4',                               
    'ACQ_USER5',                               
    'ACQ_USER6',                               
    'ACQ_USER7',                               
    'ACQ_USER8')

class ELimits():
    __slots__ = ("kspace_encoding_step_1",
                "kspace_encoding_step_2",
                "average",
                "slice",
                "contrast",
                "phase",
                "repetition",
                "set",
                "segment")
    
    def __init__(self) -> None:
        pass
    

class AcqInfo():
    __slots__ = ("acquisition_time_stamp",
                 "active_channels",
                 "available_channels",
                 "average",
                 "center_sample",
                 "channel_mask",
                 "contrast",
                 "discard_post",
                 "discard_pre",
                 "encoding_space_ref",
                 "flags",
                 "kspace_encode_step_1",
                 "kspace_encode_step_2",
                 "measurement_uid",
                 "number_of_samples",
                 "patient_table_position",
                 "phase",
                 "phase_dir",
                 "physiology_time_stamp",
                 "position",
                 "read_dir",
                 "repetition",
                 "sample_time_us",
                 "scan_counter",
                 "segment",
                 "set",
                 "slice",
                 "slice_dir",
                 "trajectory_dimensions",
                 "user_float",
                 "user_int",
                 "version",
                 )

    def __init__(self, num_acq: int) -> None:
        for slot in self.__slots__:
            if slot == "physiology_time_stamp":
                setattr(self, slot, torch.zeros((num_acq,ismrmrd.constants.PHYS_STAMPS), dtype=torch.float32))
            elif slot =="channel_mask":
                setattr(self, slot, torch.zeros((num_acq,ismrmrd.constants.CHANNEL_MASKS), dtype=torch.int64))
            elif slot == "position" or slot == "read_dir" or slot == "phase_dir" or slot == "slice_dir":
                setattr(self, slot, torch.zeros((num_acq,ismrmrd.constants.DIRECTION_LENGTH), dtype=torch.float32))
            elif slot == "patient_table_position":
                setattr(self, slot, torch.zeros((num_acq,ismrmrd.constants.POSITION_LENGTH), dtype=torch.float32))
            elif slot == "user_int":
                setattr(self, slot, torch.zeros((num_acq,ismrmrd.constants.USER_INTS), dtype=torch.int64))
            elif slot == "user_float":
                setattr(self, slot, torch.zeros((num_acq,ismrmrd.constants.USER_FLOATS), dtype=torch.float32))
            else:
                setattr(self, slot, torch.zeros((num_acq,), dtype=torch.int64))

    def from_ismrmrd_acq_header(self, curr_idx: int, acq: ismrmrd.Acquisition) -> None:
        for slot in self.__slots__:
            curr_attr = getattr(self, slot)
            if slot in ("kspace_encode_step_1", "kspace_encode_step_2", "average", "slice", 
                        "contrast", "phase", "repetition", "set", "segment"):
                curr_attr[curr_idx,...] = torch.tensor(getattr(acq.idx, slot), dtype=curr_attr.dtype)
                
            else:
                curr_attr[curr_idx,...] = torch.tensor(getattr(acq, slot), dtype=curr_attr.dtype)
            setattr(self, slot, curr_attr)

def _return_par_tensor(par, array_attr) -> Union(torch.tensor, None):
        if par is None:
            return None
        else:
            par_tensor = []
            for attr in array_attr:
                par_tensor.append(getattr(par, attr))
            return(torch.tensor(par_tensor))
    
def return_par_matrix_tensor(par: ismrmrd.xsd.ismrmrdschema.ismrmrd.matrixSizeType) -> Union(torch.tensor, None):
    return(_return_par_tensor(par, array_attr=("x", "y", "z")))

def return_par_enc_limits_tensor(par: ismrmrd.xsd.ismrmrdschema.ismrmrd.limitType) -> Union(torch.tensor, None):
    return(_return_par_tensor(par, array_attr=("minimum", "maximum", "center")))

def return_acc_factor_tensor(par: ismrmrd.xsd.ismrmrdschema.ismrmrd.accelerationFactorType) -> Union(torch.tensor, None):
    return(_return_par_tensor(par, array_attr=("kspace_encoding_step_1", "kspace_encoding_step_2")))

def bitmask_flag_to_strings(flag: int):
    if flag > 0:
        bmask = "{0:064b}".format(flag)
        bitmask_idx = [m.start() + 1 for m in re.finditer('1', bmask[::-1])]  
    else:
        bitmask_idx = [0, ]
    flag_strings = []
    for knd in range(len(bitmask_idx)):
        flag_strings.append(ACQ_FLAGS[bitmask_idx[knd]])
    return(flag_strings)
        
class KHeader():
    __slots__ = ("protocol_name", "pat_pos", "meas_id", "institution", "receiver_noise_bwdth", "b0", "model", "vendor",
    "elimits", "acc_factor", "rec_matrix", "rec_fov", "enc_matrix", "enc_fov", "etl", "num_coils", "acq_info")

    def __init__(self) -> None:
        pass
 
    def from_ismrmrd_header(self, header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader) -> None:

        # Encoding
        assert len(header.encoding) == 1, "Multiple encodings are not supported."
        enc = header.encoding[0]
        self.etl = enc.echoTrainLength
        self.enc_fov = return_par_matrix_tensor(enc.encodedSpace.fieldOfView_mm)
        self.enc_matrix = return_par_matrix_tensor(enc.encodedSpace.matrixSize)
        self.rec_fov = return_par_matrix_tensor(enc.reconSpace.fieldOfView_mm)
        self.rec_matrix = return_par_matrix_tensor(enc.reconSpace.matrixSize)
        self.acc_factor = return_acc_factor_tensor(enc.parallelImaging.accelerationFactor)
        
        self.elimits = ELimits()
        for climit in self.elimits.__slots__:
            setattr(self.elimits, climit, return_par_enc_limits_tensor(getattr(enc.encodingLimits, climit)))
            
        # AcquisitionSystemInformation
        self.vendor = header.acquisitionSystemInformation.systemVendor
        self.model = header.acquisitionSystemInformation.systemModel
        self.b0 = header.acquisitionSystemInformation.systemFieldStrength_T
        self.receiver_noise_bwdth = header.acquisitionSystemInformation.relativeReceiverNoiseBandwidth
        self.institution = header.acquisitionSystemInformation.institutionName
        self.num_coils = header.acquisitionSystemInformation.receiverChannels
        
        # MeasurementInformation
        self.meas_id = header.measurementInformation.measurementID
        self.pat_pos = header.measurementInformation.patientPosition.value
        self.protocol_name  = header.measurementInformation.protocolName

class KData():
    def __init__(self,
                 header: KHeader,
                 data: torch.Tensor,
                 traj: torch.Tensor) -> None:
        self.header: KHeader = header
        self._data: torch.Tensor = data
        self._traj: torch.tensor = traj

    @classmethod
    def from_file(cls,
                  filename: Union[str, Path],
                  ktrajectory_calculator: KTrajectory) -> KData:
        
        # Check file is valid
        if not os.path.isfile(filename):
            print("%s is not a valid file" % filename)
            raise SystemExit
        
        # Read header
        dset = ismrmrd.Dataset(filename, "dataset", create_if_needed=False)
        hdr_xml = dset.read_xml_header()
        hdr = ismrmrd.xsd.CreateFromDocument(hdr_xml)
        dset.close()
        
        kheader = KHeader()
        kheader.from_ismrmrd_header(hdr)
        
        # Read k-space data
        with ismrmrd.File(filename) as mrd:
            acqs = mrd["dataset"].acquisitions[:]
  
        # Get indices for imaging data
        im_idx = []
        unique_acq_flags = set()
        for idx, acq in enumerate(acqs):
            for el in bitmask_flag_to_strings(acq.flags):
                unique_acq_flags.add(el)
            if 'ACQ_IS_NOISE_MEASUREMENT' not in bitmask_flag_to_strings(acq.flags):
                im_idx.append(idx)
                
        # Create AcqInfo
        kheader.acq_info = AcqInfo(len(im_idx))
                
        # Get k-space data
        kdata = torch.zeros((len(im_idx), kheader.num_coils, acqs[im_idx[0]].number_of_samples), dtype=torch.complex64)
        for idx in range(len(im_idx)):
            acq = acqs[im_idx[idx]]
            kdata[idx,:,:] = torch.tensor(acq.data, dtype=torch.complex64)
            # TODO: Make this faster
            kheader.acq_info.from_ismrmrd_acq_header(idx, acq)
            
        # Calculate trajectory
        ktraj = ktrajectory_calculator.calc_traj(kheader)
        
            
        # TODO: Check for partial Fourier and reflected readouts
        
        # Sort k-space data into (dim4, ncoils, k2, k1, k0)
        kdim_labels = ("kspace_encode_step_1", "kspace_encode_step_2", "average", "slice", "contrast", "phase", "repetition", "set")
        kdim_num = np.asarray([len(np.unique(getattr(kheader.acq_info, acq_label))) for acq_label in kdim_labels])
        sort_ki = np.stack((kheader.acq_info.kspace_encode_step_1, kheader.acq_info.kspace_encode_step_2, kheader.acq_info.average, 
                                    kheader.acq_info.slice, kheader.acq_info.contrast, kheader.acq_info.phase, 
                                    kheader.acq_info.repetition, kheader.acq_info.set), axis=0)
        sort_idx = np.lexsort(sort_ki)

        #TODO: Ensure each dim4 covers the same ky-kz positions
        
        new_shape = (np.prod(kdim_num[2:]), kdim_num[1], kdim_num[0],)
        kdata = torch.reshape(kdata[sort_idx, :, :], new_shape + kdata.shape[1:])
        kdata = torch.moveaxis(kdata, (0,1,2,3,4), (0,2,3,1,4))
        
        ktraj = torch.reshape(ktraj[sort_idx,:,:], new_shape + ktraj.shape[1:])
        ktraj = torch.moveaxis(ktraj, (0,1,2,3,4), (0,2,3,1,4))
        
        for slot in kheader.acq_info.__slots__:
            curr_attr = getattr(kheader.acq_info, slot)
            curr_shape = new_shape
            if curr_attr.ndim == 2:
                curr_shape += (curr_attr.shape[1],)
            setattr(kheader.acq_info, slot, np.reshape(curr_attr[sort_idx,...], curr_shape))

        return cls(kheader, kdata, ktraj)

    @property
    def traj(self) -> torch.Tensor:
        return self._traj

    @traj.setter
    def traj(self, value: torch.Tensor):
        self._traj = value

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @data.setter
    def data(self, value: torch.Tensor):
        self._data = value
