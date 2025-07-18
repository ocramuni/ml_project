import numpy as np
import os
from torch.utils.data import Dataset

class SmallNorbDataset(Dataset):
    """small NORB dataset."""
    def __init__(self, transform=None):
        """
        Arguments:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images, self.info = self.get_all()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.images[idx]
        label = self.info[idx][0]
        azimuth = self.info[idx][3]

        if self.transform:
            images = self.transform(images[0]), self.transform(images[1])

        return images, label, azimuth

    def getPath(self, which_set, filetype, dirname='data'):
        """
        Getting the path for the desired dataset.
        which_set: train, test
        filetype: dat, cat, info
        """     
        if which_set == 'train':
            instance_list = '46789'
        elif which_set == 'test':
            instance_list = '01235'
        filename = 'smallnorb-5x%sx9x18x6x2x96x96-%s-%s.mat' % \
            (instance_list, which_set + 'ing', filetype)
        return os.path.join(dirname, filename)


    def readNums(self, file_handle, num_type, count):
        """
        Reads 4 bytes from file, returns it as a 32-bit integer.
        """
        num_bytes = count * np.dtype(num_type).itemsize
        string = file_handle.read(num_bytes)
        return np.fromstring(string, dtype=num_type)


    def readHeader(self, file_handle, debug=False):
        """
        Reads the header of the file.
        file_handle: an open file handle.
        returns: data type, element size, rank, shape, size
        """
    
        key_to_type = {0x1E3D4C51: ('float32', 4),
                       # 0x1E3D4C52 : ('packed matrix', 0),
                       0x1E3D4C53: ('float64', 8),
                       0x1E3D4C54: ('int32', 4),
                       0x1E3D4C55: ('uint8', 1),
                       0x1E3D4C56: ('int16', 2)}
    
        type_key = self.readNums(file_handle, 'int32', 1)[0]
        elem_type, elem_size = key_to_type[type_key]
        if debug:
            print("header's type key, type, type size: ",
                  type_key, elem_type, elem_size)
        if elem_type == 'packed matrix':
            raise NotImplementedError('packed matrix not supported')
    
        num_dims = self.readNums(file_handle, 'int32', 1)[0]
        if debug:
            print('# of dimensions, according to header: ', num_dims)
    
        shape = np.fromfile(file_handle,
                            dtype='int32',
                            count=max(num_dims, 3))[:num_dims]
    
        if debug:
            print('Tensor shape, as listed in header:', shape)
    
        return elem_type, elem_size, shape


    def parseNORBFile(self, file_handle, debug=False):
        """
        Parse file into numpy array and return.
        file_handle: an open file handle.
        """
        elem_type, elem_size, shape = self.readHeader(file_handle, debug)
        beginning = file_handle.tell()
        num_elems = np.prod(shape)
        result = np.fromfile(file_handle,
                             dtype=elem_type,
                             count=num_elems).reshape(shape)
        return result

    def sort_info(self, data, info):
        feat_order = info.view('i4,i4,i4,i4,i4').argsort(kind='mergesort', order=['f0', 'f3', 'f2', 'f1', 'f4'], axis=0).reshape(-1)
        data, info = data[feat_order], info[feat_order]
        return data, info

    def get_data(self, which_set='train', file_type='dat', path='data', debug=True):
        file_path = self.getPath(which_set, file_type, path)
        file_handle = open(file_path, 'rb')
        norb_data = self.parseNORBFile(file_handle, debug)
        return norb_data

    def get_all(self):
        dataset_path = "dataset/smallnorb/"
        train_info = self.get_data('train', 'info', dataset_path, debug=False)
        raw_train_data = self.get_data('train', 'dat', dataset_path, debug=False)
        raw_train_label = self.get_data('train', 'cat', dataset_path, debug=False)
        
        test_info = self.get_data('test', 'info', dataset_path, debug=False)
        raw_test_data = self.get_data('test', 'dat', dataset_path, debug=False)
        raw_test_label = self.get_data('test', 'cat', dataset_path, debug=False)
        
        train_info = np.concatenate([raw_train_label[:, np.newaxis], train_info], axis=-1) # add label as first column on train_info
        test_info = np.concatenate([raw_test_label[:, np.newaxis], test_info], axis=-1) # add label as first column on test_info
        
        data = np.concatenate([raw_train_data, raw_test_data], axis=0) # 48600x2x96x96
        info = np.concatenate([train_info, test_info], axis=0) # 48600x5
        
        all_data, all_info = self.sort_info(data, info)
        return all_data, all_info


