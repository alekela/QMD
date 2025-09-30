from .field import fields_list
from .field import fields_dtype

import numpy as np

class Storage:    
    
    def __init__(self, fields = [], size = 0):
        self.data = dict()
        self.size = size
        self.fields = []
        for field in fields:
            if field in fields_list:
                self.data[field] = np.zeros(size, dtype=fields_dtype[field])
            else:
                raise Exception('Field ', field, ' is not found in fields list')

        for field in self.data:
            self.fields.append(field)
            
    def load(self, data):
        self.data = dict()
        self.size = 0
        self.fields = []
        k = 0 
        if type(data) == dict:
            for field in data.keys():
                if field in fields_list:
                    self.data[field] = np.array(data[field], dtype=fields_dtype[field])
                    self.size = len(data[field])
                    self.fields.append(field)
                else:
                    raise Exception('Field ', field, ' is not found in fields list')
        elif type(data) == list:
            self.size = len(data)
            for elem in data:
                for field in elem:
                    if field in fields_list:
                        if field not in self.data:
                            self.data[field] = []
                        self.data[field].append(elem[field])
                    else:
                        raise Exception('Field ', field, ' is not found in fields list')
        
            for field in self.data:
                self.data[field] = np.array(self.data[field], dtype=fields_dtype[field])
                self.fields.append(field)
            
    def resize(self, newsize):
        for field in self.data:
                self.data[field].resize(newsize)

        self.size = newsize
            
    def as_dict(self):
        d = dict()
        d.update(self.data)
        return d

    def __getitem__(self, idx_field):
        if isinstance(idx_field, int) or isinstance(idx_field, np.int64):
            item = {}
            for field in self.fields:
                item[field] = self.data[field][idx_field]
            return item
        else:
            return self.data[idx_field]
    
    def __setitem__(self, key, value):
        if self.size == len(value):
            self.data[key] = value
        else:
            raise Exception('Storage: assigning field with wrong size!')
            
    def __iter__ (self):
        self.n = 0
        return self
    
    def __next__(self):
        if self.n < self.size:
            item = {}
            for field in self.fields:
                item[field] = self.data[field][self.n]
            self.n += 1
            return item
        else:
            raise StopIteration
