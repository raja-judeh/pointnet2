'''
    Rotated Dataset for ShapeNetPart Segmentation
'''

import os
import os.path
import json
import numpy as np
import sys

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class PartNormalDataset():
    def __init__(self, root, input_cat=None, npoints = 2500, classification = False, split='train', normalize=True, return_cls_label = False, random_sampling=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        
        self.classification = classification
        self.normalize = normalize
        self.return_cls_label = return_cls_label
        self.random_sampling = random_sampling
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k:v for k,v in list(self.cat.items())}
        #print(self.cat)

        if input_cat is not None:
            # Choose only one category
            self.cat = {input_cat: self.cat[input_cat]}
            
        self.meta = {}
        for item in self.cat:
            self.meta[item] = []

            if split=='train':
                dir_point = os.path.join(self.root, 'train', self.cat[item])
            elif split=='test_rand':
                dir_point = os.path.join(self.root, 'test_random', self.cat[item])
            elif split=='test_sys':
                dir_point = os.path.join(self.root, 'test_systematic2', self.cat[item])
            else:
                print(('Unknown split: %s. Exiting..'%(split)))
                exit(-1)    
                
            fns = sorted(os.listdir(dir_point))
                
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0]) 
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))
        
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        # sorted(self.cat) is needed to assign the same class numbers to caategories each time
        self.classes = dict(list(zip(sorted(self.cat), list(range(len(self.cat))))))  

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        for cat in sorted(self.seg_classes.keys()):
            print((cat, self.seg_classes[cat]))
        
        self.cache = {} # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000
        
    def __getitem__(self, index):
        if index in self.cache:
            point_set, normal, seg, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:,0:3]
            if self.normalize:
                point_set = pc_normalize(point_set)
            normal = data[:,3:6]
            seg = data[:,-1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, normal, seg, cls)
                   
        if self.random_sampling:
            choice = np.random.choice(len(seg), self.npoints, replace=True)
        else:
            choice = np.arange(self.npoints)
            choice = choice % len(seg)

        #resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        normal = normal[choice,:]
        if self.classification:
            return point_set, normal, cls
        else:
            if self.return_cls_label:
                return point_set, normal, seg, cls
            else:
                return point_set, normal, seg
        
    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    DATA_DIR = '/volume/USERSTORE/jude_ra/master_thesis/pointnet2'
    DATA_PATH = os.path.join(DATA_DIR, 'data', 'shapenetcore_normal_rotated')
    d = PartNormalDataset(root = DATA_PATH, split='test', npoints=2048, return_cls_label=True)
    print((len(d)))

    i = 400
    ps, normal, seg, cls = d[i]
    print(d.datapath[i])
    print('class label:', cls)
    print(np.max(seg), np.min(seg))
    print((ps.shape, seg.shape, normal.shape))
    
    sys.path.append(os.path.join(ROOT_DIR, 'utils'))
    import show3d_balls
    show3d_balls.showpoints(ps, normal+1, ballradius=8)



