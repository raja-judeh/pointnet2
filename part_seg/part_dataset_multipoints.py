'''
    Dataset for ShapeNetPart segmentation
'''

import numpy as np
import itertools
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
    def __init__(self, root, npoints=2500, npoint_pairs=100 ,split='train', normalize=True):
        self.npoints = npoints
        self.npoint_pairs = npoint_pairs
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        
        self.normalize = normalize
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k:v for k,v in list(self.cat.items())}
        self.all_cat = self.cat
            
        self.meta = {}
        # Note: JSON object is unordered collection
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:

            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            #print(fns[0][0:-4])
            if split=='trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split=='train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split=='val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split=='test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print(('Unknown split: %s. Exiting..'%(split)))
                exit(-1)
                
            #print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0]) 
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))
        
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))
            
        # sorted(self.cat) is needed to assign the same class numbers to caategories each time
        self.classes = dict(list(zip(sorted(self.all_cat), list(range(len(self.all_cat))))))  
        
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        for cat in sorted(self.seg_classes.keys()):
            print((cat, self.seg_classes[cat]))
        
        self.cache = {} # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def _get_rand_pp(self, seg, mapping_idx):
        parts, counts = np.unique(seg, return_counts=True) # get only the parts that are present in the given object
        parts = parts[counts>10] # don't consider the noisy labels
        nparts = len(parts)
        n = self.npoint_pairs # number of point-pairs per part

        if nparts == 1:    
            # only similar points exist
            idxs = np.squeeze(np.argwhere(seg==parts))
            pp_idx = np.random.choice(idxs, (n,2), replace=True) 
            pp_label = np.array([1]*n)
        else:
            similar_points = np.zeros((n,nparts,2))
            dissimilar_points = np.zeros((n,nparts))

            for part_idx, part in enumerate(parts):
                idxs = np.squeeze(np.argwhere(seg==part))
                similar_points[:,part_idx] = np.random.choice(idxs, (n,2), replace=True)
                dissimilar_points[:,part_idx] = np.random.choice(idxs, (n,), replace=True)

            # similar pairs
            similar_pairs = np.reshape(similar_points, (-1,2)) # (n*nparts,2)
            similar_labels = np.array([1]*(n*nparts))

            # dissimilar pairs
            part_combinations = list(itertools.combinations(np.arange(nparts),2))
            ncombs = len(part_combinations)

            dissimilar_pairs = dissimilar_points[:,part_combinations]
            dissimilar_pairs = np.reshape(dissimilar_pairs, (-1,2)) # (n*ncombs,2)
            dissimilar_labels = np.array([0]*(n*ncombs))

            # concatenate similar and dissimilar pairs
            pp_idx = np.concatenate([similar_pairs, dissimilar_pairs], axis=0) # (n*ncombs + n*nparts,2)
            pp_label = np.concatenate([similar_labels, dissimilar_labels]) #(n*ncombs + n*nparts)
        
        # map indices from original point cloud to the resampled point cloud
        sort_idx = mapping_idx.argsort()

        pp_idx_mapped1 = np.searchsorted(mapping_idx, pp_idx[:,0], sorter=sort_idx)
        pp_idx_mapped1 = np.take(sort_idx, pp_idx_mapped1, mode='clip')
        mask1 = mapping_idx[pp_idx_mapped1] == pp_idx[:,0]

        pp_idx_mapped2 = np.searchsorted(mapping_idx,pp_idx[:,1], sorter=sort_idx)
        pp_idx_mapped2 = np.take(sort_idx, pp_idx_mapped2, mode='clip')
        mask2 = mapping_idx[pp_idx_mapped2] == pp_idx[:,1]

        pp_idx_mapped = np.stack([pp_idx_mapped1,pp_idx_mapped2], axis=1)

        mask = mask1 & mask2
        pp_idx = pp_idx_mapped[mask]
        pp_label = pp_label[mask]

      # resample and shuffle
        choice = np.resize(np.arange(len(pp_idx)), n*15 + n*6) # all resampled to the same number of point-pairs that will be retrieved from the Motorbike
        _ = np.random.shuffle(choice)
        pp_idx = pp_idx[choice]
        pp_label = pp_label[choice]

        return pp_idx, pp_label
    
 
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
       
        orig_seg = seg # store the labels of the original point cloud before resampling

        # resample the point cloud to the desired number of points
        mapping_idx = np.random.choice(len(orig_seg), self.npoints) # indices to retireive the final resample point cloud
        point_set = point_set[mapping_idx,:]
        normal = normal[mapping_idx,:]
        seg = orig_seg[mapping_idx]

        # generate random point pairs
        pp_idx, pp_label = self._get_rand_pp(orig_seg, mapping_idx) #indices are mapped to the final retrieved point cloud

        return point_set, normal, seg, cls, pp_idx, pp_label

    
    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    d = PartNormalDataset(root = '/volume/USERSTORE/jude_ra/master_thesis/pointnet2/data/shapenetcore_partanno_segmentation_benchmark_v0_normal', split='trainval', npoints=3000, npoint_pairs=2000)
    print((len(d)))

    i = 500
    ps, normal, seg, cls, pp_idx, pp_label = d[i]
    print(seg.shape,pp_idx.shape,pp_label.shape)
    
  

