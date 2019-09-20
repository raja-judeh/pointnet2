import argparse
import math
from datetime import datetime
import h5py

import numpy as np
import tensorflow as tf
import itertools
import pickle

import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = '/volume/USERSTORE/jude_ra/master_thesis/pointnet2/data/'
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
import part_dataset_all_normal, part_dataset_all_normal_rotated

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_part_seg_msg_one_hot', help='Model name [default: pointnet2_part_seg]')
parser.add_argument('--model_path', default='log_folders/log_msg_one_hot/model.ckpt', help='model checkpoint file path')
parser.add_argument('--log_dir', default='log_folders/log_eval_msg_one_hot', help='Log dir')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--test_rotated', type=bool, default=False, help='test on rotated data')
parser.add_argument('--recentering', type=bool, default=False, help='recenter data after occlusion')
FLAGS = parser.parse_args()

GPU_INDEX = FLAGS.gpu

BATCH_SIZE = 1
NUM_POINT = FLAGS.num_point

VIEW_AXES = list(itertools.product(*zip([-1]*3,[1]*3)))
NUM_VIEW_AXES = len(VIEW_AXES)

MODEL_PATH = FLAGS.model_path
MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')

TEST_ROTATED = FLAGS.test_rotated
RECENTERING = FLAGS.recentering

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
NUM_CLASSES = 50

if TEST_ROTATED:
    DATA_PATH = os.path.join(DATA_DIR, 'shapenetcore_normal_rotated')
    TEST_DATASET = part_dataset_all_normal_rotated.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, split='test_rand', return_cls_label=True, random_sampling=True)
else:
    # Shapenet official train/test split
    DATA_PATH = os.path.join(DATA_DIR, 'shapenetcore_partanno_segmentation_benchmark_v0_normal')
    TEST_DATASET = part_dataset_all_normal.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, split='test', return_cls_label=True, random_sampling=False)


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def pc_centering(pc):
    l = pc.shape[0]
    centroid = np.mean(pc[:,:3], axis=0)
    pc[:,:3] = pc[:,:3] - centroid
    return pc


# Occlusion function
def apply_occlusion(pc, view_axis, resample=False, return_indices=True):
    '''
    Input:
        pc: point cloud (num_points,num_feats)
        view_axis: axis of view (3,)
        resample: resample each occluded point cloud to the original number of points
    Output:
        pcs_occ: occluded (resampled) point clouds with 0% - 90% (10,num_points,num_feats)
    '''
    
    num_points = pc.shape[0]
    
    n = view_axis / np.linalg.norm(view_axis) #normal of the plane
    d = -np.dot(pc[:,:3],n) #distance from every point to a plane passes in the origin (projection of points into the normal)
    
    qs = np.arange(0,1,0.1) #sequence of quantiles
    qvs = np.quantile(d,qs) #sequence of quantile values
    
    pcs_occ = [pc[d>=qv,:] for qv in qvs] #occluded point clouds
    indices = [np.squeeze(np.argwhere(d>=qv)) for qv in qvs]
    
    if resample:
        pcs_occ = [np.resize(pc_occ, (num_points,pc_occ.shape[1])) for pc_occ in pcs_occ]
        indices = [np.resize(idx, num_points) for idx in indices]
    
    if return_indices:
        return pcs_occ, indices
    else:
        return pcs_occ


def evaluate():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, cls_labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            print("--- Get model and loss")
            pred, end_points = MODEL.get_model(pointclouds_pl, cls_labels_pl, is_training_pl)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'output'), sess.graph)

        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)

        #print(tf.trainable_variables())

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'cls_labels_pl': cls_labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss}

        eval_one_epoch(sess, ops)
        writer.close()


def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx #always equals 1 because BATCH_SIZE=1
    batch_data = np.zeros((bsize, NUM_POINT, 6))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_cls_label = np.zeros((bsize,), dtype=np.int32)
    for i in range(bsize):
        ps,normal,seg,cls = dataset[idxs[i+start_idx]]
        batch_data[i,:,0:3] = ps
        batch_data[i,:,3:6] = normal
        batch_label[i,:] = seg
        batch_cls_label[i] = cls
    return np.squeeze(batch_data), np.squeeze(batch_label), np.squeeze(batch_cls_label)


def eval_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = int((len(TEST_DATASET)+BATCH_SIZE-1)/BATCH_SIZE)

    total_correct_occ = np.zeros(10)
    total_seen_occ = np.zeros(10)
    loss_sum_occ = np.zeros(10)
    total_seen_class_occ = np.zeros((10,NUM_CLASSES))
    total_correct_class_occ = np.zeros((10,NUM_CLASSES))

    seg_classes = TEST_DATASET.seg_classes
    shape_ious = {cat:[] for cat in list(seg_classes.keys())}
    seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in list(seg_classes.keys()):
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    log_string(str(datetime.now()))
    
    for batch_idx in range(num_batches):
        if batch_idx %1==0:
            log_string('%03d/%03d'%(batch_idx, num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(len(TEST_DATASET), (batch_idx+1) * BATCH_SIZE)
        cur_batch_size = end_idx-start_idx
        pc_data, pc_label, pc_cls_label = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)

        # ---------------------------------------------------------------------
        pred_val_occ = np.zeros((NUM_VIEW_AXES,10,NUM_POINT,NUM_CLASSES))
        pcs_label_occ = np.zeros((NUM_VIEW_AXES,10,NUM_POINT))
        loss_val_occ = np.zeros((NUM_VIEW_AXES,10))
        for axis_idx, view_axis in enumerate(VIEW_AXES):
            pcs_data_occ, indices_occ = apply_occlusion(pc_data, view_axis, resample=True)  
            for occ_idx, pc_data_occ in enumerate(pcs_data_occ):
                if RECENTERING:
                    pc_data_occ = pc_centering(pc_data_occ)

                pc_label_occ = pc_label[indices_occ[occ_idx]]
                pcs_label_occ[axis_idx,occ_idx,:] = pc_label_occ
                feed_dict = {ops['pointclouds_pl']: np.expand_dims(pc_data_occ,0),
                             ops['labels_pl']: np.expand_dims(pc_label_occ,0),
                             ops['cls_labels_pl']: np.expand_dims(pc_cls_label,0),
                             ops['is_training_pl']: is_training}

                loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
                pred_val_occ[axis_idx,occ_idx,:,:] = np.squeeze(pred_val) 
                loss_val_occ[axis_idx,occ_idx] = loss_val
        # ---------------------------------------------------------------------

        # Constrain pred to the groundtruth classes (selected by seg_classes[cat])
        pred_val_occ_logits = pred_val_occ
        pred_label_occ = np.zeros((NUM_VIEW_AXES,10,NUM_POINT)).astype(np.int32)
        cat = seg_label_to_cat[pcs_label_occ[0,0,0]] #all point clouds have the same category in one batch iter
        for axis_idx in range(NUM_VIEW_AXES):
            for occ_idx in range(10):
                logits = pred_val_occ_logits[axis_idx,occ_idx,:,:]
                plabel = np.argmax(logits[:,seg_classes[cat]], 1) + seg_classes[cat][0]
                pred_label_occ[axis_idx,occ_idx,:] = plabel
    
        # Summed over all batches
        correct_occ = np.sum(pred_label_occ==pcs_label_occ, axis=(0,-1)) #summing over view_axes and points
        total_correct_occ += correct_occ
        total_seen_occ += NUM_VIEW_AXES * NUM_POINT
        loss_sum_occ += np.mean(loss_val_occ, axis=0) #avg over view_axes

        for l in range(NUM_CLASSES):
            total_seen_class_occ[:,l] += np.sum(pcs_label_occ==l, axis=(0,-1))
            total_correct_class_occ[:,l] += (np.sum((pred_label_occ==l) & (pcs_label_occ==l), axis=(0,-1)))

        segl = pcs_label_occ 
        segp = pred_label_occ
        part_ious = np.zeros((NUM_VIEW_AXES,10,len(seg_classes[cat])))
        for l in seg_classes[cat]:
            for axis_idx in range(NUM_VIEW_AXES):
                for occ_idx in range(10):
                    if (np.sum(segl[axis_idx,occ_idx]==l) == 0) and (np.sum(segp[axis_idx,occ_idx]==l) == 0):
                        # True if union = 0
                        part_ious[axis_idx,occ_idx,l-seg_classes[cat][0]] = np.nan
                    else:
                        intersection = np.sum((segl[axis_idx,occ_idx]==l) & (segp[axis_idx,occ_idx]==l))
                        union = np.sum((segl[axis_idx,occ_idx]==l) | (segp[axis_idx,occ_idx]==l))
                        part_ious[axis_idx,occ_idx,l-seg_classes[cat][0]] =  intersection / union

        part_ious_avg = np.nanmean(part_ious, axis=-1) #avg over all available part classes of the current cat
        shape_ious[cat].append(np.mean(part_ious_avg, axis=0)) #avg over view_axes

    # This starts after going over all batches
    all_shape_ious = []
    for cat in list(shape_ious.keys()):
        for ious_occ in shape_ious[cat]:
            all_shape_ious.append(ious_occ)
        shape_ious[cat] = np.mean(np.array(shape_ious[cat]), axis=0) #avg over all batches for each cat

    mean_shape_ious = np.mean(np.array(list(shape_ious.values())), axis=0) #avg over all cats after avg over all batches for each cat
    all_shape_mIoU = np.mean(np.array(all_shape_ious), axis=0) #avg over all batches regardlss of cat

    for i in range(10):
        log_string('Occlusion: %d\n' % i)
        log_string('eval mean loss: %f' % (loss_sum_occ[i] / len(TEST_DATASET)))
        log_string('eval accuracy: %f'% (total_correct_occ[i] / total_seen_occ[i]))
        log_string('eval avg class acc: %f' % (np.mean(total_correct_class_occ[i]/total_seen_class_occ[i])))

        for cat in sorted(shape_ious.keys()):
            if isinstance(shape_ious[cat], np.ndarray):
                log_string('eval mIoU of %s:\t %f'%(cat, shape_ious[cat][i]))
            else:
                log_string('eval mIoU of %s:\t %f'%(cat, shape_ious[cat]))

        log_string('eval mean mIoU: %f' % (mean_shape_ious[i]))

        if isinstance(all_shape_mIoU, np.ndarray):
            log_string('eval mean mIoU (all shapes): %f\n' % all_shape_mIoU[i])
        else:
            log_string('eval mean mIoU (all shapes): %f\n' % all_shape_mIoU)


    # Saving variables
    loss = loss_sum_occ / len(TEST_DATASET)
    acc = total_correct_occ / total_seen_occ
    avg_class_acc = np.mean(total_correct_class_occ[i]/total_seen_class_occ[i])
    
    evaluation = {'loss': loss,
                  'accuracy': acc,
                  'avg_class_acc': avg_class_acc,
                  'shape_iou': shape_ious,
                  'mean_shape_iou': mean_shape_ious,
                  'all_shape_iou': all_shape_mIoU}

    with open(os.path.join(LOG_DIR, 'evaluation.pickle'), 'wb') as f:
        pickle.dump(evaluation, f)

'''
    plt.figure
    plot2 = plt.plot(np.arange(0,1,0.1), loss_sum_occ)
    plt.xlabel('Occlusion Quantile')
    plt.ylabel('Loss')
    plt.title('Overall Loss')
    plt.savefig(os.path.join(LOG_DIR, 'plot_1.png'))
    plt.close()

    plt.figure
    plot1 = plt.plot(np.arange(0,1,0.1), np.mean(np.array(all_shape_ious), 0))
    plt.xlabel('Occlusion Quantile')
    plt.ylabel('IoU')
    plt.title('IoU All Shapes')
    plt.savefig(os.path.join(LOG_DIR, 'plot_2.png'))
    plt.close()

    plt.figure
    for idx, cat in enumerate(list(shape_ious.keys())):
        if isinstance(shape_ious[cat], np.ndarray):
            plot = plt.plot(np.arange(0,1,0.1), shape_ious[cat], label=cat)

    plt.xlabel('Occlusion Quantile')
    plt.ylabel('IoU')
    plt.title('IoU ' + cat)
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'plot' + str(idx) + '.png'))
    plt.close()

'''
         
if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    evaluate()
    LOG_FOUT.close()
