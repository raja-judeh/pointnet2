import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from multipoints_util import pointnet_sa_module, pointnet_fp_module, pointnet_branch


def placeholder_inputs(batch_size, num_point, num_point_pairs):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    cls_labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    pp_idx_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point_pairs, 2))
    pp_labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point_pairs))

    return pointclouds_pl, labels_pl, cls_labels_pl, pp_idx_pl, pp_labels_pl


def get_pp_pred(feats, pp_idx, bn_decay, is_training):
    batch_size = pp_idx.get_shape()[0].value
    num_point_pairs = pp_idx.get_shape()[1].value

    # gather point-pairs
    batch_idx = np.arange(batch_size)
    batch_idx = np.reshape(batch_idx, (batch_size,1))
    batch_idx = np.tile(batch_idx, 2*num_point_pairs)
    batch_idx = np.reshape(batch_idx, (batch_size,num_point_pairs,2))
    pp_idx = tf.stack([batch_idx,pp_idx], axis=-1)
    pp_feats = tf.gather_nd(feats, pp_idx) # (batch_size,num_point_pairs,2,128)

    # predict similarity labels of the point-pairs
    pp_feats = tf_util.conv2d(pp_feats, 128, [1,2],
                              padding='VALID', stride=[1,1],
                              bn=True, is_training=is_training,
                              scope='pp_fc1', bn_decay=bn_decay,
                              data_format='NHWC') 

    pp_feats = tf.squeeze(pp_feats, [2]) # (batch_size,num_point_pairs,128)
    pp_feats = tf_util.dropout(pp_feats, keep_prob=0.5, is_training=is_training, scope='pp_dp')

    pp_pred = tf_util.conv1d(pp_feats, 1, 1, padding='VALID', activation_fn=None, scope='pp_fc2')
    pp_pred = tf.squeeze(pp_pred, axis=-1)

    return pp_pred


NUM_CATEGORIES = 16

def get_model(point_cloud, cls_label, pp_idx, is_training, bn_decay=None):
    """ Part segmentation PointNet, input is BxNx6 (XYZ NormalX NormalY NormalZ), output Bx50 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    cls_label_one_hot = tf.one_hot(cls_label, depth=NUM_CATEGORIES, on_value=1.0, off_value=0.0)
    cls_label_one_hot = tf.reshape(cls_label_one_hot, [batch_size, 1, NUM_CATEGORIES])
    cls_label_one_hot = tf.tile(cls_label_one_hot, [1,num_point,1])

    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,3])

    # PoinNet branch
    l00_points = pointnet_branch(point_cloud, [32,32,64,128,256], [128,128,128], is_training, bn_decay=bn_decay)

    # Set Abstraction layers
    l1_xyz, l1_points, _ = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, _ = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')

    # Feature Propagation layers
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer2')   
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([cls_label_one_hot,l0_xyz,l0_points,l00_points], axis=-1), l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer3')

    # FC layers
    end_points['feats'] = l0_points
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')

    pred = tf_util.conv1d(net, 50, 1, padding='VALID', activation_fn=None, scope='fc2')
    pp_pred = get_pp_pred(end_points['feats'], pp_idx, bn_decay, is_training)

    return pred, pp_pred, end_points


def get_loss(pred, label, pp_pred, pp_label, end_points):
    """ pred: BxNxC,
        label: BxN, """

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    pp_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pp_pred, labels=pp_label)

    loss = tf.reduce_mean(loss)
    pp_loss = tf.reduce_mean(pp_loss)

    total_loss = loss + pp_loss

    tf.summary.scalar('classify loss', total_loss)
    tf.add_to_collection('losses', total_loss)

    return total_loss, pp_loss



if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,6))
        cls_labels = tf.zeros((32),dtype=tf.int32)
        output, ep = get_model(inputs, cls_labels, tf.constant(True))
        print(output)
