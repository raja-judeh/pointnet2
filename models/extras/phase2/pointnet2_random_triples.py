import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module_rand_tree_triples, pointnet_sa_module_msg_rand_tree_triples, pointnet_fp_module

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    cls_labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl, cls_labels_pl

NUM_CATEGORIES = 16

def get_model(point_cloud, cls_label, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx6, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,3])

    cls_label_one_hot = tf.one_hot(cls_label, depth=NUM_CATEGORIES, on_value=1.0, off_value=0.0)
    cls_label_one_hot = tf.reshape(cls_label_one_hot, [batch_size, 1, NUM_CATEGORIES])
    cls_label_one_hot = tf.tile(cls_label_one_hot, [1,num_point,1])

    # Set abstraction layers
    l1_xyz, l1_points = pointnet_sa_module_msg_rand_tree_triples(l0_xyz, l0_points, 512, [0.1,0.2,0.4], [27,81,243], [[32,32,64], [32,32,48,64],[32,32,48,64,64]], is_training, bn_decay, scope='layer1', nshuffles=1)
    l2_xyz, l2_points = pointnet_sa_module_msg_rand_tree_triples(l1_xyz, l1_points, 256, [0.4,0.5], [27,81], [[64,64,128], [64,64,96,128]], is_training, bn_decay, scope='layer2', nshuffles=1)
    l3_xyz, l3_points = pointnet_sa_module_msg_rand_tree_triples(l2_xyz, l2_points, 128, [0.5,0.6], [27,81], [[128,128,256],[128,128,196,256]], is_training, bn_decay, scope='layer3', nshuffles=1)
    l4_xyz, l4_points = pointnet_sa_module_msg_rand_tree_triples(l3_xyz, l3_points, 81, [0.6,0.8], [27,81], [[256,256,512],[256,256,384,512]], is_training, bn_decay, scope='layer4', nshuffles=1)
    l5_xyz, l5_points, l5_indices = pointnet_sa_module_rand_tree_triples(l4_xyz, l4_points, npoint=None, radius=None, nsample=None, mlp=[512,512,768,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer5')

    # Feature propagation layers
    l4_points = pointnet_fp_module(l4_xyz, l5_xyz, l4_points, l5_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='fa_layer2')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer3')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer4')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([cls_label_one_hot, l0_xyz, l0_points],axis=-1), l1_points, [128,128], is_training, bn_decay, scope='fp_layer5')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['feats'] = net 
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, 50, 1, padding='VALID', activation_fn=None, scope='fc2')

    return net, end_points


def get_loss(pred, label, end_points):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,6))
        cls_labels = tf.zeros((32),dtype=tf.int32)
        output, ep = get_model(inputs, cls_labels, tf.constant(True))
        print(output)
