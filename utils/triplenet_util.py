""" PointNet++ Layers

Author: Charles R. Qi
Date: November 2017
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import itertools
import tf_util


def sample_and_group_polygons(npoint, nsample, xyz, points):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    # find seed points
    new_xyz_idx = farthest_point_sample(npoint, xyz) # (batch_size, npoint)
    new_xyz = gather_point(xyz, new_xyz_idx) # (batch_size, npoint, 3)

    # find nearest neighbors and group points
    _, idx = knn_point(nsample, xyz, new_xyz) 
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz = tf.concat((grouped_xyz,tf.expand_dims(new_xyz, axis=2)), axis=2) # (batch_size, npoint, nsample+1, 3)

    # concatenate the indices of the neighbors to the seed indices
    idx = tf.concat((idx, tf.expand_dims(new_xyz_idx, axis=-1)), axis=-1)
    
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample+1, channel)
        new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample+1, 3+channel)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, grouped_xyz


def sample_and_group_all_polygons(nsample, xyz, points):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    ndataset = xyz.get_shape()[1].value

    # find nearest neighbors and group points
    _, idx = knn_point(nsample, xyz, xyz) 
    grouped_xyz = group_point(xyz, idx) # (batch_size, ndataset, nsample, 3)
    grouped_xyz = tf.concat((tf.expand_dims(xyz, axis=2),grouped_xyz), axis=2) # (batch_size, ndataset, nsample+1, 3)

    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, ndataset, nsample, channel)
        grouped_points = tf.concat((tf.expand_dims(points, axis=2),grouped_points), axis=2) # (batch_size, ndataset, nsample+1, channel)
        new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample+1, 3+channel)
    else:
        new_points = grouped_xyz

    return xyz, new_points, grouped_xyz


def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(list(range(nsample))).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, is_training, bn_decay, scope, bn=True, pooling='max', knn=False,  group_all=False, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing 
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format) 
            if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx

def pointnet_sp2_module(xyz, points, npoint, mlp1, mlp2, is_training, bn_decay, k=30, scope='sp_layer', group_all=False, bn=True):
    ''' PointNet SuperPoint (SP) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
    '''
    data_format = 'NHWC'
    with tf.variable_scope(scope) as sc:
        # sampling and grouping
        if group_all:
            new_xyz, new_points, _, grouped_xyz = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points, _, grouped_xyz = sample_and_group(npoint, None, k, xyz, points, knn=True)

        # point feature embedding
        for i, num_out_channel in enumerate(mlp1):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv_point%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 

        # pooling in local regions and concatenate with original points
        global_point = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        global_point = tf.tile(global_point, [1,1,k,1])
        new_points = tf.concat([new_points,global_point], axis=-1) # (B,npoint,k,mlp1[-1]*2)
        print('new_points_before', new_points.get_shape())

        # extracting fused features
        for i, num_out_channel in enumerate(mlp2):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv_fused%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 

        # Applying Attention Mechanism
        B = xyz.get_shape()[0].value # batch_size
        D = mlp2[-1] # dimensionality
        mlp3 = [D,D]

        # identity matrix to remove auto-correlations
        I = tf.ones((B,npoint,k,k))
        I = tf.linalg.set_diag(I, tf.zeros((B,npoint,k))) # (B,npoint,k,k)

        corr = tf.matmul(new_points, tf.transpose(new_points, [0,1,3,2]))
        #corr = corr * I # remove auto-correlations
        corr = tf.math.exp(corr) # (B,npoint,k,k)
        print('corr', corr.get_shape())

        S = tf.reduce_sum(corr, axis=-1, keepdims=True) # (B,npoint,k,1)
        print('S', S.get_shape())

        cm = tf.tile(new_points[:,:,None,:,:], [1,1,k,1,1]) # (B,npoint,k,k,D)
        g = cm * corr[...,None] # (B,npoint,k,k,D)
        print('g', g.get_shape())
        g = tf.reduce_sum(g, axis=3) # (B,npoint,k,D)
        g = g / S
        print('g_norm', g.get_shape())

        new_points += g # (B,npoint,k,D)

        for i, num_out_channel in enumerate(mlp3):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv_final%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 


        new_points = tf.reduce_max(new_points, axis=2) # (B,npoint,D)
        print('new_points', new_points.get_shape())

        return new_xyz, new_points


def pointnet_sp_module(xyz, points, npoint, mlp1, mlp2, is_training, bn_decay, k=30, scope='sp_layer', group_all=False, bn=True):
    ''' PointNet SuperPoint (SP) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
    '''
    data_format = 'NHWC'
    with tf.variable_scope(scope) as sc:
        # sampling and grouping
        if group_all:
            new_xyz, new_points, _, grouped_xyz = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points, _, grouped_xyz = sample_and_group(npoint, None, k, xyz, points, knn=True)

        # point feature embedding
        for i, num_out_channel in enumerate(mlp1):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv_point%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 

        # pooling in local regions and concatenate with original points
        global_point = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        global_point = tf.tile(global_point, [1,1,k,1])
        new_points = tf.concat([new_points,global_point], axis=-1) # (B,npoint,k,mlp1[-1]*2)

        # extracting fused features
        for i, num_out_channel in enumerate(mlp2):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv_fused%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 

        # Applying Attention Mechanism
        B = xyz.get_shape()[0].value # batch_size
        D = mlp2[-1] # dimensionality

        # identity matrix to remove auto-correlations
        I = tf.ones((B,npoint,k,k))
        I = tf.linalg.set_diag(I, tf.zeros((B,npoint,k))) # (B,npoint,k,k)

        weight_ci_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        weights_ci = tf.get_variable('weights_ci', [D,D], initializer=weight_ci_init, 
                                     dtype=tf.float32, trainable=True)
        biases_ci = tf.get_variable('biases_ci', [D], dtype=tf.float32, 
                                    trainable=True, initializer=tf.constant_initializer(0.0))
        ci = tf.matmul(new_points, weights_ci)
        ci = tf.nn.bias_add(ci, biases_ci) # (B,npoint,k,D)

        weight_cn_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        weights_cn = tf.get_variable('weights_cn', [D,D], initializer=weight_cn_init,
                                     dtype=tf.float32, trainable=True)
        biases_cn = tf.get_variable('biases_cn', [D], dtype=tf.float32, 
                                    trainable=True, initializer=tf.constant_initializer(0.0))
        cn = tf.matmul(new_points, weights_cn)
        cn = tf.nn.bias_add(cn, biases_cn) # (B,npoint,k,D)

        weight_cm_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        weights_cm = tf.get_variable('weights_cm', [D,D], initializer=weight_cm_init,
                                     dtype=tf.float32, trainable=True)
        biases_cm = tf.get_variable('biases_cm', [D], dtype=tf.float32, 
                                    trainable=True, initializer=tf.constant_initializer(0.0))
        cm = tf.matmul(new_points, weights_cm)
        cm = tf.nn.bias_add(cm, biases_cm) # (B,npoint,k,D)

        corr = tf.matmul(ci, tf.transpose(cn, [0,1,3,2]))
        #corr = corr * I # remove auto-correlations
        corr = tf.math.exp(corr) # (B,npoint,k,k)

        S = tf.reduce_sum(corr, axis=-1, keepdims=True) # (B,npoint,k,1)

        cm = tf.tile(cm[:,:,None,:,:], [1,1,k,1,1]) # (B,npoint,k,k,D)
        g = cm * corr[...,None] # (B,npoint,k,k,D)
        g = tf.reduce_sum(g, axis=3) # (B,npoint,k,D)
        g = g / S

        weight_o_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        weights_o = tf.get_variable('weights_o', [D,D], initializer=weight_o_init,
                                    dtype=tf.float32, trainable=True)
        o = tf.matmul(g, weights_o) 
        o += new_points # (B,npoint,k,D)

        new_points = tf.reduce_max(o, axis=2) # (B,npoint,D)

        return new_xyz, new_points


def triplenet_sa_module(xyz, points, npoint, mlp, is_training, bn_decay, scope, knn=3, only_combinations=False, group_all=False, bn=True):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            mlp: list of int32 -- output size for MLP on each point
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NHWC'
    with tf.variable_scope(scope) as sc:
        # sampling and grouping
        #if group_all:
        #    new_xyz, new_points, grouped_xyz = sample_and_group_all_polygons(nsample=knn, xyz=xyz, points=points)
        #else:
        #    new_xyz, new_points, grouped_xyz = sample_and_group_polygons(npoint=npoint, nsample=knn, xyz=xyz, points=points)

        if group_all:
            new_xyz, new_points, grouped_xyz = sample_and_group_all_polygons(nsample=knn, xyz=xyz, points=points)
        else:
            new_xyz, new_points, grouped_xyz = sample_and_group_polygons(npoint=npoint, nsample=knn, xyz=xyz, points=points)

        # Generate all possible permutations of all possible triangles of each point and its k-nearest neighbors
        if only_combinations:
            point_idxs = np.arange(1,knn+1)
            point_idxs = np.array(list(itertools.combinations(point_idxs,3)))
            _ = list(map(np.random.shuffle, point_idxs))
            point_idxs = np.insert(point_idxs,0,0,axis=-1)
            point_idxs = np.ravel(point_idxs)
        else:
            point_idxs = np.arange(1,knn+1)
            point_idxs = np.array(list(itertools.permutations(point_idxs,2)))
            point_idxs = np.insert(point_idxs,0,0,axis=-1)
            point_idxs = np.ravel(point_idxs)

        # Extract the permuted points
        new_points_tr = tf.transpose(new_points, [2,0,1,3]) #bringing the points to the first axis
        new_points_tr = tf.gather(new_points_tr, point_idxs)
        new_points = tf.transpose(new_points_tr, [1,2,0,3])

        #new_points2 = tf.reshape(new_points,[new_points.get_shape()[0].value,new_points.get_shape()[1].value,-1,3*new_points.get_shape()[-1].value])
        #new_points2 = tf.reduce_max(new_points2, axis=[2])

        # process each triplet permutation individually
        new_points = tf_util.conv2d(new_points, mlp[0], [1,4],
                                        padding='VALID', stride=[1,4],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(0), bn_decay=bn_decay,
                                        data_format=data_format) 

        mlp.pop(0) #remove first element from the mlp list
        for i, num_out_channel in enumerate(mlp, start=1):
            # triplet permutations feature embedding
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 
        
        # max pooling over all the permutations
        new_points = tf.reduce_max(new_points, axis=[2], name='maxpool')
        #new_points = tf.concat([new_points,new_points2], axis=-1)

        if group_all:
            #new_points = tf.concat([new_points,points], axis=-1)
            new_points = tf.reduce_max(new_points, axis=[1], keepdims=True, name='global_maxpool') #(batch_size,1,mlp[-1])

        return new_xyz, new_points


def triplenet_onepoint_sa_module(xyz, points, npoint, mlp, is_training, bn_decay, scope, group_all=False ,bn=True, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, _, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, grouped_xyz = sample_and_group_polygons(npoint, 11, xyz, points)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        # Pooling in Local Regions
        new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])

        return new_xyz, new_points


def polynet_sa_module(xyz, points, npoint, mlp, is_training, bn_decay, scope, knn=3, only_combinations=False, group_all=False, bn=True):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            mlp: list of int32 -- output size for MLP on each point
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NHWC'
    with tf.variable_scope(scope) as sc:
        # sampling and grouping
        if group_all:
            new_xyz, new_points, grouped_xyz = sample_and_group_all_polygons(nsample=knn, xyz=xyz, points=points)
        else:
            new_xyz, new_points, grouped_xyz = sample_and_group_polygons(npoint=npoint, nsample=knn, xyz=xyz, points=points)

       # Generate all possible permutations of all possible triangles of each point and its k-nearest neighbors
        if only_combinations:
            point_idxs = np.arange(1,knn+1)
            _ = np.random.shuffle(point_idxs)
            point_idxs = np.insert(point_idxs,0,0)
        else:
            point_idxs = np.arange(1,knn+1)
            point_idxs = np.array(list(itertools.permutations(point_idxs)))
            point_idxs = np.insert(point_idxs,0,0,axis=-1)
            point_idxs = np.ravel(point_idxs)

        # Extract the permuted points
        new_points_tr = tf.transpose(new_points, [2,0,1,3]) #bringing the points to the first axis
        new_points_tr = tf.gather(new_points_tr, point_idxs)
        new_points = tf.transpose(new_points_tr, [1,2,0,3])

        # process each triplet permutation individually
        new_points = tf_util.conv2d(new_points, mlp[0], [1,knn+1],
                                        padding='VALID', stride=[1,knn+1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(0), bn_decay=bn_decay,
                                        data_format=data_format) 

        mlp.pop(0) #remove first element from the mlp list
        for i, num_out_channel in enumerate(mlp, start=1):
            # polygon permutations feature embedding
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 
        
        # max pooling over all the permutations
        new_points = tf.reduce_max(new_points, axis=[2], name='maxpool')

        if group_all:
            new_points = tf.reduce_max(new_points, axis=[1], keepdims=True, name='global_maxpool') #(batch_size,1,mlp[-1])

        return new_xyz, new_points

def pointnet_sa_module_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, is_training, bn_decay, scope, bn=True, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])
            if points is not None:
                grouped_points = group_point(points, idx)
                if use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0,3,1,2])
            for j,num_out_channel in enumerate(mlp_list[i]):
                grouped_points = tf_util.conv2d(grouped_points, num_out_channel, [1,1],
                                                padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                                scope='conv%d_%d'%(i,j), bn_decay=bn_decay)
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0,2,3,1])
            new_points = tf.reduce_max(grouped_points, axis=[2])
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)
        return new_xyz, new_points_concat


def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:                                                                                                      
            xyz1: (batch_size, ndataset1, 3) TF tensor                                                              
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1                                           
            points1: (batch_size, ndataset1, nchannel1) TF tensor                                                   
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point                                                 
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1


def triplenet_fp_module(xyz1, xyz2, points1, points2, mlp1, mlp2, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:                                                                                                      
            xyz1: (batch_size, ndataset1, 3) TF tensor                                                              
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1                                           
            points1: (batch_size, ndataset1, nchannel1) TF tensor                                                   
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point                                                 
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    batch_size = xyz1.get_shape()[0].value
    ndataset1 = xyz1.get_shape()[1].value
    with tf.variable_scope(scope) as sc:
        # retrieve three nearest neighbors 
        _, knn_idx = three_nn(xyz1, xyz2) #shape=(B,ndataset1,3)
        B_idx = np.arange(batch_size)
        B_idx = np.reshape(B_idx, (batch_size,1))
        B_idx = np.tile(B_idx, 3*ndataset1)
        B_idx = np.reshape(B_idx, (batch_size,ndataset1,3))
        knn_idx = tf.stack((B_idx,knn_idx),-1)
        knn_points = tf.gather_nd(points2, knn_idx) #shape=(B,ndataset1,3,nchannel2)

         # Generate all possible permutations of the 3 nearest neighbors
        knn_points_tr = tf.transpose(knn_points, [2,0,1,3]) #bringing the neighbors to the first axis
        point_idxs = np.array([0,1,2,0,2,1,1,0,2,1,2,0,2,0,1,2,1,0])
        knn_points_tr = tf.gather(knn_points_tr, point_idxs)
        knn_points = tf.transpose(knn_points_tr, [1,2,0,3])

        new_points1 = tf_util.conv2d(knn_points, mlp1[0], [1,3],
                                         padding='VALID', stride=[1,3],
                                         bn=bn, is_training=is_training,
                                         scope='interpol_%d'%(0), bn_decay=bn_decay)

        mlp1.pop(0)
        for i, num_out_channel in enumerate(mlp1, start=1):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='interpol_%d'%(i), bn_decay=bn_decay)
        
        new_points1 = tf.reduce_max(new_points1, axis=[2], name='maxpool')

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[new_points1, points1]) # B,ndataset1,nchannel1+nchannel2

        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp2):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)

        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1


def pointnet_perpoint_module(point_cloud, mlp, is_training,  bn_decay=None, bn=True, scope='perpoint_layer', use_nchw=False):
    ''' PointNet Per-Point Feature Extraction Module (PointNet branch without max-pooling)
        Input:
            point_cloud: (batch_size, ndataset, 6)
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_point: (batch_size, ndataset, 128)          
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        new_points = tf.expand_dims(point_cloud, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 
    
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        new_points = tf.squeeze(new_points, [2]) # B,ndataset,mlp[-1]

    return new_points




