import gzip
import json
import math
import os
import pathlib
import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tqdm import tqdm


def _default_fs_mapper(file):
    if isinstance(file, dict):
        return file
    if not isinstance(file, str):
        return file.__dict__
    if str(file).endswith('.gz'):
        with gzip.open(file, 'rt', encoding='utf-8') as zipfile:
            return json.load(zipfile)
    else:
        with open(file, 'r', encoding='utf-8') as rf:
            return json.load(rf)


def _default_obj_mapper(obj: dict, _):
    return obj


def _feature(values, obj_cls):
    if not isinstance(values, list):
        values = [values]
    if obj_cls is int:
        return tf.train.Feature(int64_list=tf.train.Int64List(
            value=[int(v) for v in values]))
    elif obj_cls is float:
        return tf.train.Feature(float_list=tf.train.FloatList(
            value=[float(v) for v in values]))
    elif obj_cls is str:
        return tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[str(v).encode() for v in values]))
    else:
        print('TFR ERROR: unknown value type for {}'.format(obj_cls))
        return None


def _bytes_feature(values):
    if not isinstance(values, list):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _convert_to_tfr(obj, meta):
    ctx = {}
    seq = {}
    types, shapes = meta['types'], meta['shapes']
    for k in obj:
        val = obj[k] if isinstance(obj[k], list) else [obj[k]]
        if len(shapes[k]) < 2:
            ctx[k] = _feature(val, types[k])
        elif len(shapes[k]) == 2:
            seq[k] = tf.train.FeatureList(
                feature=[_feature(ls, types[k]) for ls in val])
        elif len(shapes[k]) == 3:
            # convert to bytes first:
            dtype = np.float32 if types[k] is float else np.int32
            seq[k] = [_bytes_feature(
                [np.array(ls1).astype(dtype).tobytes()
                 for ls1 in ls0]) for ls0 in val]
            seq[k] = tf.train.FeatureList(feature=seq[k])
        else:
            print('VFR ERROR: unsupported shape', shapes[k], 'for key', k)
    return tf.train.SequenceExample(
        feature_lists=tf.train.FeatureLists(feature_list=seq),
        context=tf.train.Features(feature=ctx)
    )


def _write_tfr_trunk(
        trunk,
        obj_mapper,
        meta,
        file):
    index, js_files = trunk
    with tf.io.TFRecordWriter(file + '-{}.tfr'.format(index)) as writer:
        for f in js_files:
            rd = _convert_to_tfr(obj_mapper(_default_fs_mapper(f), f), meta)
            writer.write(rd.SerializeToString())


def write_objs_as_tfr(
        js_files,
        output_path,
        obj_mapper=_default_obj_mapper,
        trunk_size=50,
        num_workers=30,
        num_samples_to_infer_shape=100,
        additional_meta=None):
    print('writing ', len(js_files), ' tfr to', output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cache = []
    for s in tqdm(js_files):
        if len(cache) >= num_samples_to_infer_shape:
            break
        obj = obj_mapper(_default_fs_mapper(s), s)
        if not obj:
            continue
        cache.append(obj)
    meta = _infer_shape(cache)
    for k in sorted(meta['shapes']):
        print('INFO: {}\'s inferred shape is {} '
              'and type is {}'.format(
            k, meta['shapes'][k], meta['types'][k]))
    num_trunk = round(len(js_files) / trunk_size + 0.5)

    trunks = [(i, js_files[i * trunk_size:(i + 1) * trunk_size]) for i in range(num_trunk)]

    with ProcessPoolExecutor(max_workers=num_workers) as e:
        for _ in tqdm(
                e.map(
                    partial(_write_tfr_trunk, obj_mapper=obj_mapper, meta=meta,
                            file=os.path.join(output_path, 'data')),
                    trunks
                ),
                total=len(trunks)
        ):
            pass
    meta['total_len'] = len(js_files)
    if additional_meta:
        for k, v in additional_meta.items():
            meta[k] = v

    with open(os.path.join(output_path, 'meta.pk'), 'wb') as f:
        pickle.dump(meta, f)

    return read_tfr_to_ds(output_path)


def _parse_tfr(tfr, meta, attr_exclude=None):
    def get_feature_def(clz, shape):
        if len(shape) == 3:
            return tf.io.VarLenFeature(tf.string)
        elif len(shape) < 3 and clz is str:
            return tf.io.VarLenFeature(tf.string)
        elif len(shape) < 3 and clz is int:
            return tf.io.VarLenFeature(tf.int64)
        elif len(shape) < 3 and clz is float:
            return tf.io.VarLenFeature(tf.float32)
        else:
            print("TFR ERROR: Unsupported configuration ", clz, shape)
        return None

    shapes = meta['shapes']
    types = meta['types']
    ctx_def = {k: get_feature_def(types[k], shapes[k]) for k in shapes if
               len(shapes[k]) < 2}
    seq_def = {k: get_feature_def(types[k], shapes[k]) for k in shapes if
               len(shapes[k]) >= 2}

    def map_fn(k, val):
        # if val.dtype == tf.string and types[k] in (int, float):
        #     np_dtype = np.float32 if types[k] is float else np.int32
        #     tf_dtype = tf.float32 if types[k] is float else tf.int32
        #     val0 = tf.RaggedTensor.from_sparse(val)
        #     val = tf.ragged.map_flat_values(
        #         tf.io.decode_raw, val0, tf_dtype)
        # elif types[k] is str:
        #     val = tf.sparse.to_dense(
        #         val, default_value=b'')
        # else:
        #     val = tf.sparse.to_dense(val)
        #     actual_shape = [shapes[k][i] or tf.shape(val)[i] for i in
        #                     range(len(shapes[k]))]
        #     val = tf.reshape(val, actual_shape)
        # if int64_to_int32 and val.dtype is tf.int64:
        #     val = tf.cast(val, tf.int32)
        if len(val.shape) > 1:
            val = tf.RaggedTensor.from_sparse(val)
            # if len(shapes[k]) > 2:
            #     tf_dtype = tf.float32 if types[k] is float else tf.int32
            #     val = tf.io.decode_raw(val.values, tf_dtype, fixed_length=shapes[-1])

        # actual_shape = [shapes[k][i] or tf.shape(val)[i] for i in
        #                 range(len(shapes[k]))]
        # val = tf.reshape(val, actual_shape)
        else:
            if shapes[k][0] is not None:
                val = tf.reshape(tf.sparse.to_dense(val), [shapes[k][0]])
        if len(shapes[k]) > 2:
            tf_dtype = tf.float32 if types[k] is float else tf.int32
            val = tf.ragged.map_flat_values(
                tf.io.decode_raw, val, tf_dtype)
        return val

    ctx_p, seq_p = tf.io.parse_single_sequence_example(
        serialized=tfr,
        sequence_features=seq_def,
        context_features=ctx_def
    )
    merged = {**ctx_p, **seq_p}
    if attr_exclude is not None:
        for k_id in attr_exclude:
            if k_id in merged:
                del merged[k_id]
    merged = {k: map_fn(k, merged[k]) for k in merged}
    return merged


def read_tfr_to_ds(file_path, parallelism=16,
                   attr_exclude=None, meta=None):
    # read tfr
    files = [str(f) for f in pathlib.Path(file_path).glob('**/*.tfr')]
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=parallelism)
    if meta is None:
        with open(os.path.join(file_path, 'meta.pk'), 'rb') as f:
            meta = pickle.load(f)

    ds = ds.map(lambda x: _parse_tfr(x, meta, attr_exclude),
                num_parallel_calls=parallelism)
    ds.meta = meta
    return ds


def _infer_shape(objs_to_infer, show_progress=False):
    objs_to_infer = tqdm(objs_to_infer) if show_progress else objs_to_infer
    keys = {k for obj in objs_to_infer for k in obj}
    shapes = {k: [] for k in keys}
    types = {}
    for k in keys:
        if show_progress:
            print('processing key {}'.format(k))
        vals = [obj[k] if isinstance(obj[k], list) else [obj[k]] for obj in
                objs_to_infer]
        while isinstance(vals[0], list):
            shapes[k].append(-1)
            for ls in vals:
                if len(ls) < 1:
                    continue
                if shapes[k][-1] == -1:
                    shapes[k][-1] = len(ls)
                elif shapes[k][-1] != len(ls):
                    shapes[k][-1] = None
            vals = [ele for val in vals for ele in val]
        types[k] = type(vals[0])
    return {'types': types, 'shapes': shapes, 'total_len': 0}


def make_batch(ds, batch_size, map_parallelism=8, cache=False, pad=False, prefetch=-1):
    """
    Batch the tf dataset and prune the tensors
    :param ds: the tf dataset
    :param batch_size: the batch size
    :param map_parallelism: number of map parallelism threads
    :param cache: whether to cache the tf dataset
    :param pad: whether to pad the ragged tensor
    :param prefetch: the number of elements to be pre-fetched in tf dataset
    :return: the batched tf dataset
    """
    assert batch_size > 1, 'Batch size should be greater than 1'

    def _map_to_tensor(x):
        if isinstance(x, tf.SparseTensor):
            x = tf.RaggedTensor.from_sparse(x)
        if pad and isinstance(x, tf.RaggedTensor):
            x = x.to_tensor()
        return x

    ds = ds.batch(batch_size)
    ds = ds.map(
        lambda v: nest.map_structure(_map_to_tensor, v),
        num_parallel_calls=map_parallelism
    )
    if cache:
        ds = ds.cache()
    elif prefetch > 1:
        ds = ds.prefetch(prefetch)
    return ds


def read_tfr(file_path: str, parallelism: int = 16) -> tf.data.Dataset:
    """
    Load tf record into tf dataset
    :param file_path: the path to the records
    :param parallelism: number of parallelism
    :return: the tf dataset
    """
    # read tfr
    files = [str(f) for f in pathlib.Path(file_path).glob('**/*.tfr')]
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=parallelism)
    # load meta
    with open(os.path.join(file_path, 'meta.pk'), 'rb') as f:
        meta = pickle.load(f)
    # parse dataset
    ds = ds.map(lambda x: _parse_tfr(x, meta), num_parallel_calls=parallelism)
    ds.meta = meta
    return ds


def write_tfr(pairs: List[dict], output_path: str, trunk_size: int = 50) -> tf.data.Dataset:
    """
    Write data into tf reocrd in trunks
    :param pairs: data pairs
    :param output_path: the output path
    :param trunk_size: size of the trunk
    :return: the tf dataset
    """
    total_pairs = len(pairs)
    print('Writing ', total_pairs, 'tf records to', output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    meta = _infer_shape(pairs[:100])
    meta['total_len'] = total_pairs

    # write each trunk to tf record
    total_trunks = math.ceil(total_pairs / trunk_size)
    file_name = os.path.join(output_path, 'data')
    for i in tqdm(range(total_trunks), desc='Writing tfrs'):
        trunk = pairs[i * trunk_size: (i + 1) * trunk_size]
        with tf.io.TFRecordWriter(file_name + '-{}.tfr'.format(i)) as writer:
            for pair in trunk:
                sample = _convert_to_tfr(pair, meta)
                writer.write(sample.SerializeToString())

    with open(os.path.join(output_path, 'meta.pk'), 'wb') as f:
        pickle.dump(meta, f)

    return read_tfr_to_ds(output_path)
