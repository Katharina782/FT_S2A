import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import functools

import tensorflow as tf
import tensorflow_hub as hub
#import joblib
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import pandas as pd
import numpy as np
import tqdm as tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import *
#import pybedtools
#import matplotlib.pyplot as plt
#import matplotlib as mpl
#import seaborn as sns# Code snippet from github deepmind repo



# @title `Enformer`, `EnformerScoreVariantsNormalized`, `EnformerScoreVariantsPCANormalized`,

class Enformer:

  def __init__(self, tfhub_url, device):
    with tf.device(device):
      self._model = hub.load(tfhub_url).model

  def predict_on_batch(self, inputs):
    predictions = self._model.predict_on_batch(inputs)
    return {k: v.numpy() for k, v in predictions.items()}

  @tf.function
  def contribution_input_grad(self, input_sequence,
                              target_mask, output_head='human'):
    input_sequence = input_sequence[tf.newaxis]

    target_mask_mass = tf.reduce_sum(target_mask)
    with tf.GradientTape() as tape:
      tape.watch(input_sequence)
      prediction = tf.reduce_sum(
          target_mask[tf.newaxis] *
          self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

    input_grad = tape.gradient(prediction, input_sequence) * input_sequence
    input_grad = tf.squeeze(input_grad, axis=0)
    return tf.reduce_sum(input_grad, axis=-1)



def get_organism_path(organism):
   return os.path.join('/data/mikulik/mnt/gcs_basenj/', organism)

def get_metadata(organism):
    # Keys:
    # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
    # pool_width, crop_bp, target_length
    path = os.path.join(get_organism_path(organism), 'statistics.json')
    with tf.io.gfile.GFile(path, 'r') as f:
        return json.load(f) 

def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


def tfrecord_files(organism, subset):
    # Sort the values by int(*).
    return sorted(tf.io.gfile.glob(os.path.join(
        get_organism_path(organism), 'tfrecords', f'{subset}-*.tfr'
    )), key=lambda x: int(x.split('-')[-1].split('.')[0]))



def deserialize(serialized_example, metadata):
    """Deserialize bytes stored in TFRecordFile."""
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
    sequence = tf.cast(sequence, tf.float32)

    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target,
                        (metadata['target_length'], metadata['num_targets']))
    target = tf.cast(target, tf.float32)

    return {'sequence': sequence,
            'target': target}

def get_dataset(organism, subset, num_threads=8):
    metadata = get_metadata(organism)
    dataset = tf.data.TFRecordDataset(tfrecord_files(organism, subset),
                                        compression_type='ZLIB',
                                        num_parallel_reads=num_threads)
    dataset = dataset.map(functools.partial(deserialize, metadata=metadata),
                            num_parallel_calls=num_threads)
    return dataset


def get_targets(data_dir, organism):
    targets_txt = os.path.join(data_dir, organism, "targets.txt")
    return pd.read_csv(targets_txt, sep='\t')


def _reduced_shape(shape, axis):
  if axis is None:
    return tf.TensorShape([])
  return tf.TensorShape([d for i, d in enumerate(shape) if i not in axis])


class CorrelationStats(tf.keras.metrics.Metric):
  """Contains shared code for PearsonR and R2."""

  def __init__(self, reduce_axis=None, name='pearsonr'):
    """Pearson correlation coefficient.

    Args:
      reduce_axis: Specifies over which axis to compute the correlation (say
        (0, 1). If not specified, it will compute the correlation across the
        whole tensor.
      name: Metric name.
    """
    super(CorrelationStats, self).__init__(name=name)
    self._reduce_axis = reduce_axis
    self._shape = None  # Specified in _initialize.

  def _initialize(self, input_shape):
    # Remaining dimensions after reducing over self._reduce_axis.
    self._shape = _reduced_shape(input_shape, self._reduce_axis)

    weight_kwargs = dict(shape=self._shape, initializer='zeros')
    self._count = self.add_weight(name='count', **weight_kwargs)
    self._product_sum = self.add_weight(name='product_sum', **weight_kwargs)
    self._true_sum = self.add_weight(name='true_sum', **weight_kwargs)
    self._true_squared_sum = self.add_weight(name='true_squared_sum',
                                             **weight_kwargs)
    self._pred_sum = self.add_weight(name='pred_sum', **weight_kwargs)
    self._pred_squared_sum = self.add_weight(name='pred_squared_sum',
                                             **weight_kwargs)

  def update_state(self, y_true, y_pred):#, sample_weight=None):
    """Update the metric state.

    Args:
      y_true: Multi-dimensional float tensor [batch, ...] containing the ground
        truth values.
      y_pred: float tensor with the same shape as y_true containing predicted
        values.
      sample_weight: 1D tensor aligned with y_true batch dimension specifying
        the weight of individual observations.
    """
    if self._shape is None:
      # Explicit initialization check.
      self._initialize(y_true.shape)
    y_true.shape.assert_is_compatible_with(y_pred.shape)
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')

    self._product_sum.assign_add(
        tf.reduce_sum(y_true * y_pred, axis=self._reduce_axis))

    self._true_sum.assign_add(
        tf.reduce_sum(y_true, axis=self._reduce_axis))

    self._true_squared_sum.assign_add(
        tf.reduce_sum(tf.math.square(y_true), axis=self._reduce_axis))

    self._pred_sum.assign_add(
        tf.reduce_sum(y_pred, axis=self._reduce_axis))

    self._pred_squared_sum.assign_add(
        tf.reduce_sum(tf.math.square(y_pred), axis=self._reduce_axis))

    self._count.assign_add(
        tf.reduce_sum(tf.ones_like(y_true), axis=self._reduce_axis))

  def result(self):
    raise NotImplementedError('Must be implemented in subclasses.')

  def reset_states(self):
    if self._shape is not None:
      tf.keras.backend.batch_set_value([(v, np.zeros(self._shape))
                                        for v in self.variables])


class PearsonR(CorrelationStats):
  """Pearson correlation coefficient.

  Computed as:
  ((x - x_avg) * (y - y_avg) / sqrt(Var[x] * Var[y])
  """

  def __init__(self, reduce_axis=(0,), name='pearsonr'):
    """Pearson correlation coefficient.

    Args:
      reduce_axis: Specifies over which axis to compute the correlation.
      name: Metric name.
    """
    super(PearsonR, self).__init__(reduce_axis=reduce_axis,
                                   name=name)

  def result(self):
    true_mean = self._true_sum / self._count
    pred_mean = self._pred_sum / self._count

    covariance = (self._product_sum
                  - true_mean * self._pred_sum
                  - pred_mean * self._true_sum
                  + self._count * true_mean * pred_mean)

    true_var = self._true_squared_sum - self._count * tf.math.square(true_mean)
    pred_var = self._pred_squared_sum - self._count * tf.math.square(pred_mean)
    tp_var = tf.math.sqrt(true_var) * tf.math.sqrt(pred_var)
    correlation = covariance / tp_var

    return correlation


class R2(CorrelationStats):
  """R-squared  (fraction of explained variance)."""

  def __init__(self, reduce_axis=None, name='R2'):
    """R-squared metric.

    Args:
      reduce_axis: Specifies over which axis to compute the correlation.
      name: Metric name.
    """
    super(R2, self).__init__(reduce_axis=reduce_axis,
                             name=name)

  def result(self):
    true_mean = self._true_sum / self._count
    total = self._true_squared_sum - self._count * tf.math.square(true_mean)
    residuals = (self._pred_squared_sum - 2 * self._product_sum
                 + self._true_squared_sum)

    return tf.ones_like(residuals) - residuals / total


class MetricDict:
  def __init__(self, metrics):
    self._metrics = metrics

  def update_state(self, y_true, y_pred):
    for k, metric in self._metrics.items():
      metric.update_state(y_true, y_pred)

  def result(self):
    return {k: metric.result() for k, metric in self._metrics.items()}




  # This is inspired by the lucidrains implementation in order to be able to specify the length of the input sequence
class DeepMindData:
  def __init__(self, organism:str, subset:str, seq_len:int, fasta_path:str, n_to_test:int = -1):
      assert subset in {"train", "valid", "test"}
      assert organism in {"human", "mouse"}
      self.organism = organism
      self.subset = subset
      self.base_dir = get_organism_path(organism)
      self.seq_len = seq_len
      self.fasta_reader = FastaStringExtractor(fasta_path)
      self.n_to_test = n_to_test
      with tf.io.gfile.GFile(f"{self.base_dir}/sequences.bed", 'r') as f:
        region_df = pd.read_csv(f, sep="\t", header=None)
        region_df.columns = ['chrom', 'start', 'end', 'subset']
        self.region_df = region_df.query('subset==@subset').reset_index(drop=True)
      
  def __iter__(self):
    #worker_info = torch.utils.data.get_worker_info()
    #assert worker_info is None, "Only support single process loading"
    # If num_threads > 1, the following will actually shuffle the inputs! luckily we catch this with the sequence comparison
    basenji_iterator = get_dataset(self.organism, self.subset, num_threads=1).as_numpy_iterator()
    for i, records in enumerate(basenji_iterator):
      loc_row = self.region_df.iloc[i]
      target_interval = Interval(loc_row['chrom'], loc_row['start'], loc_row['end'])
      sequence_one_hot = one_hot_encode(self.fasta_reader.extract(target_interval.resize(self.seq_len)))
      if self.n_to_test >= 0 and i < self.n_to_test:
        old_sequence_onehot = records["sequence_old"]
        if old_sequence_onehot.shape[0] > sequence_one_hot.shape[0]:
          diff = old_sequence_onehot.shape[0] - sequence_one_hot.shape[0]
          trim = diff//2
          np.testing.assert_equal(old_sequence_onehot[trim:(-trim)], sequence_one_hot)
        elif sequence_one_hot.shape[0] > old_sequence_onehot.shape[0]:
          diff = sequence_one_hot.shape[0] - old_sequence_onehot.shape[0]
          trim = diff//2
          np.testing.assert_equal(old_sequence_onehot, sequence_one_hot[trim:(-trim)])
        else:
          np.testing.assert_equal(old_sequence_onehot, sequence_one_hot)
      yield {
          "sequence": tf.convert_to_tensor(sequence_one_hot),
          "target": tf.convert_to_tensor(records["target"]),
      }
     