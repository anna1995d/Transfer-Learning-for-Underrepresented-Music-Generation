import abc

from magenta.contrib import training as contrib_training
from magenta.models.music_vae import lstm_models
from magenta.models.music_vae import base_model
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import tf_slim

ds = tfp.distributions
from tensor2tensor.utils.hparam import HParams

class StudentTeacher(object):
  """Student-teacher version of MusicVAE"""

  def __init__(self): 
    self._teacher = list()
    self._student = base_model.MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                                        lstm_models.CategoricalLstmDecoder(),
                                        name_or_scope = 'student')

  def build(self, hparams, output_depth, is_training):
    """Builds the models.
    Must be called within a graph.
    Args:
      hparams: An HParams object containing model hyperparameters. See
          `get_default_hparams` below for required values.
      output_depth: Size of final output dimension.
      is_training: Whether or not the model will be used for training.
    """
    tf.logging.info("This is version 101 build")
    # tf.logging.info('aBuilding student model with %s, teacher model with %s, and hparams:\n%s',
    #                 self._student.__class__.__name__,
    #                 self._teacher.__class__.__name__, hparams.values())
    self.global_step = tf.train.get_or_create_global_step()
    self._hparams = hparams

    self._student.build(hparams,
                        output_depth,
                        is_training=True) 
  
  @property
  def student(self):
    return self._student

  @property
  def teacher(self):
    return self._teacher

  @property
  def hparams(self):
    return self._hparams

  def _compute_model_loss(
      self, input_sequence, output_sequence, sequence_length, control_sequence):
    """Builds a model with loss for train/eval."""

    # returns the metric map
    student_loss = self._student._compute_model_loss(input_sequence,
                                                     output_sequence,
                                                     sequence_length,
                                                     control_sequence)[1]
    teacher_loss = self._teacher[]
    

    alpha = self._hparams.alpha

    self.loss = student_loss['loss'] * alpha + teacher_loss['loss'] * (1 - alpha)

    # scalars_to_summarize = {
    #     'loss': self.loss,
    #     'losses/r_loss': r_loss,
    #     'losses/kl_loss': kl_cost,
    #     'losses/kl_bits': kl_div / tf.math.log(2.0),
    #     'losses/kl_beta': beta,
    # }
    scalars_to_summarize = {
        'loss': self.loss
    }

    return scalars_to_summarize
    
  def train(self, input_sequence, output_sequence, sequence_length,
            control_sequence=None):
    """Train on the given sequences, returning an optimizer.
    Args:
      input_sequence: The sequence to be fed to the encoder.
      output_sequence: The sequence expected from the decoder.
      sequence_length: The length of the given sequences (which must be
          identical).
      control_sequence: (Optional) sequence on which to condition. This will be
          concatenated depthwise to the model inputs for both encoding and
          decoding.
    Returns:
      optimizer: A tf.train.Optimizer.
    """

    scalars_to_summarize = self._compute_model_loss(
        input_sequence, output_sequence, sequence_length, control_sequence)

    hparams = self.hparams
    lr = ((hparams.learning_rate - hparams.min_learning_rate) *
          tf.pow(hparams.decay_rate, tf.to_float(self.global_step)) +
          hparams.min_learning_rate)

    optimizer = tf.train.AdamOptimizer(lr)

    tf.summary.scalar('learning_rate', lr)
    for n, t in scalars_to_summarize.items():
      tf.summary.scalar(n, tf.reduce_mean(t))

    return optimizer

  def eval(self, input_sequence, output_sequence, sequence_length,
           control_sequence=None):
    """Evaluate on the given sequences, returning metric update ops.

    Args:
      input_sequence: The sequence to be fed to the encoder.
      output_sequence: The sequence expected from the decoder.
      sequence_length: The length of the given sequences (which must be
        identical).
      control_sequence: (Optional) sequence on which to condition the decoder.

    Returns:
      metric_update_ops: tf.metrics update ops.
    """
    return self._student.eval(input_sequence, output_sequence, sequence_length, control_sequence)

  def sample(self, n, max_length=None, z=None, c_input=None, **kwargs):
    """Sample with an optional conditional embedding `z`."""
    return self._student._decoder.sample(n, max_length, z, c_input, **kwargs)
  