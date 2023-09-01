# Copyright 2021 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""MusicVAE training script."""
import os

from magenta.models.music_vae import configs
from magenta.models.music_vae import data
import tensorflow.compat.v1 as tf
import tf_slim

import copy
import os
import re
import tarfile
import tempfile
import numpy as np

BASE_DIR = "gs://download.magenta.tensorflow.org/models/music_vae/colab2"

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'master', '',
    'The TensorFlow master to use.')
flags.DEFINE_string(
    'examples_path', None,
    'Path to a TFRecord file of NoteSequence examples. Overrides the config.')
flags.DEFINE_string(
    'tfds_name', None,
    'TensorFlow Datasets dataset name to use. Overrides the config.')
flags.DEFINE_string(
    'run_dir', None,
    'Path where checkpoints and summary events will be located during '
    'training and evaluation. Separate subdirectories `train` and `eval` '
    'will be created within this directory.')
flags.DEFINE_integer(
    'num_steps', 500, #200000
    'Number of training steps or `None` for infinite.')
flags.DEFINE_integer(
    'eval_num_batches', None,
    'Number of batches to use during evaluation or `None` for all batches '
    'in the data source.')
flags.DEFINE_integer(
    'checkpoints_to_keep', 0, #100
    'Maximum number of checkpoints to keep in `train` mode or 0 for infinite.')
flags.DEFINE_integer(
    'keep_checkpoint_every_n_hours', 1,
    'In addition to checkpoints_to_keep, keep a checkpoint every N hours.')
flags.DEFINE_string(
    'mode', 'train',
    'Which mode to use (`train` or `eval`).')
flags.DEFINE_string(
    'config', '',
    'The name of the config to use.')
flags.DEFINE_string(
    'hparams', '',
    'A comma-separated list of `name=value` hyperparameter values to merge '
    'with those in the config.')
flags.DEFINE_bool(
    'cache_dataset', True,
    'Whether to cache the dataset in memory for improved training speed. May '
    'cause memory errors for very large datasets.')
flags.DEFINE_integer(
    'task', 0,
    'The task number for this worker.')
flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter server tasks.')
flags.DEFINE_integer(
    'num_sync_workers', 0,
    'The number of synchronized workers.')
flags.DEFINE_string(
    'eval_dir_suffix', '',
    'Suffix to add to eval output directory.')
flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')
flags.DEFINE_bool(
    'finetune', False,
    'If True the will only train specific weights.')
flags.DEFINE_string(
    'trainable_vars', '',
    'A comma separated string of variable names for variables to be '
    'finetuned.')
flags.DEFINE_integer(
  'ckpt_no', 0,
  'If 0 use the official magenta ckpt otherwise the number of the ckpt'
  'in the train_dir.')
flags.DEFINE_string(
    'ckpt_path', '',
    'Path to a checkpoint to load the model from.')


def _parse_var_list(var_str):
  if var_str:
    var_list = ((var_str.lower()).strip(' ')).split(',')
    for i in range(len(var_list)):
      var_list[i] += ':0'
    return var_list
  else:
    return None


# Should not be called from within the graph to avoid redundant summaries.
def _trial_summary(hparams, examples_path, output_dir):
  """Writes a tensorboard text summary of the trial."""

  examples_path_summary = tf.summary.text(
      'examples_path', tf.constant(examples_path, name='examples_path'),
      collections=[])

  hparams_dict = hparams.values()

  # Create a markdown table from hparams.
  header = '| Key | Value |\n| :--- | :--- |\n'
  keys = sorted(hparams_dict.keys())
  lines = ['| %s | %s |' % (key, str(hparams_dict[key])) for key in keys]
  hparams_table = header + '\n'.join(lines) + '\n'

  hparam_summary = tf.summary.text(
      'hparams', tf.constant(hparams_table, name='hparams'), collections=[])

  with tf.Session() as sess:
    writer = tf.summary.FileWriter(output_dir, graph=sess.graph)
    writer.add_summary(examples_path_summary.eval())
    writer.add_summary(hparam_summary.eval())
    writer.close()


def _get_input_tensors(dataset, config):
  """Get input tensors from dataset."""
  batch_size = config.hparams.batch_size
  iterator = tf.data.make_one_shot_iterator(dataset)
  (input_sequence, output_sequence, control_sequence,
   sequence_length) = iterator.get_next()
  input_sequence.set_shape(
      [batch_size, None, config.data_converter.input_depth])
  output_sequence.set_shape(
      [batch_size, None, config.data_converter.output_depth])
  if not config.data_converter.control_depth:
    control_sequence = None
  else:
    control_sequence.set_shape(
        [batch_size, None, config.data_converter.control_depth])
  sequence_length.set_shape([batch_size] + sequence_length.shape[1:].as_list())

  return {
      'input_sequence': input_sequence,
      'output_sequence': output_sequence,
      'control_sequence': control_sequence,
      'sequence_length': sequence_length
  }


def train(run_dir,
          config,
          config_name,
          dataset_fn,
          checkpoints_to_keep=100,
          keep_checkpoint_every_n_hours=1,
          num_steps=None,
          master='',
          num_sync_workers=0,
          num_ps_tasks=0,
          task=0,
          finetune=False,
          trainable_vars='all',
          ckpt_no=-1,
          ckpt_path=None):
  """Train loop."""

  train_dir = os.path.join(run_dir, 'train')
  if finetune:
    if ckpt_path:
      checkpoint_path = ckpt_path
    elif ckpt_no >= 0:
      checkpoint_path = os.path.join(run_dir, 'train/model.ckpt-{}'.format(ckpt_no))
    else:
      checkpoint_path = BASE_DIR + '/checkpoints/' + config_name[4:] + '.ckpt'

  tf.gfile.MakeDirs(train_dir)
  is_chief = (task == 0)
  if is_chief:
    _trial_summary(
        config.hparams, config.train_examples_path or config.tfds_name,
        train_dir)
  with tf.Graph().as_default():
    with tf.device(tf.train.replica_device_setter(
        num_ps_tasks, merge_devices=True)):

      model = config.model
      model.build(config.hparams,
                  config.data_converter.output_depth,
                  is_training=True)

      optimizer = model.train(**_get_input_tensors(dataset_fn(), config))

      hooks = []
      if num_sync_workers:
        optimizer = tf.train.SyncReplicasOptimizer(
            optimizer,
            num_sync_workers)
        hooks.append(optimizer.make_session_run_hook(is_chief))

      grads, var_list = list(zip(*optimizer.compute_gradients(model.loss)))

      global_norm = tf.global_norm(grads)
      tf.summary.scalar('global_norm', global_norm)

      if config.hparams.clip_mode == 'value':
        g = config.hparams.grad_clip
        clipped_grads = [tf.clip_by_value(grad, -g, g) for grad in grads]
      elif config.hparams.clip_mode == 'global_norm':
        clipped_grads = tf.cond(
            global_norm < config.hparams.grad_norm_clip_to_zero,
            lambda: tf.clip_by_global_norm(  # pylint:disable=g-long-lambda
                grads, config.hparams.grad_clip, use_norm=global_norm)[0],
            lambda: [tf.zeros(tf.shape(g)) for g in grads])
      else:
        raise ValueError(
            'Unknown clip_mode: {}'.format(config.hparams.clip_mode))

      if finetune:
        # Create init_fn to initialize the Model from checkpoint
        # VERY USEFUL function to list all model variables: 
        # tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
        variables_to_restore = [v for v in 
        tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES) if v.name!='global_step:0']
        from tf_slim.ops.variables import assign_from_checkpoint_fn
        init_fn = assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)

        if trainable_vars !='all':
          print("=================================================")
          print("==================Finetune INFO==================")
          print("Training weights are...")
          for v in trainable_vars: print(v)
          print("================================================")

      # Training (not frozen) weights and their clipped grad is listed
      clippedgrad_var_pair_list = list()
      for i in range(len(var_list)):
        if finetune and trainable_vars!='all':
          if var_list[i].name in trainable_vars:
            clippedgrad_var_pair_list.append([clipped_grads[i], var_list[i]])
        else:
          clippedgrad_var_pair_list.append([clipped_grads[i], var_list[i]])

      
      train_op = optimizer.apply_gradients(
          clippedgrad_var_pair_list,
          global_step=model.global_step,
          name='train_step')

      logging_dict = {'global_step': model.global_step,
                      'loss': model.loss}

      hooks.append(tf.train.LoggingTensorHook(logging_dict, every_n_iter=100))
      if num_steps:
        hooks.append(tf.train.StopAtStepHook(last_step=num_steps))

      scaffold = tf.train.Scaffold(
        # init_fn=init_fn, # This doesn't work because of a lambda in tf source
        saver=tf.train.Saver(
          max_to_keep=checkpoints_to_keep,
          keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours))

      hooks.append(tf.estimator.CheckpointSaverHook(
        checkpoint_dir=train_dir,
        save_steps=100,
        checkpoint_basename='model.ckpt',
        scaffold=scaffold))

      if finetune:
        scaffold._init_fn = init_fn # The hack to add init_fn

      tf_slim.training.train(
          train_op=train_op,
          logdir=train_dir,
          scaffold=scaffold,
          hooks=hooks,
          save_checkpoint_secs=None, #60,
          master=master,
          is_chief=is_chief)    


def evaluate(run_dir,
             eval_dir,
             config,
             config_name,
             dataset_fn,
             num_batches,
             ckpt_no=-1,
             ckpt_path=None,
             master=''):

  """Evaluate the model Once."""
  tf.gfile.MakeDirs(eval_dir)

  _trial_summary(
      config.hparams, config.eval_examples_path or config.tfds_name, eval_dir)
  with tf.Graph().as_default():
    model = config.model
    model.build(config.hparams,
                config.data_converter.output_depth,
                is_training=False)

    dataset = dataset_fn().take(num_batches)

    eval_op = model.eval(**_get_input_tensors(dataset, config))

    hooks = [
        tf_slim.evaluation.StopAfterNEvalsHook(num_batches),
        tf_slim.evaluation.SummaryAtEndHook(eval_dir)
    ]

    if ckpt_path:
      checkpoint_path = ckpt_path
    elif ckpt_no >= 0:
      checkpoint_path = os.path.join(run_dir, 'train/model.ckpt-{}'.format(ckpt_no))
    else:
      checkpoint_path = BASE_DIR + '/checkpoints/' + config_name[4:] + '.ckpt'

    # Create init_fn to initialize the Model from checkpoint
    # VERY USEFUL function to list all model variables: 
    # tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
    # variables_to_restore = [v for v in 
    # tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES) if v.name!='global_step:0']
    # from tf_slim.ops.variables import assign_from_checkpoint_fn
    # init_fn = assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)

    tf_slim.evaluation.evaluate_once(
      logdir=eval_dir,
      master=master,
      checkpoint_path=checkpoint_path,
      eval_op=eval_op,
      hooks=hooks)


def run(config_map,
        tf_file_reader=tf.data.TFRecordDataset,
        file_reader=tf.python_io.tf_record_iterator):
  """Load model params, save config file and start trainer.

  Args:
    config_map: Dictionary mapping configuration name to Config object.
    tf_file_reader: The tf.data.Dataset class to use for reading files.
    file_reader: The Python reader to use for reading files.

  Raises:
    ValueError: if required flags are missing or invalid.
  """
  if not FLAGS.run_dir:
    raise ValueError('Invalid run directory: %s' % FLAGS.run_dir)
  run_dir = os.path.expanduser(FLAGS.run_dir)

  if FLAGS.mode not in ['train', 'eval']:
    raise ValueError('Invalid mode: %s' % FLAGS.mode)

  if FLAGS.config not in config_map:
    raise ValueError('Invalid config: %s' % FLAGS.config)
  config = config_map[FLAGS.config]
  if FLAGS.hparams:
    config.hparams.parse(FLAGS.hparams)
  config_update_map = {}
  if FLAGS.examples_path:
    config_update_map['%s_examples_path' % FLAGS.mode] = os.path.expanduser(
        FLAGS.examples_path)
  if FLAGS.tfds_name:
    if FLAGS.examples_path:
      raise ValueError(
          'At most one of --examples_path and --tfds_name can be set.')
    config_update_map['tfds_name'] = FLAGS.tfds_name
    config_update_map['eval_examples_path'] = None
    config_update_map['train_examples_path'] = None
  config = configs.update_config(config, config_update_map)
  if FLAGS.num_sync_workers:
    config.hparams.batch_size //= FLAGS.num_sync_workers

  if FLAGS.mode == 'train':
    is_training = True
  elif FLAGS.mode == 'eval':
    is_training = False
  else:
    raise ValueError('Invalid mode: {}'.format(FLAGS.mode))

  if FLAGS.finetune and not is_training:
    raise ValueError('Invalid value. Cannot finetune in eval mode: {}'.format(FLAGS.mode))

  if FLAGS.finetune:
    if FLAGS.trainable_vars.lower() == 'all':
      trainable_vars = 'all'
    elif FLAGS.trainable_vars.lower() == 'last_layer':
      trainable_vars = ['decoder/output_projection/kernel:0', 'decoder/output_projection/bias:0']
    else:
      trainable_vars = _parse_var_list(FLAGS.trainable_vars)
      if not trainable_vars:
        raise ValueError('Invalid value. Trainable variables cannot be empty.')
  else:
    trainable_vars = None

  def dataset_fn():
    return data.get_dataset(
        config,
        tf_file_reader=tf_file_reader,
        is_training=is_training,
        cache_dataset=FLAGS.cache_dataset)

  if is_training:
    train(
        run_dir=run_dir,
        config=config,
        config_name=FLAGS.config,
        dataset_fn=dataset_fn,
        checkpoints_to_keep=FLAGS.checkpoints_to_keep,
        keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
        num_steps=FLAGS.num_steps,
        master=FLAGS.master,
        num_sync_workers=FLAGS.num_sync_workers,
        num_ps_tasks=FLAGS.num_ps_tasks,
        task=FLAGS.task,
        finetune = FLAGS.finetune,
        trainable_vars = trainable_vars,
        ckpt_path=FLAGS.ckpt_path
        )
  else:
    num_batches = FLAGS.eval_num_batches or data.count_examples(
        config.eval_examples_path,
        config.tfds_name,
        config.data_converter,
        file_reader) // config.hparams.batch_size
    print("=======================================")
    print("Batch size = ", config.hparams.batch_size)
    print("num_batches = ", num_batches)
    print("=======================================")
    eval_dir = os.path.join(run_dir, 'eval' + FLAGS.eval_dir_suffix)
    evaluate(
        run_dir,
        eval_dir,
        config=config,
        config_name=FLAGS.config,
        dataset_fn=dataset_fn,
        num_batches=num_batches,
        ckpt_no=FLAGS.ckpt_no,
        ckpt_path=FLAGS.ckpt_path,
        master=FLAGS.master)


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)
  run(configs.CONFIG_MAP)


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
