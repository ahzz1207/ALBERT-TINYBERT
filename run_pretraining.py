# coding=utf-8
# Copyright 2019 The Googpple Research Authors.
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

# Lint as: python2, python3
"""Run masked LM/next sentence masked_lm pre-training for ALBERT."""

from __future__ import absolute_import, division, print_function

import functools
import json
import os
import time
import copy

import tensorflow as tf
from absl import app, flags, logging
from six.moves import range

import albert_model
import tinybert_model
import input_pipeline
from albert import AlbertConfig, AlbertModel
from tinybert import TinybertConfig, TinybertModel
from model_training_utils import run_customized_training_loop
from optimization import LAMB, AdamWeightDecay, WarmUp
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
FLAGS = flags.FLAGS 

## Required parameters
flags.DEFINE_string(
    "albert_config_file", "/work/ALBERT-TF2.0-master/model_configs/base/config.json",
    "The config json file corresponding to the pre-trained ALBERT model. "
    "This specifies the model architecture.")

flags.DEFINE_bool(
    "finetune_tinybert", True, "decided witch loss will u choose to train"
)

flags.DEFINE_bool(
    "start_with_train_model", False, "model restore from ckpt of trained_model ,else restore from every .h5 file")

flags.DEFINE_string(
    "tinybert_config_file", "/work/ALBERT-TF2.0-master/model_configs/base/config_tiny.json",
    "The config json file corresponding to the pre-trained TINYBERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_files", "/work/pretrain_data/*.tfrecord",
    "Input TF example files (can be a glob or comma separated).")
  
flags.DEFINE_string("meta_data_file_path", None,
                    "The path in which input meta data will be written.")

flags.DEFINE_string(
    "output_dir", "/work/ALBERT-TF2.0-master/model_out4/",
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained two model).")

flags.DEFINE_string(
    "init_checkpoint_albertbert", None,
    "Initial checkpoint (usually from a pre-trained ALBERT model).")

flags.DEFINE_string(
    "init_checkpoint_tinybert", None,
    "Initial checkpoint (usually from a pre-trained TINYBERT model).")


flags.DEFINE_integer(
    "max_seq_length", 256,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 25,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 128, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 16, "Total batch size for eval.")

flags.DEFINE_enum("optimizer", "adamw", ["adamw", "lamb"],
                  "The optimizer for training.")

flags.DEFINE_float("learning_rate", 4e-5, "The initial learning rate.")

flags.DEFINE_integer("steps_per_loop", 1000, "One loop run numbers steps")

flags.DEFINE_integer("steps_per_epoch", 100000, "how many steps to save model")

flags.DEFINE_integer("num_train_epochs", 3, "Number of training epochs.")

flags.DEFINE_float("warmup_proportion", 0.1, "Number of warmup steps.")

flags.DEFINE_float("weight_decay", 0.01, "weight_decay")

flags.DEFINE_float("adam_epsilon", 1e-6, "adam_epsilon")

flags.DEFINE_enum(
    "strategy_type", "mirror", ["one", "mirror"],
    "Training strategy for single or multi gpu training")


def get_pretrain_input_data(input_file_pattern, seq_length,
                            max_predictions_per_seq, batch_size, strategy):
  """Returns input dataset from input file string."""

  # When using TPU pods, we need to clone dataset across
  # workers and need to pass in function that returns the dataset rather
  # than passing dataset instance itself.
  use_dataset_fn = isinstance(strategy, tf.distribute.experimental.TPUStrategy)
  if use_dataset_fn:
    if batch_size % strategy.num_replicas_in_sync != 0:
      raise ValueError(
          'Batch size must be divisible by number of replicas : {}'.format(
              strategy.num_replicas_in_sync))

    # As auto rebatching is not supported in
    # `experimental_distribute_datasets_from_function()` API, which is
    # required when cloning dataset to multiple workers in eager mode,
    # we use per-replica batch size.
    batch_size = int(batch_size / strategy.num_replicas_in_sync)

  def _dataset_fn(ctx=None):
    """Returns tf.data.Dataset for distributed BERT pretraining."""
    input_patterns = input_file_pattern.split(',')
    train_dataset = input_pipeline.create_pretrain_dataset(
        input_patterns,
        seq_length,
        max_predictions_per_seq,
        batch_size,
        is_training=True,
        input_pipeline_context=ctx)
    return train_dataset

  return _dataset_fn if use_dataset_fn else _dataset_fn()


def get_loss_fn(loss_factor=1.0):
  """Returns loss function for BERT pretraining."""

  def _bert_pretrain_loss_fn(unused_labels, losses, **unused_args):
    return tf.keras.backend.mean(losses) * loss_factor

  return _bert_pretrain_loss_fn


def run_customized_training(strategy,
                            albert_config,
                            tinybert_config,
                            max_seq_length,
                            max_predictions_per_seq,
                            model_dir,
                            steps_per_epoch,
                            steps_per_loop,
                            epochs,
                            initial_lr,
                            warmup_steps,
                            input_files,
                            train_batch_size,
                            use_mlm_loss):
  """Run BERT pretrain model training using low-level API."""

  train_input_fn = functools.partial(get_pretrain_input_data, input_files,
                                     max_seq_length, max_predictions_per_seq,
                                     train_batch_size, strategy)

  with strategy.scope():
    # albert, albert_encoder = albert_model.pretrain_model(
    #     albert_config, max_seq_length, max_predictions_per_seq)
    train_model, albert, tinybert = tinybert_model.train_tinybert_model(
        tinybert_config, albert_config, max_seq_length, max_predictions_per_seq)
    albert.summary()
    tinybert.summary()
    train_model.summary()
    # train_model.to_json()
    # albert.to_json()
    # tinybert.to_json()
    if FLAGS.init_checkpoint:
      logging.info(f"model pre-trained weights loaded from {FLAGS.init_checkpoint}")
      train_model.load_weights(FLAGS.init_checkpoint)

    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=initial_lr,
                                                decay_steps=int(steps_per_epoch*epochs),end_learning_rate=0.0)

    if warmup_steps:
        learning_rate_fn = WarmUp(initial_learning_rate=initial_lr,
                                decay_schedule_fn=learning_rate_fn,
                                warmup_steps=warmup_steps)
    if FLAGS.optimizer == "lamp":
        optimizer_fn = LAMB
    else:
        optimizer_fn = AdamWeightDecay

    optimizer = optimizer_fn(
        learning_rate=learning_rate_fn,
        weight_decay_rate=FLAGS.weight_decay,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=FLAGS.adam_epsilon,
        exclude_from_weight_decay=['layer_norm', 'bias']) 
    train_model.optimizer = optimizer
  # 注意这里的model_dir是albert和tinybert共享，需要修改
  if FLAGS.do_train:
    trained_model = run_customized_training_loop(
        strategy=strategy,
        models=[albert, tinybert, train_model],
        model=train_model,
        albert=albert,
        tinybert=tinybert,
        start_wtih_trained_model=FLAGS.start_with_train_model,
        loss_fn=get_loss_fn(
            loss_factor=1.0 /
            strategy.num_replicas_in_sync),
        model_dir = model_dir,
        train_input_fn=train_input_fn,
        steps_per_epoch=steps_per_epoch,
        steps_per_loop=steps_per_loop,
        epochs=epochs,
        use_mlm_loss=use_mlm_loss)
  # Creates the BERT core model outside distribution strategy scope.
  training, albert, tinybert = tinybert_model.train_tinybert_model(
                                          tinybert_config, albert_config, 
                                          max_seq_length, 
                                          max_predictions_per_seq)

  # Restores the core model from model checkpoints and save weights only
  # contains the core model.
  # 在training的过程中会保存ckpt的模型文件，在训练结束后从ckpt读出模型再存储为h5文件
  # 寻找albert模型文件
  checkpoint_model = tf.train.Checkpoint(model=training)
  latest_checkpoint_file = tf.train.latest_checkpoint(model_dir)
  assert latest_checkpoint_file
  logging.info('Checkpoint file %s found and restoring from '
               'checkpoint', latest_checkpoint_file)
  status = checkpoint_model.restore(latest_checkpoint_file)
  status.assert_existing_objects_matched().expect_partial()
  # 寻找tinybert模型文件
  # checkpoint_tinybert = tf.train.Checkpoint(model=tinybert)
  # latest_tinybert_checkpoint_file = tf.train.latest_checkpoint(tinybert_model_dir)
  # assert latest_tinybert_checkpoint_file
  # logging.info('Checkpoint_Tinybert file %s found and restoring from '
  #              'checkpoint', latest_tinybert_checkpoint_file)
  # status_tinybert = checkpoint_albert.restore(latest_tinybert_checkpoint_file)
  # status_tinybert.assert_existing_objects_matched().expect_partial()
  # 创建存储文件
  if not os.path.exists(model_dir + '/models/'):
			os.makedirs(model_dir + '/models/')
  albert.save_weights(f"{model_dir}/models/albert_model.h5")
  tinybert.save_weights(f"{model_dir}/models/tinybert_model.h5")


def run_bert_pretrain(strategy,input_meta_data):
  """Runs BERT pre-training."""

  albert_config = AlbertConfig.from_json_file(FLAGS.albert_config_file)
  tinybert_config = TinybertConfig.from_json_file(FLAGS.tinybert_config_file)
  # print(albert_config, tinybert_config)
  if not strategy:
    raise ValueError('Distribution strategy is not specified.')

  # Runs customized training loop.
  logging.info('Training using customized training loop TF 2.0 with distrubuted'
               'strategy.')

  num_train_steps = None
  num_warmup_steps = None
  steps_per_epoch = None
  if FLAGS.do_train:
    len_train_examples = input_meta_data['train_data_size']
    logging.info('Training instance number is'+str(len_train_examples))
    steps_per_epoch = int(len_train_examples / FLAGS.train_batch_size)
    num_train_steps = int(
        len_train_examples / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  return run_customized_training(
      strategy,
      albert_config,
      tinybert_config,
      input_meta_data["max_seq_length"],
      input_meta_data["max_predictions_per_seq"],
      FLAGS.output_dir,
      FLAGS.steps_per_epoch,
      FLAGS.steps_per_loop,
      FLAGS.num_train_epochs,
      FLAGS.learning_rate,
      num_warmup_steps,
      FLAGS.input_files,
      FLAGS.train_batch_size,
      FLAGS.finetune_tinybert)


def main(_):
  # Users should always run this script under TF 2.x
  assert tf.version.VERSION.startswith('2.')

  # with tf.io.gfile.GFile(FLAGS.meta_data_file_path, 'rb') as reader:
  #   input_meta_data = json.loads(reader.read().decode('utf-8'))
  input_meta_data = {"max_seq_length":256, "max_predictions_per_seq":25, 'train_data_size':50000000}
  strategy = None
  if FLAGS.strategy_type == 'mirror':
    strategy = tf.distribute.MirroredStrategy()
  elif FLAGS.strategy_type == 'one':
    strategy = tf.distribute.OneDeviceStrategy('GPU:0')
  else:
    raise ValueError('The distribution strategy type is not supported: %s' %
                     FLAGS.strategy_type)
  if strategy:
    print('***** Number of cores used : ', strategy.num_replicas_in_sync)

  run_bert_pretrain(strategy,input_meta_data)

if __name__ == "__main__":
  flags.mark_flag_as_required("input_files")
  flags.mark_flag_as_required("albert_config_file")
  flags.mark_flag_as_required("output_dir")
  app.run(main)
