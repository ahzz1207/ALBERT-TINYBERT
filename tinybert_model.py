"""TINYBERT models that are compatible with TF 2.0."""

from __future__ import absolute_import, division, print_function

import copy

import tensorflow as tf

from albert import AlbertConfig, AlbertModel
from tinybert import TinybertConfig, TinybertModel
from utils import tf_utils

class TinyBertPretrainLayer(tf.keras.layers.Layer):
    """Wrapper layer for pre-training a ALBERT model.
    This layer wraps an existing `albert_layer` which is a Keras Layer.
    It outputs `sequence_output` from TransformerBlock sub-layer and
    `sentence_output` which are suitable for feeding into a ALBertPretrainLoss
    layer. This layer can be used along with an unsupervised input to
    pre-train the embeddings for `albert_layer`.
    """

    def __init__(self,
                config,
                tinybert_layer,
                initializer=None,
                float_type=tf.float32,
                **kwargs):
        super(TinyBertPretrainLayer, self).__init__(**kwargs)
        self.config = copy.deepcopy(config)
        self.float_type = float_type

        self.embedding_table = tinybert_layer.embedding_lookup.embeddings
        self.num_next_sentence_label = 2
        if initializer:
            self.initializer = initializer
        else:
            self.initializer = tf.keras.initializers.TruncatedNormal(
                stddev=self.config.initializer_range)

    def build(self, unused_input_shapes):
        """Implements build() for the layer."""
        self.output_bias = self.add_weight(
            shape=[self.config.vocab_size],
            name='predictions/output_bias',
            initializer=tf.keras.initializers.Zeros())
        self.lm_dense = tf.keras.layers.Dense(
            self.config.embedding_size,
            activation=tf_utils.get_activation(self.config.hidden_act),
            kernel_initializer=self.initializer,
            name='predictions/transform/dense')
        self.lm_layer_norm = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-12, name='predictions/transform/LayerNorm')

        # Next sentence binary classification dense layer including bias to match
        # TF1.x BERT variable shapes.
        with tf.name_scope('seq_relationship'):
            self.next_seq_weights = self.add_weight(
                shape=[self.num_next_sentence_label, self.config.hidden_size],
                name='output_weights',
                initializer=self.initializer)
            self.next_seq_bias = self.add_weight(
                shape=[self.num_next_sentence_label],
                name='output_bias',
                initializer=tf.keras.initializers.Zeros())
        super(TinyBertPretrainLayer, self).build(unused_input_shapes)

    def __call__(self,
                pooled_output,
                sequence_output=None,
                masked_lm_positions=None,
                **kwargs):
        inputs = tf_utils.pack_inputs(
            [pooled_output, sequence_output, masked_lm_positions])
        return super(TinyBertPretrainLayer, self).__call__(inputs, **kwargs)

    def call(self, inputs):
        """Implements call() for the layer."""
        unpacked_inputs = tf_utils.unpack_inputs(inputs)
        pooled_output = unpacked_inputs[0]
        sequence_output = unpacked_inputs[1]
        masked_lm_positions = unpacked_inputs[2]

        mask_lm_input_tensor = tf_utils.gather_indexes(sequence_output, masked_lm_positions)
        lm_output = self.lm_dense(mask_lm_input_tensor)
        lm_output = self.lm_layer_norm(lm_output)
        lm_output = tf.matmul(lm_output, self.embedding_table, transpose_b=True)
        lm_output = tf.nn.bias_add(lm_output, self.output_bias)
        lm_output = tf.nn.log_softmax(lm_output, axis=-1)

        logits = tf.matmul(pooled_output, self.next_seq_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.next_seq_bias)
        sentence_output = tf.nn.log_softmax(logits, axis=-1)
        return (lm_output, sentence_output)


class ALBertPretrainLayer(tf.keras.layers.Layer):
  """Wrapper layer for pre-training a ALBERT model.
  This layer wraps an existing `albert_layer` which is a Keras Layer.
  It outputs `sequence_output` from TransformerBlock sub-layer and
  `sentence_output` which are suitable for feeding into a ALBertPretrainLoss
  layer. This layer can be used along with an unsupervised input to
  pre-train the embeddings for `albert_layer`.
  """

  def __init__(self,
               config,
               albert_layer,
               initializer=None,
               float_type=tf.float32,
               **kwargs):
    super(ALBertPretrainLayer, self).__init__(**kwargs)
    self.config = copy.deepcopy(config)
    self.float_type = float_type

    self.embedding_table = albert_layer.embedding_lookup.embeddings
    self.num_next_sentence_label = 2
    if initializer:
      self.initializer = initializer
    else:
      self.initializer = tf.keras.initializers.TruncatedNormal(
          stddev=self.config.initializer_range)

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.output_bias = self.add_weight(
        shape=[self.config.vocab_size],
        name='predictions/output_bias',
        initializer=tf.keras.initializers.Zeros())
    self.lm_dense = tf.keras.layers.Dense(
        self.config.embedding_size,
        activation=tf_utils.get_activation(self.config.hidden_act),
        kernel_initializer=self.initializer,
        name='predictions/transform/dense')
    self.lm_layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-12, name='predictions/transform/LayerNorm')

    # Next sentence binary classification dense layer including bias to match
    # TF1.x BERT variable shapes.
    with tf.name_scope('seq_relationship'):
      self.next_seq_weights = self.add_weight(
          shape=[self.num_next_sentence_label, self.config.hidden_size],
          name='output_weights',
          initializer=self.initializer)
      self.next_seq_bias = self.add_weight(
          shape=[self.num_next_sentence_label],
          name='output_bias',
          initializer=tf.keras.initializers.Zeros())
    super(ALBertPretrainLayer, self).build(unused_input_shapes)

  def __call__(self,
               pooled_output,
               sequence_output=None,
               masked_lm_positions=None,
               **kwargs):
    inputs = tf_utils.pack_inputs(
        [pooled_output, sequence_output, masked_lm_positions])
    return super(ALBertPretrainLayer, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    """Implements call() for the layer."""
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    pooled_output = unpacked_inputs[0]
    sequence_output = unpacked_inputs[1]
    masked_lm_positions = unpacked_inputs[2]

    mask_lm_input_tensor = tf_utils.gather_indexes(sequence_output, masked_lm_positions)
    lm_output = self.lm_dense(mask_lm_input_tensor)
    lm_output = self.lm_layer_norm(lm_output)
    lm_output = tf.matmul(lm_output, self.embedding_table, transpose_b=True)
    lm_output = tf.nn.bias_add(lm_output, self.output_bias)
    # lm_output = tf.nn.softmax(lm_output, axis=-1)
    lm_output = tf.nn.log_softmax(lm_output, axis=-1)

    logits = tf.matmul(pooled_output, self.next_seq_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, self.next_seq_bias)
    sentence_output = tf.nn.log_softmax(logits, axis=-1)
    return (lm_output, sentence_output)


class TinyBertPretrainLossAndMetricLayer(tf.keras.layers.Layer):
  """Returns layer that computes custom loss and metrics for pretraining."""

  def __init__(self, bert_config, **kwargs):
    super(TinyBertPretrainLossAndMetricLayer, self).__init__(**kwargs)
    self.config = copy.deepcopy(bert_config)

  def __call__(self,
               lm_output,
               sentence_output=None,
               lm_label_ids=None,
               lm_label_weights=None,
               sentence_labels=None,
               **kwargs):
    inputs = tf_utils.pack_inputs([
        lm_output, sentence_output, lm_label_ids, lm_label_weights,
        sentence_labels
    ])
    return super(TinyBertPretrainLossAndMetricLayer, self).__call__(
        inputs, **kwargs)

  def _add_metrics(self, lm_output, lm_labels, lm_label_weights,
                   lm_per_example_loss, sentence_output, sentence_labels,
                   sentence_per_example_loss):
    """Adds metrics."""
    masked_lm_accuracy = tf.keras.metrics.sparse_categorical_accuracy(
        lm_labels, lm_output)
    masked_lm_accuracy = tf.reduce_mean(masked_lm_accuracy * lm_label_weights)
    self.add_metric(
        masked_lm_accuracy, name='masked_lm_accuracy', aggregation='mean')

    lm_example_loss = tf.reshape(lm_per_example_loss, [-1])
    lm_example_loss = tf.reduce_mean(lm_example_loss * lm_label_weights)
    self.add_metric(lm_example_loss, name='lm_example_loss', aggregation='mean')

    sentence_order_accuracy = tf.keras.metrics.sparse_categorical_accuracy(
        sentence_labels, sentence_output)
    self.add_metric(
        sentence_order_accuracy,name='sentence_order_accuracy',aggregation='mean')

    sentence_order_mean_loss = tf.reduce_mean(sentence_per_example_loss)
    self.add_metric(
        sentence_order_mean_loss, name='sentence_order_mean_loss', aggregation='mean')

  def call(self, inputs):
    """Implements call() for the layer."""
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    lm_output = unpacked_inputs[0]
    sentence_output = unpacked_inputs[1]
    lm_label_ids = unpacked_inputs[2]
    lm_label_ids = tf.keras.backend.reshape(lm_label_ids, [-1])
    lm_label_ids_one_hot = tf.keras.backend.one_hot(lm_label_ids,
                                                    self.config.vocab_size)
    lm_label_weights = tf.keras.backend.cast(unpacked_inputs[3], tf.float32)
    lm_label_weights = tf.keras.backend.reshape(lm_label_weights, [-1])
    lm_per_example_loss = -tf.keras.backend.sum(
        lm_output * lm_label_ids_one_hot, axis=[-1])
    numerator = tf.keras.backend.sum(lm_label_weights * lm_per_example_loss)
    denominator = tf.keras.backend.sum(lm_label_weights) + 1e-5
    mask_label_loss = numerator / denominator

    sentence_labels = unpacked_inputs[4]
    sentence_labels = tf.keras.backend.reshape(sentence_labels, [-1])
    sentence_label_one_hot = tf.keras.backend.one_hot(sentence_labels, 2)
    per_example_loss_sentence = -tf.keras.backend.sum(
        sentence_label_one_hot * sentence_output, axis=-1)
    sentence_loss = tf.keras.backend.mean(per_example_loss_sentence)
    loss = mask_label_loss + sentence_loss
    # TODO(hongkuny): Avoids the hack and switches add_loss.
    final_loss = tf.fill(
        tf.keras.backend.shape(per_example_loss_sentence), loss)

    self._add_metrics(lm_output, lm_label_ids, lm_label_weights,
                      lm_per_example_loss, sentence_output, sentence_labels,
                      per_example_loss_sentence)
    return final_loss


class TinybertLossLayer(tf.keras.layers.Layer):
    """
        这一计算层用于蒸馏，将tinybert的transformer层对应bert中的每一层，
        并且计算两个bert的pooled_out间的mse
    """
    
    def __init__(self,
                 config,
                 initializer = None,
                 **kwargs):
        super(TinybertLossLayer, self).__init__(**kwargs)
        self.config = copy.deepcopy(config)
        # self.albert_embedding_table = albert_layer.embedding_lookup.embeddings
        # self.tinybert_embedding_table = tinybert_layer.embedding_lookup.embeddings
        self.num_layers = config.num_hidden_layers
        if initializer:
            self.initializer = initializer
        else:
            self.initializer = tf.keras.initializers.TruncatedNormal(
                stddev=self.config.initializer_range)
    
    def __call__(self,
                 albert_embedding_table,
                 tinybert_embedding_table,
                 albert_pooled_output,
                 tinybert_pooled_output=None,
                 albert_seq_output=None,
                 tinybert_seq_output=None,
                 albert_atten_score=None,
                 tinybert_atten_score=None,
                 albert_lm_out=None,
                 tinybert_lm_out=None,
                 lm_label_ids=None,
                 lm_label_weights=None,
                 **kwargs):
        inputs = tf_utils.pack_inputs([albert_embedding_table, tinybert_embedding_table, albert_pooled_output, tinybert_pooled_output, albert_seq_output, tinybert_seq_output, albert_atten_score, tinybert_atten_score, albert_lm_out, tinybert_lm_out,lm_label_ids, lm_label_weights])
        return super(TinybertLossLayer, self).__call__(inputs, **kwargs)
    
    def call(self, inputs):
        """Implements call() for the layer."""
        unpacked_inputs = tf_utils.unpack_inputs(inputs)
        albert_embedding_table = unpacked_inputs[0]
        tinybert_embedding_table = unpacked_inputs[1]
        albert_pooled_output = unpacked_inputs[2]
        tinybert_pooled_output = unpacked_inputs[3]
        albert_sequence_output = unpacked_inputs[4]
        tinybert_sequence_output = unpacked_inputs[5]
        albert_atten_score = unpacked_inputs[6]
        tinybert_atten_score = unpacked_inputs[7]
        albert_lm_out = unpacked_inputs[8]
        tinybert_lm_out = unpacked_inputs[9]
        
        embeddings_loss = tf.keras.losses.MSE(y_true=albert_embedding_table, 
                                              y_pred=tinybert_embedding_table)
        embeddings_loss = tf.reduce_mean(embeddings_loss)
        
        attention_loss = 0
        hidden_loss = 0 
        for i in range(5):
            #print(albert_atten_score, tinybert_atten_score, albert_sequence_output, tinybert_sequence_output)
            atten_loss = tf.keras.losses.MSE(y_true=albert_atten_score[i*2],
                                                y_pred=tinybert_atten_score[i])
            attention_loss += tf.reduce_mean(atten_loss)
            seq_loss = tf.keras.losses.MSE(y_true=albert_sequence_output[i*2], 
                                            y_pred=tinybert_sequence_output[i])
            hidden_loss += tf.reduce_mean(seq_loss)
        
        lm_out_loss_fn = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        lm_out_loss = lm_out_loss_fn(albert_lm_out,
                                    tinybert_lm_out)
        lm_out_loss = tf.reduce_mean(lm_out_loss)
        print(embeddings_loss ,lm_out_loss)
        return embeddings_loss + attention_loss + hidden_loss + lm_out_loss
    
    
def train_tinybert_model(tinybert_config,
                   albert_config,
                   seq_length,
                   max_predictions_per_seq,
                   initializer=None):
    input_word_ids = tf.keras.layers.Input(
        shape=(seq_length,), name='input_word_ids', dtype=tf.int32)
    input_mask = tf.keras.layers.Input(
        shape=(seq_length,), name='input_mask', dtype=tf.int32)
    input_type_ids = tf.keras.layers.Input(
        shape=(seq_length,), name='input_type_ids', dtype=tf.int32)
    masked_lm_positions = tf.keras.layers.Input(
        shape=(max_predictions_per_seq,),
        name='masked_lm_positions',
        dtype=tf.int32)
    masked_lm_weights = tf.keras.layers.Input(
        shape=(max_predictions_per_seq,),
        name='masked_lm_weights',
        dtype=tf.int32)
    next_sentence_labels = tf.keras.layers.Input(
        shape=(1,), name='next_sentence_labels', dtype=tf.int32)
    masked_lm_ids = tf.keras.layers.Input(
        shape=(max_predictions_per_seq,), name='masked_lm_ids', dtype=tf.int32)
    
    ##bert_base teacher
    float_type = tf.float32
    albert_encoder = "albert_model"
    albert_layer = AlbertModel(config=albert_config, float_type=float_type, name=albert_encoder)
    albert_pooled_output, albert_sequence_output, albert_attention_scores = albert_layer(input_word_ids, input_mask,input_type_ids)
    ##tinybert student
    float_type = tf.float32
    tinybert_encoder = "tinybert_model"
    tinybert_layer = TinybertModel(config=tinybert_config, float_type=float_type, name=tinybert_encoder)
    tinybert_pooled_output, tinybert_sequence_output, tinybert_attention_scores = tinybert_layer(input_word_ids, input_mask, input_type_ids)
    print(tinybert_attention_scores, tinybert_sequence_output)
    albert_teacher = tf.keras.Model(
        inputs = [input_word_ids, input_mask, input_type_ids],
        outputs = [albert_pooled_output, albert_sequence_output, albert_attention_scores]
    )
    
    tinybert_student = tf.keras.Model(
        inputs = [input_word_ids, input_mask, input_type_ids],
        outputs = [tinybert_pooled_output, tinybert_sequence_output, tinybert_attention_scores]
    )
    
    teacher_pooled_output = albert_teacher.outputs[0]
    teacher_sequence_output = albert_teacher.outputs[1]
    teacher_atten_score = albert_teacher.outputs[2]
    student_pooled_output = tinybert_student.outputs[0]
    student_sequence_output = tinybert_student.outputs[1]
    student_atten_score = tinybert_student.outputs[2]
    # dislit loss
        
    tinybert_pretrain_layer = TinyBertPretrainLayer(tinybert_config,
                                                    tinybert_student.get_layer(tinybert_encoder),
                                                    initializer=initializer,
                                                    name='tinybert_cls')
    tinybert_lm_output, tinybert_sentence_output = tinybert_pretrain_layer(student_pooled_output, student_sequence_output, masked_lm_positions)
   
    albert_pretrain_layer = ALBertPretrainLayer(albert_config,
                                                albert_teacher.get_layer(albert_encoder),
                                                initializer=initializer,
                                                name='albert_cls')
    albert_lm_output, albert_sentence_output = albert_pretrain_layer(teacher_pooled_output, teacher_sequence_output, masked_lm_positions)
    
    print(teacher_atten_score, teacher_sequence_output)
    tinybert_loss_layer = TinybertLossLayer(tinybert_config, name="dislit")
    loss_output = tinybert_loss_layer(
                                      albert_layer.embedding_lookup.embeddings,
                                      tinybert_layer.embedding_lookup.embeddings,
                                      teacher_pooled_output,
                                      student_pooled_output,
                                      teacher_sequence_output,
                                      student_sequence_output,
                                      teacher_atten_score,
                                      student_atten_score,
                                      albert_lm_output,
                                      tinybert_lm_output,
                                      masked_lm_ids,
                                      masked_lm_weights)
    
    # pretrain_loss
    tinybert_pretrain_loss_metrics_layer = TinyBertPretrainLossAndMetricLayer(tinybert_config, name="metric")
    
    pretrain_loss = tinybert_pretrain_loss_metrics_layer(tinybert_lm_output, tinybert_sentence_output, masked_lm_ids,
                                                masked_lm_weights, next_sentence_labels)    
    # total_loss
    total_loss = loss_output + pretrain_loss
    
    return tf.keras.Model(
        inputs={
            'input_word_ids': input_word_ids,
            'input_mask': input_mask,
            'input_type_ids': input_type_ids,
            'masked_lm_positions': masked_lm_positions,
            'masked_lm_ids': masked_lm_ids,
            'masked_lm_weights': masked_lm_weights,
            'next_sentence_labels': next_sentence_labels,
        },
        outputs = total_loss), albert_teacher, tinybert_student