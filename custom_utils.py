from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.metrics import Metric
import numpy as np

class SparseMatthewsCorrelationCoefficient(Metric):
    """Computes the Matthews Correlation Coefficient.
    The statistic is also known as the phi coefficient.
    The Matthews correlation coefficient (MCC) is used in 
    machine learning as a measure of the quality of binary 
    and multiclass classifications. It takes into account 
    true and false positives and negatives and is generally 
    regarded as a balanced measure which can be used even 
    if the classes are of very different sizes. The correlation 
    coefficient value of MCC is between -1 and +1. A 
    coefficient of +1 represents a perfect prediction, 
    0 an average random prediction and -1 an inverse 
    prediction. The statistic is also known as 
    the phi coefficient.
    MCC = (TP * TN) - (FP * FN) /
          ((TP + FP) * (TP + FN) * (TN + FP ) * (TN + FN))^(1/2)
    Args:
       num_classes : Number of unique classes in the dataset.
    Returns:
       mcc : float (the Matthews correlation coefficient)
    Usage:
    ```python
    actuals = tf.constant([[1, 0, 1], [0, 1, 0]],
             dtype=tf.int32)
    preds = tf.constant([[1, 0, 0],[0, 1, 1]],
             dtype=tf.int32)             
    # Matthews correlation coefficient
    mcc = MatthewsCorrelationCoefficient(num_classes=3)
	output.update_state(actuals, preds)
	print('MCC:', output.result().numpy())
	# mcc : 
    ```
    """
    
    def __init__(self, 
                 num_classes, 
                 name='Sparse_Matthews_Correlation_Coefficient', 
                 dtype=tf.int32):
        super(SparseMatthewsCorrelationCoefficient, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(
            'true_positives',
            shape=[self.num_classes],
            initializer='zeros',
            dtype=self.dtype)
        self.false_positives = self.add_weight(
            'false_positives',
            shape=[self.num_classes],
            initializer='zeros',
            dtype=self.dtype)
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=[self.num_classes],
            initializer='zeros',
            dtype=self.dtype)
        self.true_negatives = self.add_weight(
            'true_negatives',
            shape=[self.num_classes],
            initializer='zeros',
            dtype=self.dtype)

    def update_state(self, y_true, y_pred):
        y_pred = tf.expand_dims(tf.argmax(y_pred, axis=1), -1)
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        true_positive = tf.math.count_nonzero(y_true * y_pred, 0)
        # predicted sum
        pred_sum = tf.math.count_nonzero(y_pred, 0)
        # Ground truth label sum
        true_sum = tf.math.count_nonzero(y_true, 0)
        false_positive = pred_sum - true_positive
        false_negative = true_sum - true_positive
        true_negative = y_true.get_shape(
        )[0] - true_positive - false_positive - false_negative

        # true positive state_update
        self.true_positives.assign_add(tf.cast(true_positive, self.dtype))
        # false positive state_update
        self.false_positives.assign_add(tf.cast(false_positive, self.dtype))
        # false negative state_update
        self.false_negatives.assign_add(tf.cast(false_negative, self.dtype))
        # true negative state_update
        self.true_negatives.assign_add(tf.cast(true_negative, self.dtype))


    def result(self):
    	  # numerator
    	  numerator1 = tf.cast(self.true_positives * self.true_negatives, self.dtype)
    	  numerator2 = tf.cast(self.false_positives * self.false_negatives, self.dtype)
    	  numerator = numerator1 - numerator2
    	  # denominator
    	  denominator1 = tf.cast(self.true_positives + self.false_positives, self.dtype)
    	  denominator2 = tf.cast(self.true_positives + self.false_negatives, self.dtype)
    	  denominator3 = tf.cast(self.true_negatives + self.false_positives, self.dtype)
    	  denominator4 = tf.cast(self.true_negatives + self.false_negatives, self.dtype)
    	  denominator = tf.math.sqrt(denominator1 * denominator2 * denominator3 * denominator4)
    	  mcc = tf.math.divide_no_nan(numerator, denominator)

    	  return mcc

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
        }
        base_config = super(SparseMatthewsCorrelationCoefficient, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def reset_states(self):
        """Resets all of the metric state variables."""
        self.true_positives.assign(np.zeros(self.num_classes), np.int32)
        self.false_positives.assign(np.zeros(self.num_classes), np.int32)
        self.false_negatives.assign(np.zeros(self.num_classes), np.int32)
        self.true_negatives.assign(np.zeros(self.num_classes), np.int32)

class PearsonCorr(tf.keras.metrics.Metric):
    def __init__(self, name='PearsonCorrelation', **kwargs):
      super(PearsonCorr, self).__init__(name=name, **kwargs)
      self.ux = self.add_weight(name='pearCor_mu_x', initializer='zeros')
      self.uy = self.add_weight(name='pearCor_mu_y', initializer='zeros')
      self.n = self.add_weight(name='pearCor_n', initializer='zeros')
      self.N = self.add_weight(name='pearCor_N', initializer='zeros')
      self.D = self.add_weight(name='pearCor_D', initializer='zeros')
      self.E = self.add_weight(name='pearCor_E', initializer='zeros')

    def update_state(self, x, y, sample_weight=None):
      x = tf.cast(x, tf.float32)
      y = tf.cast(y, tf.float32)
      x = tf.squeeze(x)
      y = tf.squeeze(y)

      m = tf.cast(tf.shape(x)[0], tf.float32)

      self.ux.assign((self.ux * self.n + tf.reduce_sum(x)) / (self.n + m))
      self.uy.assign((self.uy * self.n + tf.reduce_sum(y)) / (self.n + m))
      self.n.assign_add(m)

      x_zero_mean = x - self.ux
      y_zero_mean = y - self.uy
      
      self.N.assign_add(tf.tensordot(x_zero_mean, y_zero_mean, axes=1))
      self.D.assign_add(tf.tensordot(x_zero_mean, x_zero_mean, axes=1))
      self.E.assign_add(tf.tensordot(y_zero_mean, y_zero_mean, axes=1))

    def result(self):
      return self.N / (tf.sqrt(self.D * self.E))

glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
}

import logging
logger = logging.getLogger(__name__)
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures

def glue_convert_examples_to_features_unused0(examples, tokenizer,
                                      max_length=512,
                                      task=None,
                                      label_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """
    is_tf_dataset = False
    if isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=label))

    if is_tf_dataset:
        def gen():
            for ex in features:
              ex.input_ids[0] = 1
              yield ({'input_ids': ex.input_ids,
                         'attention_mask': ex.attention_mask,
                         'token_type_ids': ex.token_type_ids},
                          ex.label)

        return tf.data.Dataset.from_generator(gen,
            ({'input_ids': tf.int32,
              'attention_mask': tf.int32,
              'token_type_ids': tf.int32},
              tf.int64),
            ({'input_ids': tf.TensorShape([None]),
              'attention_mask': tf.TensorShape([None]),
              'token_type_ids': tf.TensorShape([None])},
             tf.TensorShape([])))

    return features