import classifier_data_lib
import tokenization
from absl import app, flags, logging
import tensorflow as tf
import json
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "train_data_path", '/work/chineseGLUEdatasets.v0.0.1/tnews/train.tf_record',
    "train_data path for tfrecords for the task.")

flags.DEFINE_string(
    "eval_data_path", '/work/chineseGLUEdatasets.v0.0.1/tnews/dev.tf_record',
    "eval_data path for tfrecords for the task.")

flags.DEFINE_string(
    "input_data_dir", '/work/chineseGLUEdatasets.v0.0.1/tnews/',
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string("input_meta_data_path",'/work/chineseGLUEdatasets.v0.0.1/tnews/input_mate_data.json',"input_meta_data_path")

flags.DEFINE_integer(
    "max_seq_length", 256,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "vocab_file", '/work/ALBERT-TF2.0-master/model_configs/base/vocab_chinese.txt',
    "The vocabulary file that the ALBERT model was trained on.")

flags.DEFINE_string("task_name", 'inews', "The name of the task to train.")

def create_inews_train_dev_file():
    tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file)
    
    input_meta_data = {"max_seq_length": 256,
                       "train_data_size": 0,
                       "eval_data_size": 0,
                        "num_labels": 0,
                       "processor_type": None}
    
    processors = {
    "cola": classifier_data_lib.ColaProcessor,
    "sts": classifier_data_lib.StsbProcessor,
    "sst": classifier_data_lib.Sst2Processor,
    "mnli": classifier_data_lib.MnliProcessor,
    "qnli": classifier_data_lib.QnliProcessor,
    "qqp": classifier_data_lib.QqpProcessor,
    "rte": classifier_data_lib.RteProcessor,
    "mrpc": classifier_data_lib.MrpcProcessor,
    "wnli": classifier_data_lib.WnliProcessor,
    "xnli": classifier_data_lib.XnliProcessor,
    "inews": classifier_data_lib.InewsProcessor,
    }

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    logging.info("processor is : ", FLAGS.task_name)
    input_meta_data["processor_type"] = FLAGS.task_name

    train_examples, labels = processor.get_train_examples(FLAGS.input_data_dir)
    label_list = processor.get_labels(labels)
    label_map = {i:label for i,label in enumerate(label_list)}
    input_meta_data["train_data_size"] = len(train_examples)
    input_meta_data["num_labels"] = len(label_list)
    
    classifier_data_lib.file_based_convert_examples_to_features(train_examples,
                                        label_list, FLAGS.max_seq_length,
                                        tokenizer, FLAGS.train_data_path)


    eval_examples, _ = processor.get_dev_examples(FLAGS.input_data_dir)
    label_list = processor.get_labels(labels)
    label_map = {i:label for i,label in enumerate(label_list)}
    input_meta_data["eval_data_size"] = len(eval_examples)
    
    classifier_data_lib.file_based_convert_examples_to_features(eval_examples,
                                        label_list, FLAGS.max_seq_length,
                                        tokenizer, FLAGS.eval_data_path)

    with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'w') as reader:
        reader.write(json.dumps(input_meta_data))

    print("done")
   
   
def create_tnews_train_dev_file():
    tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file)
    
    input_meta_data = {"max_seq_length": 256,
                       "train_data_size": 0,
                       "eval_data_size": 0,
                        "num_labels": 0,
                       "processor_type": None}
    
    processors = {
    "cola": classifier_data_lib.ColaProcessor,
    "sts": classifier_data_lib.StsbProcessor,
    "sst": classifier_data_lib.Sst2Processor,
    "mnli": classifier_data_lib.MnliProcessor,
    "qnli": classifier_data_lib.QnliProcessor,
    "qqp": classifier_data_lib.QqpProcessor,
    "rte": classifier_data_lib.RteProcessor,
    "mrpc": classifier_data_lib.MrpcProcessor,
    "wnli": classifier_data_lib.WnliProcessor,
    "xnli": classifier_data_lib.XnliProcessor,
    "inews": classifier_data_lib.InewsProcessor,
    }

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    logging.info("processor is : ", FLAGS.task_name)
    input_meta_data["processor_type"] = FLAGS.task_name

    train_examples, labels = processor.get_train_examples(FLAGS.input_data_dir)
    label_list = processor.get_labels(labels)
    label_map = {i:label for i,label in enumerate(label_list)}
    input_meta_data["train_data_size"] = len(train_examples)
    input_meta_data["num_labels"] = len(label_list)
    
    classifier_data_lib.file_based_convert_examples_to_features(train_examples,
                                        label_list, FLAGS.max_seq_length,
                                        tokenizer, FLAGS.train_data_path)


    eval_examples, _ = processor.get_dev_examples(FLAGS.input_data_dir)
    label_list = processor.get_labels(labels)
    label_map = {i:label for i,label in enumerate(label_list)}
    input_meta_data["eval_data_size"] = len(eval_examples)
    
    classifier_data_lib.file_based_convert_examples_to_features(eval_examples,
                                        label_list, FLAGS.max_seq_length,
                                        tokenizer, FLAGS.eval_data_path)

    with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'w') as reader:
        reader.write(json.dumps(input_meta_data))

    print("done")
      

 

def create_xnli_train_dev_file():
    tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file)
    
    input_meta_data = {"max_seq_length": 256,
                       "train_data_size": 0,
                       "eval_data_size": 0,
                        "num_labels": 0,
                       "processor_type": None}
    
    processors = {
    "cola": classifier_data_lib.ColaProcessor,
    "sts": classifier_data_lib.StsbProcessor,
    "sst": classifier_data_lib.Sst2Processor,
    "mnli": classifier_data_lib.MnliProcessor,
    "qnli": classifier_data_lib.QnliProcessor,
    "qqp": classifier_data_lib.QqpProcessor,
    "rte": classifier_data_lib.RteProcessor,
    "mrpc": classifier_data_lib.MrpcProcessor,
    "wnli": classifier_data_lib.WnliProcessor,
    "xnli": classifier_data_lib.XnliProcessor,
    "inews": classifier_data_lib.InewsProcessor,
    }

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    logging.info("processor is : ", FLAGS.task_name)
    input_meta_data["processor_type"] = FLAGS.task_name

    train_examples = processor.get_train_examples(FLAGS.input_data_dir)
    label_list = processor.get_labels()
    label_map = {i:label for i,label in enumerate(label_list)}
    input_meta_data["train_data_size"] = len(train_examples)
    input_meta_data["num_labels"] = len(label_list)
    
    classifier_data_lib.file_based_convert_examples_to_features(train_examples,
                                        label_list, FLAGS.max_seq_length,
                                        tokenizer, FLAGS.train_data_path)


    eval_examples = processor.get_dev_examples(FLAGS.input_data_dir)
    label_list = processor.get_labels()
    label_map = {i:label for i,label in enumerate(label_list)}
    input_meta_data["eval_data_size"] = len(eval_examples)
    
    classifier_data_lib.file_based_convert_examples_to_features(eval_examples,
                                        label_list, FLAGS.max_seq_length,
                                        tokenizer, FLAGS.eval_data_path)

    with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'w') as reader:
        reader.write(json.dumps(input_meta_data))

    print("done")
    


def main(_):
    create_tnews_train_dev_file()

if __name__ == "__main__":
    app.run(main)