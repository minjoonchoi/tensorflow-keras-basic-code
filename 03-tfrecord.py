
import sys
import argparse

import tensorflow as tf
import numpy as np

def _bytes_feature(value):
    """string / byte value를 bytes_list 타입의 tf.train.Feature로 변환"""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """float / double value를 float_list 타입의 tf.train.Feature로 변환"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """bool / enum / int / uint value를 int64_list 타입의 tf.train.Feature로 변환"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#----------------------------------------------------------------------------

def convert():
    """각 타입 별 value를 tf.train.Feature 클래스 변수로 변환"""
    print(_bytes_feature(b'test_string'))
    print(_bytes_feature(u'test_bytes'.encode('utf-8')))

    print(_int64_feature(True))
    print(_int64_feature(1))
    
    feature = _float_feature(np.exp(1))
    
    print(feature)
    print()

    print(feature.SerializeToString())

#----------------------------------------------------------------------------

def serialize_example(feature0, feature1, feature2, feature3):
    """입력받는 매개변수들을 타입변환하고 tf.train.Example message를 생성"""
    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _int64_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _float_feature(feature3),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

#----------------------------------------------------------------------------

def tf_example():
    """각 타입 별 value를 tf.train.Example message를 생성"""
    serialized_example = serialize_example(False, 4, b'goat', 0.9876)
    print(serialized_example)
    
    example_proto = tf.train.Example.FromString(serialized_example)
    print(example_proto)

#----------------------------------------------------------------------------

def tf_record():
    """Numpy array들을 기반으로 tf.train.Example을 생성하고, TFRecord 파일에 저장한 뒤 결과 확인
    """
    
    #----------------------------------------------------------------------------
    # Data preparation
    
    # The number of observations in the dataset.
    n_observations = int(1e4)

    # Boolean feature, encoded as False or True.
    feature0 = np.random.choice([False, True], n_observations)

    # Integer feature, random from 0 to 4.
    feature1 = np.random.randint(0, 5, n_observations)
    
    # String feature.
    strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
    feature2 = strings[feature1]

    # Float feature, from a standard normal distribution.
    feature3 = np.random.randn(n_observations)
    
    #----------------------------------------------------------------------------
    # Writing TFRecords
    
    def tf_serialize_example(f0,f1,f2,f3):
        tf_string = tf.py_function(serialize_example,
            (f0, f1, f2, f3),  # Pass these args to the above function.
            tf.string      # The return type is `tf.string`.
            )
        return tf.reshape(tf_string, ()) # The result is a scalar.
    
    features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
    
    # Serializing Option 1 : map function
    serialized_features_dataset = features_dataset.map(tf_serialize_example)
    
    # Serializing Option 2 : generator
    def generator():
        for features in features_dataset:
            yield serialize_example(*features)
    serialized_features_dataset = tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())
    
    filename = 'test.tfrecord'
    
    # Writing option 1 : tf.data.experimental.TFRecordWriter
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_features_dataset)
    
    
    # Writing option 1 : tf.io.TFRecordWriter
    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(n_observations):
            example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
            writer.write(example)
            
    #----------------------------------------------------------------------------
    # Reading TFRecords
    
    filenames = [filename]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    
    for raw_record in raw_dataset.take(10):
      print(repr(raw_record))
    
    feature_description = {
    'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)
    
    parsed_dataset = raw_dataset.map(_parse_function)
    
    for parsed_record in parsed_dataset.take(10):
        print(repr(parsed_record))


#----------------------------------------------------------------------------

def tf_record_with_img():
    """Image 파일을 기반으로 tf.train.Example을 생성하고, TFRecord 파일에 저장한 뒤 결과 확인
    """
    
    #----------------------------------------------------------------------------
    # Data preparation
    
    cat_in_snow  = tf.keras.utils.get_file(
        '320px-Felis_catus-cat_on_snow.jpg',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')
    
    williamsburg_bridge = tf.keras.utils.get_file(
        '194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')
    
    image_labels = {
        cat_in_snow : 0,
        williamsburg_bridge : 1,
    }
    
    image_string = open(cat_in_snow, 'rb').read()
    label = image_labels[cat_in_snow]

    def image_example(image_string, label):
        image_shape = tf.io.decode_jpeg(image_string).shape

        feature = {
            'height': _int64_feature(image_shape[0]),
            'width': _int64_feature(image_shape[1]),
            'depth': _int64_feature(image_shape[2]),
            'label': _int64_feature(label),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_string),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

    #----------------------------------------------------------------------------
    # Writing TFRecords
    
    record_file = 'images.tfrecords'
    with tf.io.TFRecordWriter(record_file) as writer:
        for filename, label in image_labels.items():
            image_string = open(filename, 'rb').read()
            tf_example = image_example(image_string, label)
            writer.write(tf_example.SerializeToString())
        
    #----------------------------------------------------------------------------
    # Reading TFRecords
    raw_image_dataset = tf.data.TFRecordDataset(record_file)

    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_image_function(example_proto):
        return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    
    for image_features in parsed_image_dataset:
        # image_raw = image_features['image_raw'].numpy()
        label = image_features['label'].numpy()
        print(label)


#----------------------------------------------------------------------------

def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog        = prog,
        description = 'TFRecord 예제 코드',
        epilog      = 'Type "%s <command> -h" for more information.' % prog)

    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command(cmd='convert', desc='각 타입 별 value를 tf.train.Feature 클래스 변수로 변환')
    
    p = add_command(cmd='tf_example',  desc='각 타입 별 value를 tf.train.Example message를 생성')
    
    p = add_command(cmd='tf_record',  desc='Numpy array들을 기반으로 tf.train.Example을 생성하고, TFRecord 파일에 저장한 뒤 결과 확인')
    
    p = add_command(cmd='tf_record_with_img',  desc='Image 파일을 기반으로 tf.train.Example을 생성하고, TFRecord 파일에 저장한 뒤 결과 확인')

    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    execute_cmdline(sys.argv)