import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
import argparse
import sys
import requests

# Helper libraries
import numpy as np

print(tf.__version__)
print(keras.__version__)

width = 1640 #120
height = 1232 #68

def build_graph(mbx, mby):
    g = tf.Graph()
    with g.as_default():
        input = tf.placeholder(shape=[1, 104, 78, 2], dtype=tf.float32, name="input")
        # shape = tf.shape(input)
        print(input)
        # split = input[:,:,:,0:2]
        # stack = tf.cast(input[:,:,:,0:2], dtype=tf.float32)
        # print(stack)
        # cast = tf.cast(input, dtype=tf.float32)
        norm = tf.norm(input, axis=3, keepdims=True, name="norm")
        print(norm)
        # reshape = tf.reshape(norm, shape=[shape[0], shape[1]*shape[2]])
        # red = tf.math.reduce_max(norm, axis=1)
        red = tf.nn.avg_pool(norm, ksize=[1,104,78,1], strides=[1,104,78,1], padding="SAME")
        print(red)
        output = tf.identity(red, name="output")
        print(output)

    return g, input, output


def save_graph(g, input, output, version):
    with g.as_default():
        tf.train.write_graph(g.as_graph_def(), 'model', 'process_motion.pb', as_text=False)

        with g.as_default():
            with tf.Session() as sess:
                tf.saved_model.simple_save(sess, 'saved_model/{}'.format(version), inputs={"input":input}, outputs={"output": output})


def main(args):
    mbx = 1 + int((args.width + 15) / 16)
    mby = 1 + int(args.height / 16)

    g, input, output = build_graph(mbx, mby)

    if args.save_model:
        save_graph(g, input, output, args.version)

    if args.read_url == "":
        return
    
    while True:
        res = requests.get(args.read_url)

        if not res:
            print('status code: {}'.format(res.status_code))
            return
        
        dat = np.frombuffer(res.content, dtype=np.int8)
        dat = np.reshape(dat, [1, mbx, mby, 4])
        dat = dat[:,:,:,0:2]
        # dat = dat.astype(np.float)

        with g.as_default():
            with tf.Session() as sess:
                o = sess.run((output), feed_dict={input:dat})

        print('motion: {}'.format(o))

        if not args.repeat:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", default=width, help="width of video")
    parser.add_argument("--height", default=height, help="height of video")
    parser.add_argument("--read_url", default="", help="url to read motion vectors from")
    parser.add_argument("--repeat", default=False, help="repeatedly call the model on the read_url")
    parser.add_argument("--version", default="1", help="version of saved_model")
    parser.add_argument("--save_model", default=False, help="output the frozen and savedmodel")
    main(parser.parse_args())

