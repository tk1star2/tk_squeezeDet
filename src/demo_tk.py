from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os.path
import numpy as np
import tensorflow as tf
from config import *
from nets import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'KITTI',
                           """Currently support PASCAL_VOC or KITTI dataset.""")
tf.app.flags.DEFINE_string('data_path', '', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'test',
                           """Only used for VOC data."""
                           """Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('year', '2007',
                            """VOC challenge year. 2007 or 2012"""
                            """Only used for VOC data""")
tf.app.flags.DEFINE_string('eval_dir', '/tmp/bichen/logs/squeezeDet/eval',
                            """Directory where to write event logs """)
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/bichen/logs/squeezeDet/train',
                            """Path to the training checkpoint.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 1,
                             """How often to check if new cpt is saved.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                             """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('net', 'squeezeDet',
                           """Neural net architecture.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")


def main(argv=None):

  """Load weights from a pre-trained squeezeDet network trained with Batch > 1
  and save the model with batch = 1 for production"""

  with tf.Graph().as_default():

    mc = kitti_squeezeDetPlus_config()
    mc.BATCH_SIZE = 1
    mc.LOAD_PRETRAINED_MODEL = False
    model = SqueezeDetPlus(mc, FLAGS.gpu)

    saver = tf.train.Saver(model.model_params)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        # Restores from checkpoint
        ckpts = set()
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        ckpts.add(ckpt.model_checkpoint_path)
        print ('Loading {}...'.format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)

        sess.run(tf.initialize_all_variables())

        # Run one image to test that it works
        read_full_name = "/data/squeezeDet_TF011/src/test2.jpg"
        im = cv2.imread(read_full_name)
        im = im.astype(np.float32, copy=False)
        im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
        input_image = im - mc.BGR_MEANS

        # Detect
        det_boxes, det_probs, det_class = sess.run(
            [model.det_boxes, model.det_probs, model.det_class],
            feed_dict={model.image_input: [input_image], model.keep_prob: 1.0})  # works fine

        # Save to disk
        checkpoint_path = os.path.join("/data/squeezeDet_TF011/logs/test_freeze", 'evalBatch1.ckpt')
        step = 1
        saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    tf.app.run()

