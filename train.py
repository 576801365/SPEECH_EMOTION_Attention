import tensorflow as tf
import numpy as np
import cnn
import _pickle as cPickle
import os
FLAGS = cnn.FLAGS

def load_traindata():
    file = open(FLAGS.datapath, 'rb')
    Train_dataset, Train_label, test_dataset, test_label, vaild_dataset, vaild_label= cPickle.load(file)
    return Train_dataset, Train_label, test_dataset, test_label, vaild_dataset, vaild_label

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
# for i in range(num_labels):
#      labels_one_hot[i, labels_dense[i]] = 1
  return labels_one_hot

def train(mode='train'):
    trainmodel = cnn.CRNN(mode)
    trainmodel._build_model()
    global_step = tf.Variable(0, trainable=False)
    acc_best = 0
    emot_name = ['hap', 'ang', 'sad', 'neu']

    traindata, trainlabel1, testdata, testlabel1, vailddataset, vaildlabel= load_traindata()
    trainlabel = dense_to_one_hot(trainlabel1, 4)
    testlabel = dense_to_one_hot(testlabel1, 4)
    vaildlabel = dense_to_one_hot(vaildlabel, 4)
    train_size = traindata.shape[0]
    test_size = testlabel.shape[0]

    with tf.variable_scope('cross_entrgy'):
        cross_entrgy = tf.nn.softmax_cross_entropy_with_logits(labels=trainmodel.inputs_label, logits=trainmodel.logits)
        loss = tf.reduce_mean(cross_entrgy)

    with tf.variable_scope('accuracy'):
        accuracy_per = tf.equal(tf.argmax(trainmodel.logits, 1), tf.argmax(trainmodel.inputs_label, 1))
        accuracy = tf.reduce_mean(tf.cast(accuracy_per, tf.float32))

    with tf.variable_scope('train_step'):
        lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                        global_step,
                                        train_size/FLAGS.train_batch_size,
                                        FLAGS.learning_rate_decay_rate,
                                        staircase=True)
        train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
        saver = tf.train.Saver()
        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            for i in range(FLAGS.epoch_num):
                start = (i*FLAGS.train_batch_size)%train_size
                end = min(start+FLAGS.train_batch_size, train_size)
                _, los, step, acc = sess.run([train_step, loss, global_step, accuracy],
                                              feed_dict={trainmodel.inputs:traindata[start:end, :, :, :],
                                                         trainmodel.inputs_label:trainlabel[start:end, :], trainmodel.keep_prob:FLAGS.train_dropout})
                if(i%10==0):
                    acc_vaild = sess.run(accuracy, 
                        feed_dict={trainmodel.inputs:vailddataset, trainmodel.inputs_label:vaildlabel, trainmodel.keep_prob:FLAGS.test_dropout})
                    if(i>9*FLAGS.epoch_num//10):
                        if(acc_best<acc_vaild):
                            acc_best = acc_vaild
                            saver.save(sess, '.model/model.ckpt')
                    print("After %d training step(s), loss on training batch is %.2f, train_acc is %.3f, vaild_acc is %.3f, best_acc is %.3f" % (
                    step, los, acc, acc_vaild, acc_best))
            print('test accuracy=%.3f' % (sess.run(accuracy, 
                                            feed_dict={trainmodel.inputs:testdata, trainmodel.inputs_label:testlabel, trainmodel.keep_prob:FLAGS.test_dropout})))
            saver.restore(sess, '.model/model.ckpt')
            print('accuracy |    |%.4f' % (sess.run(accuracy, 
                                            feed_dict={trainmodel.inputs:testdata, trainmodel.inputs_label:testlabel, trainmodel.keep_prob:FLAGS.test_dropout})))
            for i in range(4):
                print('accuracy |%s |%.3f' % (emot_name[i], sess.run(accuracy, 
                                                feed_dict={trainmodel.inputs:testdata[i*test_size//4:(i+1)*test_size//4], 
                                                trainmodel.inputs_label:testlabel[i*test_size//4:(i+1)*test_size//4], trainmodel.keep_prob:FLAGS.test_dropout})))
            #print('best accuracy is %.3f' % acc_best)
#                    saver.save(
#                        sess, os.path.join(FLAGS.savepath, FLAGS.model_name), global_step=global_step)





if __name__ == '__main__':
#    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train()