'''
Created on 11.09.2017

@author: Jason
'''

import tensorflow as tf
import numpy as np
import ANNdetection_train as nntrain
import os, argparse

def nn_run():
    
    run_id = args.runid
    tf.reset_default_graph()
    
    snr = args.snr if args.snr else 5
    test_set, labels = data_prep(snr=snr) 
    
    #standardize
    test_set[:,:-1] = (test_set[:,:-1] - np.mean(test_set[:,:-1]))/np.std(test_set[:,:-1])
    
    number_classes = len(labels)
    
    tf_x = tf.placeholder(tf.float32, [None, test_set.shape[1]-1])     # gradient strength
    tf_y = tf.placeholder(tf.float32)     # diffusion signal for z-direction

# neural network layers
    neurons_no = args.neuron_no
    activation_function = tf.nn.tanh
    l1 = tf.layers.dense(tf_x, neurons_no, activation_function)          # hidden layer
    l2 = tf.layers.dense(l1, neurons_no, activation_function)          # hidden layer
    output = tf.layers.dense(l2, number_classes, tf.nn.softmax)                     # output layer

    with tf.name_scope("loss"):
        cross_entropy = tf.losses.softmax_cross_entropy(tf_y, output)
        tf.summary.scalar('loss', cross_entropy)
    
    saver = tf.train.Saver()
    
    def nn_eval():
        acc_writer = np.empty(shape=[1,5])   
        with tf.Session() as sess:
            saver.restore(sess, "checkpoints_class/" + run_id) 
            
            lb_ind = labels
            np.random.shuffle(lb_ind)
            for label in lb_ind:
                # find features belonging to label
                test_features = test_set[(test_set[:,int(test_set.shape[1]-1)]==label)][:,:-1]
                # build one hot vector for label
                test_label = tf.one_hot(indices=label, depth=number_classes)
                test_label = np.reshape(test_label.eval(), [1,-1])
                tl_temp = test_label
                for r in range(test_features.shape[0] - 1):
                    tl_temp = np.vstack((tl_temp, test_label))
                test_label = tl_temp
                if(not test_features.size==0): #for testing purposes to skip labels which have no data
                    pred = sess.run(output, {tf_x: test_features, tf_y: test_label})
                    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(test_label, 1))
                    accuracy = sess.run(tf.reduce_mean(tf.cast(correct, 'float')))
                    print(accuracy, "Label: ", label)
                    # for gathering evaluation results (acc_eval)
                    accuracy = np.reshape(accuracy, [1,-1])
                    acc_writer[0][label] = accuracy
            # print overall accuracy
            onehot_all = tf.one_hot(indices=test_set[:,-1], depth=number_classes).eval()
            all_pred = sess.run(output, {tf_x: test_set[:,:-1], tf_y: onehot_all})
            correct_all = tf.equal(tf.argmax(all_pred, 1), tf.argmax(onehot_all, 1))
            accuracy_all = sess.run(tf.reduce_mean(tf.cast(correct_all, 'float')))
            print(accuracy_all, "All with noise", str(snr))    
            # for gathering evaluation results (acc_eval)
            accuracy_all = np.reshape(accuracy_all, [1,-1])
            fstats = os.path.join(os.getcwd() + os.sep, str(snr)+'acc_performance'+os.path.basename(args.schemepath)+'.txt')
            acc_writer[0][4] = accuracy_all
            if(os.path.isfile(fstats)):
                with open(fstats,'ab') as f2:
                    np.savetxt(f2, acc_writer, fmt='%10.6f')       
            else:
                np.savetxt(fstats, acc_writer, fmt='%10.6f')
    def nn_detect():
        with tf.Session() as sess:
            saver.restore(sess, "checkpoints_class/" + run_id) 
            test_features = test_set[:,:-1]
            pred = sess.run(output, {tf_x: test_features})
            pred = sess.run(tf.argmax(pred, 1))
            print(pred)
            
    if(args.snr):
        nn_eval()
    else:
        nn_detect()

def data_prep(snr=10):
    grand_table, labels = nntrain.data_prep(args.schemepath, args.bfloatpath)  
    #add noise
    snr = snr
    grand_table[:,:-1] = np.random.normal(grand_table[:,:-1], 1/snr)
    return grand_table, labels
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("runid")
    parser.add_argument("neuron_no", type=int)
    parser.add_argument("schemepath")
    parser.add_argument("bfloatpath")
    parser.add_argument("--snr", type=float)
    args = parser.parse_args()
    nn_run()