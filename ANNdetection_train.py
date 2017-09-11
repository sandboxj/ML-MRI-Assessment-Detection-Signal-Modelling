'''
Created on 11.09.2017

@author: Jason
'''
import tensorflow as tf
import numpy as np
import datetime as dt
import os, re, argparse

def nn_train():    
    iterations = args.iterations if (args.iterations) else 101
    starter_learning_rate = args.learningrate if args.learningrate else 0.001
    
    train_set, labels = data_prep(args.schemepath, args.bfloatpath) 
    
    #standardize
    train_set[:,:-1] = (train_set[:,:-1] - np.mean(train_set[:,:-1]))/np.std(train_set[:,:-1])
    
    number_classes = len(labels)
    
    tf_x = tf.placeholder(tf.float32, [None, train_set.shape[1]-1])     # gradient strength
    tf_y = tf.placeholder(tf.float32)     # diffusion signal for z-direction

# neural network layers
    neurons_no = args.neuron_no
    activation_function = tf.nn.tanh
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    l1 = tf.layers.dense(tf_x, neurons_no, activation_function, kernel_regularizer=regularizer)          # hidden layer
    l2 = tf.layers.dense(l1, neurons_no, activation_function, kernel_regularizer=regularizer)          # hidden layer
    output = tf.layers.dense(l2, number_classes, tf.nn.softmax)                     # output layer
    
    with tf.name_scope("loss"):
#         loss = tf.losses.huber_loss(tf_y, output)   # compute cost
#         loss = (max_abs_loss(tf_y, output))
        cross_entropy = tf.losses.softmax_cross_entropy(tf_y, output) + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#         cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_y, logits=output))
        tf.summary.scalar('loss', cross_entropy)
        
    global_step = tf.Variable(0, trainable=False)
    decay = 0.8
    per_step = 5000
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, per_step, decay, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy, global_step=global_step)
    
    run_id = "class" + str(starter_learning_rate) + "_" + "_" + str(neurons_no) + "_" + str(iterations) + dt.datetime.now().strftime('%Y%m%d%H%M')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables()) 
        
        lb_ind = labels
        early_stoppage = False
        for step in range(iterations):
            if(early_stoppage):
                break
            else:
                np.random.shuffle(lb_ind)
                for label in lb_ind:
                    np.random.shuffle(train_set)
                    train_features = train_set[(train_set[:,int(train_set.shape[1]-1)]==label)][:,:-1]
                    train_features = np.random.normal(train_features, 1/5)
                    train_label = tf.one_hot(indices=label, depth=number_classes)
                    train_label = np.reshape(train_label.eval(), [1,-1])
                    tl_temp = train_label
                    for r in range(train_features.shape[0] - 1):
                        tl_temp = np.vstack((tl_temp, train_label))
                    train_label = tl_temp
                    if(not train_features.size==0): #for testing purposes to skip labels which have no data
                        _, _, pred = sess.run([train_op, cross_entropy, output], {tf_x: train_features, tf_y: train_label})
    #                     if step % 2 == 0:
                        if step % int(iterations/20) == 0:
                            correct = tf.equal(tf.argmax(pred, 1), tf.argmax(train_label, 1))
                            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    #                         # cross validate with whole set + noise at SNR 5
                            cv_label = tf.one_hot(indices=train_set[:,-1], depth=number_classes).eval()
    #                         cv_label = np.reshape(cv_label.eval(), [1,-1])
                            cv_features = np.random.normal(train_set[:,:-1], 1/5)
#                             cv_features = train_set[:,:-1]
                            cv_pred = sess.run(output, {tf_x: cv_features, tf_y: cv_label})
                            cv_correct = tf.equal(tf.argmax(cv_pred, 1), tf.argmax(cv_label, 1))
                            cv_accuracy = tf.reduce_mean(tf.cast(cv_correct, 'float'))
                            cv_acc_val = sess.run(cv_accuracy)
                            print("Step:", step, sess.run(accuracy), "Label: ", label, "\t CV accuracy: ", cv_acc_val)
                            if(args.stoppage):
                                if(cv_acc_val>0.98):
                                    early_stoppage=True
                                    break
        saver.save(sess, "checkpoints_class/" + run_id)

def data_prep(schemepath, bfloatpath):
    
    # load experiment 
    scheme = np.loadtxt(schemepath, skiprows=1)
    additional_input_values = 1 #label
    grand_table = np.empty(shape=[0,(scheme.shape[0] + additional_input_values)])
    
    #load experiment signals
    bfloat_list = []
    bfloatpath = bfloatpath
    print(bfloatpath + " is used to locate .Bfloat files\n")
    for file in os.listdir(bfloatpath):
        if file.endswith(".Bfloat"):
            bfloat_list.append(bfloatpath + file)
            
    print("Found .Bfloat files: ")
    for bfloat in bfloat_list:
        print(bfloat)   
        data = np.fromfile(bfloat, dtype='>f')
        dsignals = np.absolute(data)
        
        #gather geometry stats
        #IMPORTANT: filename MUST be in the format generated by the MATLAB script with stats in file name
        gstats = os.path.basename(os.path.splitext(os.path.splitext(bfloat)[0])[0])
        gstats = re.findall(r"[-+]?\d*\.\d+|\d+",gstats)
        _ = gstats.pop(0) #rad
        amp = np.float32(gstats.pop(0))
        _ = gstats.pop(0) #g
        volfract = np.float32(gstats.pop(0))
        
        #identify label
        ind, labels = findlabels(amp, volfract)
        
        #shape data in row per bfloat
        dsignals = (np.reshape(dsignals, [1,-1]))
        
        dsignals = np.insert(dsignals, dsignals.shape[1], ind, axis=1)
        grand_table = np.append(grand_table, dsignals, axis=0)
        
#     normalize
    print("Normalize by G=|0|: ", grand_table[0][0])
    grand_table[:,:-1] = grand_table[:,:-1]/grand_table[0][0]
    return grand_table, labels

def findlabels(amp, volfract):
    #make labels
    normal = 0
    beading = 1
    edema = 2
    edema_beading = 3
    beading_thresh = 0.35
    edema_thresh = 0.45
    labels = [normal, beading, edema, edema_beading]   
    
    #identify label
    bead_ind = beading if (amp > beading_thresh) else normal
    edema_ind = edema if (volfract < edema_thresh) else normal
    ind = bead_ind + edema_ind
    
    return ind, labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("schemepath")
    parser.add_argument("bfloatpath")
    parser.add_argument("neuron_no", type=int)
    parser.add_argument("-i", "--iterations", type=int)
    parser.add_argument("-l", "--learningrate", type=float)
    parser.add_argument("-stp", "--stoppage", action="store_false")
    args = parser.parse_args()
    nn_train()