'''
Created on 11.09.2017

@author: Jason
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os, re, argparse
from math import inf

def nn_train():  
    scale = args.scale if (args.scale) else 1.0
    x, y, x_val, y_val = data_prep(scale=scale)
    iterations = args.iterations if (args.iterations) else 7001
    y = np.reshape(y,(-1,1))
    y_val = np.reshape(y_val,(-1,1))   
    #tensorshapes
    tf_x = tf.placeholder(tf.float32, [None, x.shape[1]])
    tf_y = tf.placeholder(tf.float32, [None, y.shape[1]])      
    # neural network layers
    neurons_no = args.neuron_no
    activation_function = tf.nn.relu
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    l1 = tf.layers.dense(tf_x, neurons_no, activation_function, kernel_regularizer=regularizer)          # hidden layer
    l2 = tf.layers.dense(l1, neurons_no, activation_function, kernel_regularizer=regularizer)          # hidden layer
    output = tf.layers.dense(l2, 1, activation_function)                     # output layer       
    with tf.name_scope("loss"):
        loss = tf.losses.mean_squared_error(tf_y, output)
        tf.summary.scalar('loss', loss) 
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = args.learningrate if (args.learningrate) else 0.001
    decay = 0.8
    per_step = 5000
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, per_step, decay, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)    
    val = np.column_stack((x_val, y_val))
    run_id = str(starter_learning_rate) + "_" + str(scale) + "_" + str(neurons_no) + "_" + str(iterations) + dt.datetime.now().strftime('%Y%m%d%H%M')
    np.save("vals" + run_id + ".npy", val)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        saver = tf.train.Saver(tf.trainable_variables())     
        if(args.plotoff):
            plt.ion()
        learn_curves=[[inf,inf]]
        x_rows = x.shape[0]
        batch_size = x_rows/10             
        for step in range(iterations):
            # train and net output
            for batch in np.arange(batch_size, x_rows+1, batch_size):        
                x_batch = x[int(batch - batch_size):int(batch),:]
                y_batch = y[int(batch - batch_size):int(batch),:]
                _, l, pred = sess.run([train_op, loss, output], {tf_x: x_batch, tf_y: y_batch})                    
            if step % int(iterations/20) == 0:
                # plot and show learning process
                if(args.plotoff):
                    plt.cla()
                    plt.xlabel('Gradient strength (x10^-1)')
                    plt.ylabel('Normalized diffusion signal')
                    plt.scatter(x_batch[:,3], y_batch)
                    plt.scatter(x_batch[:,3], pred, s=25, c='r', marker='s')
                    plt.pause(0.01)
                vl = sess.run(loss, {tf_x: x_val, tf_y: y_val})
                learn_curves=np.vstack((learn_curves, [[l, vl]]))
                print(l, vl)                
            # re-shuffle data each run
            x, y = shuffle_data(x, y)
            y = np.reshape(y,(-1,1))                
        if(args.plotoff):
            plt.ioff()
            plt.show()          
        #save network             
        saver.save(sess, "checkpoints/" + run_id)
        np.savetxt("lcurves_" + run_id + ".txt", learn_curves, fmt='%10.6f')        
        #save learning details            
        l = sess.run(loss, {tf_x: x_batch, tf_y: y_batch})
        vl = sess.run(loss, {tf_x: x_val, tf_y: y_val})
        print(l, vl)
        run_stats = [[vl, l, starter_learning_rate, iterations, neurons_no, scale]]
        fstats = os.path.join(os.getcwd() + os.sep,'train_performance.txt')
        if(os.path.isfile(fstats)):
            with open(fstats,'ab') as f2:
                np.savetxt(f2, run_stats, fmt='%10.6f')       
        else:
            np.savetxt(fstats, run_stats, fmt='%10.6f', header='Loss(CV)\tLoss(Train)\tlearn_rate(start)\titerations\tneuron_no', comments='')    
def data_prep(scale=1.0, training_set=True):
    #prepare x-values´
    # load gradients
    scheme = np.loadtxt(args.schemepath, skiprows=1)     
    grad_dir = scheme[:,0:3]*5  #multiplier used for scaling
    grad_str = np.absolute(scheme[:,3])*10  #multiplier used for scaling
    # form x array
    x_single = np.column_stack((grad_dir, grad_str))   
    #prepare y-values
    bfloat_list = []
    bfloatpath = args.bfloatpath
    print(bfloatpath + " is used to locate .Bfloat files\n")
    for file in os.listdir(bfloatpath):
        if file.endswith(".Bfloat"):
            bfloat_list.append(bfloatpath + file)            
    if(not training_set):
        print("Found .Bfloat files: ")
        for bfloat in bfloat_list:
            print(bfloat)    
    x = []
    y = []  
    for bfloat in bfloat_list:
        filename = bfloat
        data = np.fromfile(filename, dtype='>f')
        y_agg = np.absolute(data)/10000     #divided by number of walkers (10000)
        y.extend(y_agg)       
        #gather geometry stats
        #IMPORTANT: filename MUST be in the format generated by the MATLAB script with stats in file name
        gstats = os.path.basename(os.path.splitext(os.path.splitext(filename)[0])[0])
        gstats = re.findall(r"[-+]?\d*\.\d+|\d+",gstats)
        _ = np.full((x_single.shape[0], 1), gstats.pop(0)) #rad
        amp = np.full((x_single.shape[0], 1), np.float32(gstats.pop(0)))
        _ = np.full((x_single.shape[0], 1), gstats.pop(0)) #g
        volfract = np.full((x_single.shape[0], 1), np.float32(gstats.pop(0)))         
        x_temp = np.column_stack((x_single, amp, volfract))
        x = np.reshape(x, (-1, x_temp.shape[1]))
        x = np.vstack((x, x_temp)) 
    #randomize data order  
    y = np.reshape(y,(-1,1))  
    x, y = shuffle_data(x, y)
    #form final arrays
    rows = int(y.shape[0]*scale)
    set_sep = int(np.floor(rows * (4/5))) 
    x_train = x[0:set_sep,:]
    y_train = y[0:set_sep]    
    x_val = x[set_sep:rows,:]
    y_val = y[set_sep:rows]       
    #if this happens, signals are not based on correct grad scheme
    if(x.shape[0]!=y.shape[0]):
        print("input mismatch")
    return x_train, y_train, x_val, y_val
def shuffle_data(x, y):
    shuffle_stack = np.column_stack((x, y))
    np.random.shuffle(shuffle_stack)    
    shuffled_x = shuffle_stack[:, 0:shuffle_stack.shape[1]-1]
    shuffled_y = shuffle_stack[:,-1]  
    return shuffled_x, shuffled_y
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("schemepath")
    parser.add_argument("bfloatpath")
    parser.add_argument("neuron_no", type=int)
    parser.add_argument("-i", "--iterations", type=int)
    parser.add_argument("-l", "--learningrate", type=float)
    parser.add_argument("-s", "--scale", type=float)
    parser.add_argument("-pf", "--plotoff", action="store_false")
    args = parser.parse_args()
    nn_train()