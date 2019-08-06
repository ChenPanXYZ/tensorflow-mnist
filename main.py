"""
组员名单：陈攀（计算机1704）
参考资料：http://tensorfly.cn/tfdoc/tutorials/mnist_pros.html
         https://blog.csdn.net/xiaodongxiexie/article/details/67009112
编译环境：Python 3.6, Windows 10，Tensorflow
性能测量指标：
训练集：MNIST
测试集的正确率：95.96%
训练时候的正确率：
训练0次时,训练集的正确率 = 3.33333%
训练50次时,训练集的正确率 = 56.6667%
训练100次时,训练集的正确率 = 83.3333%
训练150次时,训练集的正确率 = 93.3333%
训练200次时,训练集的正确率 = 86.6667%
训练250次时,训练集的正确率 = 90%
训练300次时,训练集的正确率 = 93.3333%
训练350次时,训练集的正确率 = 96.6667%
训练400次时,训练集的正确率 = 93.3333%
训练450次时,训练集的正确率 = 93.3333%
训练500次时,训练集的正确率 = 86.6667%
训练550次时,训练集的正确率 = 96.6667%
训练600次时,训练集的正确率 = 96.6667%
训练650次时,训练集的正确率 = 96.6667%
训练700次时,训练集的正确率 = 90%
训练750次时,训练集的正确率 = 100%
训练800次时,训练集的正确率 = 100%
训练850次时,训练集的正确率 = 93.3333%
训练900次时,训练集的正确率 = 96.6667%
训练950次时,训练集的正确率 = 100%
备注：
1.因为电脑配置不高，所以对模型进行训练的时候只循环训练1000次。
"""
TrainTimes=1000 #训练次数为1000次
import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess=tf.InteractiveSession()

#卷积神经网络
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME') 

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])
x_img=tf.reshape(x,[-1,28,28,1])

w_conv1=tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
b_conv1=tf.Variable(tf.constant(0.1,shape=[32]))
h_conv1=tf.nn.relu(conv2d(x_img,w_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)

w_conv2=tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
b_conv2=tf.Variable(tf.constant(0.1,shape=[64]))
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

w_fc1=tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
b_fc1=tf.Variable(tf.constant(0.1,shape=[1024]))
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

w_fc2=tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
b_fc2=tf.Variable(tf.constant(0.1,shape=[10]))
y_out=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

loss=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_out),reduction_indices=[1]))

train_step=tf.train.GradientDescentOptimizer(1e-2).minimize(loss)#优化器速率设置为0.01。

correct_prediction=tf.equal(tf.argmax(y_out,1),tf.argmax(y_,1))#预测准确的次数，相当于“TP+TF”。
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))*100

#训练
tf.global_variables_initializer().run()
for i in range(TrainTimes):
    batch=mnist.train.next_batch(30)
    if i%50==0:#因为训练次数较少，所以训练50次就显示一次Accuracy，更加“连续”地看出Accuracy的变化。
        train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1})
        print ("训练%d次时,训练集的正确率 = %g%%"%((i,train_accuracy)))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.4})#因为训练时候提供的数据比较少，因此将Dropout的比例提高到0.6

#测试
print ("测试集的正确率 = %g%%"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1}))