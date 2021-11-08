import tensorflow as tf
hello = tf.constant("안녕 텐서플로우")
sess = tf.Session()
print(sess.run(hello))
a=tf.constant(7)
b=tf.constant(56)
print(sess.run(a+b))
#print(sess.run(hello).decode("UTF-8"))
