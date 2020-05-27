import tensorflow as tf
import numpy as np
from PIL import Image
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./dataset/", one_hot=True)

def create_sample(n):
    if not os.path.exists("./dataset"): os.mkdir("./dataset")
    if not os.path.exists("./dataset/mnist"): os.mkdir("./dataset/mnist")

    with open("./dataset/mnist/lables.txt", 'w+') as f:
        for i in range(n):
            img = mnist.test.images[i] * 255
            label = mnist.test.labels[i] * 255
            img = np.resize(img, (28, 28))
            im = Image.fromarray(img)
            if im.mode != 'RGB':
                im = im.convert('RGB')
            im.save('dataset/mnist/'+str("{:03}".format(i))+ '.jpeg')
            f.write('dataset/mnist/'+str("{:03}".format(i))+ '.jpeg\t{}\n'.format(np.argmax(label)))


        

def run(filename = "random"):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if filename == "random":
        # Test model on random input data.
        input_shape = input_details[0]['shape']
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    else:
        input_data = Image.open(filename)
        input_data = input_data.convert('L')
        input_data = np.array(input_data, dtype='float32')
        input_data = input_data/255
        input_data = np.resize(input_data, (1, 28*28))

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    if filename == "random":
        print("Test : {}".foramt(output_data[0]))
    else:
        gt = {}
        with open("./dataset/mnist/lables.txt", 'r') as f:
            while(True):
                line = f.readline()
                if len(line)==0: break
                line = line.split("\t")
                gt[line[0]] = line[1]
        
        print("GT : ", gt[filename][0], ", Test : ", str(output_data[0]))

if __name__=="__main__":
    create_sample(10)
    for i in range(10):
        run('dataset/mnist/'+str("{:03}".format(i))+ '.jpeg')
