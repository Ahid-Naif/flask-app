from flask import Flask
from flask import request
import numpy as np
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import json
import os

# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
module_url = ROOT_DIR+"/module"

# Create graph and finalize (optional but recommended).
g = tf.Graph()
with g.as_default():
    text_input = tf.placeholder(dtype=tf.string, shape=[None])
    embed = hub.load(module_url)
    my_result = embed(text_input)
    init_op = tf.group(
        [tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

# Create session and initialize.
session = tf.Session(graph=g)
session.run(init_op)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
  
# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return 'Hello World'

@app.route("/similar", methods=['POST'])
def similar():
    data = json.loads(request.data)

    my_result_out = session.run(
        my_result, feed_dict={text_input: [data["a"], data["b"]]})
    corr = np.inner(my_result_out, my_result_out)
    result = float("{:.2f}".format(corr[0][1]))*100

    result_file = open("result.txt", "w")
    result_file.write(str(result))
    result_file.close()

    return json.dumps({"value": result}, cls=NumpyEncoder)
  
@app.route("/getResult", methods=['get'])
def getResult():
    result = open("result.txt", "r")
    result_data = result.read()
    result.close()

    result_file = open("result.txt", "w")
    result_file.write("")
    result_file.close()

    return result_data

# main driver function
if __name__ == '__main__':
    app.run()