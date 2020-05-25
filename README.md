# convert_ckpt_pb_tflite
Understanding of ckpt, pb and tflite and the conversion among them

## 1. Understanding
### A. ckpt
  * It is heavy to use for inference, because it contains files necessary for training.
  * So it is able to re-training using ckpt.
  * Files
    * model-9900.ckpt.data-00000-of-00001 : Including all variables except the structure of the model. Model can be restored using meta and index file
    * model-9900.ckpt.index : Including the mapping information between data and meta file
    * model-9900.ckpt.meta : Including the graph of model except variables
    * model-9900.ckpt.pb(or .pbtxt) : Including the graph in binary(or text)
      * [what is difference frozen.pb and saved-model.pb?](https://stackoverflow.com/questions/52934795/what-is-difference-frozen-inference-graph-pb-and-saved-model-pb)

### B. pb
  * Binary file including only graph and variables for inference
    
### C. tflite
  * It is the light version of the tensorflow model.
  * It can be applied various method to weight lighting.
  * It is suitable for putting in devices with low computing power, such as mobile devices.

## 2. Conversion
| from \  to    | ckpt  | pb    | tflite |
| ------------- |:-----:| -----:| ------:|
| ckpt          |🍪|[ckpt_to_pb.py(freeze_graph.py)](www.naver.com)|[ckpt2tflite.py](www.naver.com)|
| pb            |[pb2ckpt.py](www.naver.com)|🍦|[pb2tflite.py](www.naver.com)|
| tflite        |[tflite2ckpt.py](www.naver.com)|[tflite2pb.py](www.naver.com)|☕️|

 a. From ckpt to pb (freeze):
```bash
$ python freeze_graph.py \
--input_graph=./model/ckpt/model.pb \
--input_binary=true \
--input_checkpoint=./model/ckpt/model-900 \
--output_graph=./model/pb/frozen_model.pb \
--output_node_names=ArgMax 
```

## 3. Compare the performance

|               | Volume  | Speed    | Performance |
| ------------- |:-----:| -----:| ------:|
| ckpt          |  |       |        |
| pb            |       |       |        |
| tflite        |       |       |      |
