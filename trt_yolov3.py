"""
TensorFlow to TensorRT converter with TensorFlow v1.15
Workflow with a fozen graph

Both of YOLOv3 and YOLOv3-Tiny frozen graphs were generated using mystic123 github https://github.com/mystic123/tensorflow-yolo-v3
"""

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import os
import time
import argparse
from utils import non_max_suppression, draw_boxes, letter_box_image
from PIL import Image
import numpy as np


desc = ('The puprose of using this code is to benchmark YOLOv3 and YOLOv3-Tiny using TF-TRT Python API, you can use different arguments in the command'
        'line as the input batch size, the precision mode etc.')
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-i', '--input', '--image', help="Set the name of the input image", type=str)
parser.add_argument('-m', '--model', help="Set the name of the model you want to use, <<yolov3>> to use YOLOv3 or <<yolov3-tiny>> to use YOLOv3-Tiny", type=str)
parser.add_argument('-p', '--precision', help="Set the precision mode [FP32, FP16 or INT8]", type=str)
parser.add_argument('-b', '--batch', help="Set The size of the batch", type=int)
parser.add_argument('-s', '--saveEngine', help="Save the TF-TRT frozen graph", type=str)
parser.add_argument('-l', '--loadEngine', help="Load the TF-TRT frozen graph required for the inference", type=str)
parser.add_argument('-c', '--countTrtOp', help="Count how many TRTEngineOps there is in the TF-TRT graph", action='store_true')
parser.add_argument('-a', '--standAlonePlan', help="Generate A Stand-Alone TensorRT Plan", action='store_true')
parser.add_argument('-e', '--placeholders', help="Get all placeholders names", action='store_true')
parser.add_argument('-t', '--tensorMap', help="Display the Tensor's names map", action='store_true')
parser.add_argument('-v', '--visualizeGraph', help="Get the URL to view the graph on TensorBoard", action='store_true')

args = parser.parse_args()

# Inference iterations
nbr_frames = 100
# Input batch size
if args.batch is not None:
    BATCH_SIZE = args.batch
else:
    BATCH_SIZE = 1
# Input image
input_img = args.input
# IoU threshold
iou_threshold = 0.4
# Confidence threshold
conf_threshold = 0.5

# Output nodes
output_nodes = ['output_boxes']
# Input tensor name
data_in_str = "import/inputs:0"
# Output tensor name
output_node_str = "import/output_boxes:0"

# Model Selection
if args.loadEngine is None:
    model = args.model
    if model == 'yolov3-tiny':
        path_frozen_graph = "tiny_yolo/frozen_darknet_yolov3_tiny_model.pb"
    elif model == 'yolov3':
        path_frozen_graph = "frozen_darknet_yolov3_model.pb"

    else:
        print("""ERROR: Please verify that the parameters set in the CLI respect the arguments mentionned in the description and try again.
        For more information use the commande : python3 trt_yolov3.py -h""")
        exit(0)

# Choose the precision mode
if args.loadEngine is None:
    if args.precision == 'FP32':
        precision = trt.TrtPrecisionMode.FP32
    elif args.precision == 'FP16':
        precision = trt.TrtPrecisionMode.FP16
    elif args.precision == 'INT8':
        precision = trt.TrtPrecisionMode.INT8
    else:
        print("""ERROR: Please verify that the parameters set in the CLI respect the arguments mentionned in the description and try again.
        For more information run the commande : python3 trt_yolov3.py -h""")
        exit(0)

def load_graph(frozen_graph):
    with tf.gfile.GFile(frozen_graph, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

def preprocess(img, batch_size):
    img = img.reshape(1, 416, 416, 3)
    img = img.repeat(batch_size, axis=0)
    return img

def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names

def save_engine(engine,graph):
    with tf.gfile.GFile(engine, "wb") as f:
        f.write(graph.SerializeToString())

def main(argv=None):

    # Load COCO classes names
    classes = load_coco_names('coco.names')
    # Load and resize the input image
    img = Image.open(input_img)
    img_resized = letter_box_image(img, 416, 416, 128)
    img_resized = img_resized.astype(np.float32)
    # Pre-process the input image
    input_image = preprocess(img_resized, BATCH_SIZE)

    with tf.compat.v1.Session() as sess:

        if args.loadEngine is not None:
            # Load the TF-TRT frozen graph
            trt_graph = load_graph(args.loadEngine)
        else:
            # Load frozen graph:
            frozen_graph = load_graph(path_frozen_graph)

            # Create a TensorRT inference graph from the frozen graph
            converter = trt.TrtGraphConverter(
                input_graph_def=frozen_graph,
                nodes_blacklist=output_nodes, # output nodes
                precision_mode=precision,# FP32, FP16 or INT8 precision mode
                max_batch_size=BATCH_SIZE, # The batch size is by default to 1
                is_dynamic_op=True, # TRTEngines are built until we run the inference (if we set is_dynamic_op as True they TRTEngines will be built since the conversion)
                minimum_segment_size=3, # by default to 3
                use_calibration = True,
            maximum_cached_engines=1) # by default to 1

            # Convert from TensorFlow to TensorFlow-TensorRT
            print("\n")
            print("Starting the conversion from TensorFlow to TensorRT... \n")
            t0 = time.time()
            trt_graph = converter.convert()
            t1 = time.time()
            print("\n")
            print("End of the conversion, please note that if you use the dynamic mode your model will not be optimized by TensorRT until you run the inference. \n")
            print("Conversion TF-TRT = {:.2f} s \n".format(t1 - t0))

        # Count how many TRTEngineOps in trt_graph
        if args.countTrtOp:
            trt_engine_ops = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
            print("Number of TRTEngineOp in trt_graph is : ", trt_engine_ops)
            all_ops = len([1 for n in trt_graph.node])
            print("Number of all_ops in in trt_graph:", all_ops)

        # Generate A Stand-Alone TensorRT Plan
        if args.standAlonePlan:
            print("****************************** Generate A Stand-Alone TensorRT Plan **********")
            for n in trt_graph.node:
                if n.op == "TRTEngineOp":
                    print("Node: %s, %s" % (n.op, n.name.replace("/", "_")))
                    with tf.gfile.GFile("%s.plan" % (n.name.replace("/", "_")), 'wb') as f:
                        f.write(n.attr["serialized_segment"].s)
                else:
                    print("Exclude Node: %s, %s" % (n.op, n.name.replace("/", "_")))

        # Save the TF-TRT frozen graph
        if args.saveEngine is not None:
            save_path = args.saveEngine
            save_engine(save_path, trt_graph)
            print("TF-TRT frozen graph saved in {} \n".format(save_path))

        # Import a serialized TensorFlow model
        tf.import_graph_def(trt_graph)
        # Get the serialized graph which was converted with TF-TRT
        trt_graph = tf.get_default_graph()

        # Get all placeholders
        if args.placeholders:
            all_placeholders = [placeholder for op in trt_graph.get_operations() if op.type == 'Placeholder' for placeholder in op.values()]
            print("***************** placeholders *****************")
            print(all_placeholders)

        # Display the Tensor's names map
        if args.tensorMap:
            all_tensors = [tensor for op in trt_graph.get_operations() for tensor in op.values()]
            print("***************** Tensors *****************")
            print(all_tensors)

        # Get the input/output nodes
        input_tensor = trt_graph.get_tensor_by_name(data_in_str)
        output_tensor = trt_graph.get_tensor_by_name(output_node_str)

        # Warmup
        detected_boxes = sess.run(output_tensor, feed_dict={input_tensor: input_image})

        # Benchmark
        start_time = time.time()
        for i in range(nbr_frames):
            sess.run(output_tensor, feed_dict={input_tensor: input_image})
        latency = ((time.time() - start_time))/nbr_frames
        fps = BATCH_SIZE / latency
        print('\n')
        print("Latency = {:.2f} ms | FPS = {:.2f} \n".format(latency*1000,fps))

        # Post-process
        filtered_boxes = non_max_suppression(detected_boxes,
                                             confidence_threshold=conf_threshold,
                                             iou_threshold=iou_threshold)
        # Draw bounding boxes
        draw_boxes(filtered_boxes, img, classes, (416,416), True)

        # Save image with detections bounding boxes
        if args.loadEngine is not None:
            output_image_path = args.input.replace(".jpg","_") + args.loadEngine.replace(".pb","") + '.png'
        else:
            output_image_path = args.input.replace(".jpg","_") + args.model + '_' + str(args.precision)+ '_bs' + str(args.batch) + '.png'
        img.save(output_image_path)
        print('Saved image with bounding boxes of detected objects to {}'.format(output_image_path))

        # Launch and display the graph on TensorBoard
        if args.visualizeGraph:
            tf.summary.FileWriter ('./tensorboard_events', sess.graph)
            print("******************Tensorboard written file DONE********************")
            os.system("tensorboard --logdir ./tensorboard_events")

if __name__ == '__main__':
    tf.app.run()


