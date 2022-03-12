# Object Detection With The TensorFlow-TensorRT (TF-TRT) Backend In Python using YOLOv3 and YOLOv3-Tiny 

**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
- [Additional resources](#additional-resources)
- [Changelog](#changelog)
- [Known issues](#known-issues)
- [Details](#Details)

## Description
To do benchmarks and detection using the optimisations of TensorRT, this script was build by following NVIDIA's link **https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html**, this is an approach to use TensorRT directly by using the TF-TRT Python API. Both of the frozen graph of YOLOv3 and YOLOv3-Tiny was generated with the github repository of mystic123 **https://github.com/mystic123/tensorflow-yolo-v3**, the main code is [trt_yolov3.py], this is an example about how to use it to implement YOLOv3 TF-TRT using an input image dog.jpg with an input resolution of HxW = 416x416, FP16 precision mode and a batch size of 1. 

`$ python3 trt_yolov3.py --image dog.jpg --model yolov3 --precision FP16 --batch 1` 

CLI arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT, --image INPUT
                        Set the name of the input image
  -m MODEL, --model MODEL
                        Set the name of the model you want to use, <<yolov3>>
                        to use YOLOv3 or <<yolov3-tiny>> to use YOLOv3-Tiny
  -p PRECISION, --precision PRECISION
                        Set the precision mode [FP32, FP16 or INT8]
  -b BATCH, --batch BATCH
                        Set The size of the batch



## How does this sample work?

First, you have to load both of the TensorFlow model and the input image and pre-process this latter, then before the conversion, the builder will be configured by the argments set in the command line by the user, the method convert() will convert the tensorflow model into TF-TRT model and than post-process the output using the Non-max suppression included in the script [utils.py] which was collected from the github of **mystic123**, in the end the image will be saved with bounding boxes detections and get the result of the benchmarks in the terminal.

**Note:** This sample is not supported on Ubuntu 14.04 and older.

## Prerequisites

Download YOLOv3 and YOLOv3-Tiny which was generated using mystic123 github https://github.com/mystic123/tensorflow-yolo-v3 with the following commands :
	
`$ wget "https://drive.google.com/uc?export=download&id=1euQo121u5x3OPvdYNZpheqiub5bzJbpy" -O frozen_darknet_yolov3_tiny_model.pb`
	
`$ wget "https://drive.google.com/uc?export=download&id=1t-ZygeJpTwzZ3i0Q374VZo5Omm9U2Xz_" -O frozen_darknet_yolov3_model.pb`

	

	
If you are using an NVIDIA Jetson board, to install Tensorflow 1.15 following this command line :
	
`$ sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 'tensorflow<2'`

If you are using a personel computer with a linux distribution like Ubuntu, you can install Tensorflow 1.15 :
1.  Using pip :
`$ sudo pip3 install tensorflow==1.15`
2.  Using anaconda with UI, by launching anaconda with the command line :
`$ anaconda-navigator`
3.  Using anaconda with the CLI :
`$ sudo conda install tensorflow==1.15`

## Running the sample

1.  Get your custom frozen graph or directly run the script trt_yolov3.py to use YOLOv3 or YOLOv3-Tiny.
	`

2.  Build TensorRTEngineOp in the graph using the method convert.
	
	`python3 trt_yolov3.py --image dog.jpg --model yolov3 --precision FP16 --batch 1`
	
	2021-09-16 17:33:15.159503: I tensorflow/compiler/tf2tensorrt/kernels/trt_engine_op.cc:733] Building a new TensorRT engine for import/TRTEngineOp_0 input shapes: [[1,416,416,3]]
	2021-09-16 17:33:15.321329: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libnvinfer.so.7
	2021-09-16 17:33:16.390368: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libnvinfer_plugin.so.7
	2021-09-16 17:37:35.535484: I tensorflow/compiler/tf2tensorrt/kernels/trt_engine_op.cc:733] Building a new TensorRT engine for import/detector/yolo-v3/TRTEngineOp_3 input shapes: [[1,26,26,256], [1,512,26,26]]
	2021-09-16 17:37:44.832719: I tensorflow/compiler/tf2tensorrt/kernels/trt_engine_op.cc:733] Building a new TensorRT engine for import/detector/yolo-v3/TRTEngineOp_4 input shapes: [[1,52,52,128], [1,256,52,52]]
	2021-09-16 17:37:51.874350: I tensorflow/compiler/tf2tensorrt/kernels/trt_engine_op.cc:733] Building a new TensorRT engine for import/detector/yolo-v3/TRTEngineOp_5 input shapes: [[1,507,2], [1,507,1], [1,507,2], [1,507,80], [1,2028,2], [1,2028,1], [1,2028,2], [1,2028,80], [1,8112,2], [1,8112,1], [1,8112,2], [1,8112,80]]
	2021-09-16 17:38:23.928248: I tensorflow/compiler/tf2tensorrt/kernels/trt_engine_op.cc:733] Building a new TensorRT engine for import/TRTEngineOp_2 input shapes: [[1,10647,1], [1,10647,1]]
	2021-09-16 17:38:36.383982: I tensorflow/compiler/tf2tensorrt/kernels/trt_engine_op.cc:733] Building a new TensorRT engine for import/TRTEngineOp_1 input shapes: [[1,10647,1], [1,10647,1]]
	Latency = 19,41 ms | FPS = 51,53
	Saved image with bounding boxes of detected objects to dog_yolov3_FP16_bs1.png.
	



# Additional resources

The following resources provide a deeper understanding about the model used in this sample, as well as the dataset it was trained on:

**Model**
- [YOLOv3](https://github.com/mystic123/tensorflow-yolo-v3)
- [YOLOv3-Tiny](https://github.com/mystic123/tensorflow-yolo-v3)

**Dataset**
- [COCO dataset](http://cocodataset.org/#home)

**Documentation**
- [TensorFlow-TensorRT](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)

# Changelog
September 2021 - Stage compet - Evaluation des solutions de Deep Learning.

# Known issues

1. Using YOLOv3 for benchmarks and detection take a lot of time to convert the model from tensorflow to TF-TRT and need to allocate a lot of RAM for the GPU.
2. The Non-max suppression doesn't work for YOLOv3-Tiny, so we have to adjust some parameters like the threshold of iou and nms threshold, anyway we can still do benchmarks to evaluate the model and the hardware using TensorRT with API.


# Details

**Scripts**
- [utils.py]  : Include the preprocess and the post-process (Non-maximum Suppression) for both of YOLOv3 et YOLOv3-Tiny.
- [tf-yolo.py] : An additional script to benchmarks YOLOv3 and YOLOv3-Tiny without having a prediction in output only with using TensorFlow (Optionnal and without optimizations).
- [trt_yolov3.py] : The main script of the project which allows us to do benchmarks with of YOLOv3 and YOLOv3-Tiny according to the parameters set in the CLI as precision and batch.

**Folder/Files and scripts**
- [frozen_darknet_yolov3_model.pb] : YOLOv3 frozen model generated with the github of **mystic123**.
- [tiny_yolo/frozen_darknet_yolov3_model.pb] : YOLOv3-Tiny frozen model generated with the github of **mystic123**
- [yolo_v3-coco.inference_only.frozen.pb] : YOLOv3 withoud post-processing used only to benchmarks YOLOv3.
- [dog.jpg] : Test image
- [dog_yolov3_FP16_bs1.png] : Output image with detections using YOLOv3 TF-TRT FP16
- [dog_yolov3_FP32_bs1.png] : Output image with detections using YOLOv3 TF-TRT FP32
	
	



