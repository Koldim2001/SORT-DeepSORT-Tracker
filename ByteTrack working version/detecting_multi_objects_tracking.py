import torch
import numpy as np
import cv2
import os
import PIL.Image as pilimg
import matplotlib
import ast
import time

import tensorrt as trt
TRT_LOGGER = trt.Logger()

# Tracker
from byte_tracker_pytorch.byte_tracker_model import BYTETracker as ByteTracker

class Detecting_Multi_Objects_Tracking_Processor():
	def __init__(self, fps, config_dict, device_name):
		
		print('-'*60)
		
		## Configs
		yolov3_config = config_dict['yolov3_config']
		byte_tracker_config = config_dict['byte_tracker_config']
		tensorrt_config = config_dict['tensorrt_config']
		
		# Device
		if torch.cuda.is_available() and device_name.find('cuda') != -1:
			self.device = torch.device(device_name)
			torch.backends.cudnn.deterministic = True
			torch.cuda.set_device(self.device)
		else:
			self.device = torch.device('cpu')
		print('Device:', self.device)

		# ByteTrack param
		first_track_thresh = byte_tracker_config.getfloat('threshold', 'first_track_thresh')
		second_track_thresh = byte_tracker_config.getfloat('threshold', 'second_track_thresh')
		match_thresh = byte_tracker_config.getfloat('threshold', 'match_thresh')
		track_buffer = byte_tracker_config.getint('byte_tracker', 'track_buffer')
		
		# Yolov3 param
		yolov3_channels = yolov3_config.getint('yolov3', 'yolov3_channels')
		model_def_config_path = yolov3_config.get('yolov3', 'model_def_config_path', raw=True)
		self.module_defs = parse_yolov3_def(model_def_config_path)
		#print('Module defs:', self.module_defs)
		
		# Yolov3 weights path
		self.weight_path = yolov3_config.get('yolov3', 'weight_path', raw=True)
		
		# Filtering object class names
		self.filtering_img_mode = yolov3_config.getboolean('class_names', 'filtering_img_mode')
		target_detection_class_names = yolov3_config.get('class_names', 'target_detection_class_names')
		target_detection_class_names = ast.literal_eval(target_detection_class_names) # e.g. ['person']
		object_class_names_list_path = yolov3_config.get('class_names', 'object_class_names_list_path', raw=True)
		self.object_class_names_list = load_classes(object_class_names_list_path) # e.g. ['person', ...]
		self.class_num = len(self.object_class_names_list)
		
		self.classes_index_list = []
		for i, c in enumerate(self.object_class_names_list):
			if c in target_detection_class_names:
				self.classes_index_list.append(i)
		
		# Input img size
		self.resize_width_height = yolov3_config.getint('yolov3', 'resize_width_height')
		
		# Threshold
		self.object_confidence_threshold = yolov3_config.getfloat('threshold', 'object_confidence_threshold')
		self.nms_thres = yolov3_config.getfloat('threshold', 'nms_thres')

		# TensorRT
		self.tensorrt_mode = tensorrt_config.getboolean('tensorrt', 'tensorrt_mode')
		
		# Yolov3
		if self.tensorrt_mode:
			self.engine_path = tensorrt_config.get('onnx_tensorrt_engine_path', 'save_tensorrt_engine_path', raw=True)
			# Load tensorrt engine
			self.__load_tensorrt_engine()
			self.__allocate_buffers()
			self.__build_yolo_layers()
		else:		
			# Init yolov3
			self.detector = Yolov3(self.module_defs, yolov3_channels).to(self.device)
			# Load pretrained weight
			self.__load_detector()
		
		# ByteTrack
		self.tracker = ByteTracker(fps, first_track_thresh, second_track_thresh, match_thresh, track_buffer, self.resize_width_height)
		
		print('-'*60)
			
	def __load_tensorrt_engine(self):
		print("Reading engine from file {}".format(self.engine_path))
		with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
			self.yolov3_trt_engine = runtime.deserialize_cuda_engine(f.read())
			self.yolov3_trt_context = self.yolov3_trt_engine.create_execution_context()
	
	def __build_yolo_layers(self):
		modules_yolo = [module_def for module_def in self.module_defs if module_def["type"] == "yolo"]
		
		self.yolo_layer_list = []
		
		for module_yolo in modules_yolo:
			anchor_idxs = [int(x) for x in module_yolo["mask"].split(",")]
			
			# Extract anchors
			anchors = [int(x) for x in module_yolo["anchors"].split(",")]
			anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
			anchors = [anchors[i] for i in anchor_idxs]	
			
			# Yolo layer
			yolo_layer = YOLOLayer(anchors, self.class_num).to(self.device).eval()
				
			self.yolo_layer_list.append(yolo_layer)
		
	def __allocate_buffers(self):		
		self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.yolov3_trt_engine)
	
	def __load_detector(self):
		
		print('Detection model load ...')
		print('Load weights from {}.'.format(self.weight_path))
		
		# Load pretrained detector
		if self.weight_path.endswith(".pt"):
			
			# Checkpoint
			checkpoint = torch.load(self.weight_path, map_location=self.device)
			self.detector.load_state_dict(checkpoint)
		else:
				
			# .weight
			self.detector.load_darknet_weights(self.weight_path)

		self.detector.eval()

	def predict(self, img): # e.g. (640, 960, 3)
		
		# Preprocessing input image outputs (batch size = 1): 1 / resizing img size
		detection_img = processing_detection_image(img.copy(), self.resize_width_height).to(self.device) # e.g. (1, 3, 416, 416)
		
		with torch.no_grad():
			
			start_time = time.time()
			
			## --- Detection ---
			# Do inference with TensorRT
			if self.tensorrt_mode:
				
				batch_size = detection_img.size()[0]
				
				# Output shapes expected by the post-processor
				output_shapes = [(batch_size, (len(self.object_class_names_list) + 5) * 3, self.resize_width_height // 32, self.resize_width_height // 32),
					     (batch_size, (len(self.object_class_names_list) + 5) * 3, self.resize_width_height // 16, self.resize_width_height // 16),
					     (batch_size, (len(self.object_class_names_list) + 5) * 3, self.resize_width_height // 8,  self.resize_width_height // 8)]
				#print('Output shapes of tensorrt engine:', output_shapes)
				
				# Set host input to the image.
				self.inputs[0].host = detection_img.cpu().numpy()
				
				# Get trt outputs
				trt_outputs = do_inference_v2(self.yolov3_trt_context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
				
				# Reshape    
				trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
				
				yolo_outputs = []
				for yolo_layer, trt_output in zip(self.yolo_layer_list, trt_outputs):
					yolo_outputs.append(yolo_layer(torch.tensor(trt_output).to(self.device), detection_img.size(2)))
				
				detections = torch.cat(yolo_outputs, 1) # e.g. (1, 10647, 85)
				
			else:

				# yolov3 outputs: 1 / number of proposals / (center_x, center_y, w, h, conf) + number of class names
				detections = self.detector(detection_img) # e.g. (1, 10647, 85)
				
			# NMS (Non maximum suppression) outputs: 1 / number of objects to detect / absolute_scale(x, y, x, y), score, class
			detections = non_max_suppression(detections, conf_thres=self.object_confidence_threshold, iou_thres=self.nms_thres, xywh=True) # e.g. (1, 3, 6)
			
			detections = detections[0]
			
			# Filtering classes outputs: number of filtering objects / absolute_scale(x, y, x, y), score, class
			if self.filtering_img_mode:
				detections = filter_classes(detections, self.classes_index_list) # e.g. (1, 6)

			# Rescaling boxes to target image size
			detections = [sd for sd in scale_coords(detections, detection_img.shape[2:4], img.shape[0:2])]
			
			## --- Multi object tracking --- 
			track_list = self.tracker.update(np.array(detections), xyxy=True)
			
			## --- Split data --- 

			# Get id list
			id_list = [t.track_id for t in track_list]

			# Get box list
			box_list = [t.tlbr for t in track_list]
			
			# Get object class names
			class_list = [self.object_class_names_list[int(t.class_name)] for t in track_list]
			
			# Get conf scores
			conf_list = [t.score for t in track_list]

			# Number of objects
			num_objects = len(box_list)
			
			end_time = time.time()

			print('Yolov3 inf time:', end_time - start_time)
					
			return id_list, box_list, class_list, conf_list, num_objects
			
