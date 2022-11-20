import queue
import cv2
import numpy as np
import onnxruntime


class TrackNetObjectDetection(object):
	def __init__(self):
		self.queue_length = 10
		self.n_classes = 256
		self.save_weights_path = "ONNX/weights/model.onnx"
		self.width, self.height = 640, 360
		self.img, self.img1, self.img2 = None, None, None
	
		#ONNX
		self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  #'TensorrtExecutionProvider'
		self.session = onnxruntime.InferenceSession(self.save_weights_path, providers=self.providers)
		self.input_cfg = self.session.get_inputs()[0]
		self.input_name = self.input_cfg.name
		self.outputs = self.session.get_outputs()
		self.output_names = [o.name for o in self.outputs]

		self.q = queue.deque()
		for i in range(0, self.queue_length):
			self.q.appendleft(None)

	def detect(self, frame, draw=False):
		canvas = frame.copy()
		output_height, output_width = frame.shape[:-1]
		self.current_frame = cv2.resize(frame, ( self.width , self.height ))
		self.current_frame = self.current_frame.astype(np.float32)
		
		#Toplam 3 image olana kadar bekler ve ardışık 3 frami alır
		self.img2 = self.img1
		self.img1 = self.img
		self.img = self.current_frame
		
		draw_x = None
		draw_y = None
		
		if self.img2 is not None:
			# 3 ardışık resmi yan yana birleştir (width , height, rgb*3)
			X = np.concatenate((self.img, self.img1, self.img2), axis=2)

			# TrackNet "Değişik bir mahlukat" olduğundan yani 640, 360, 3 değilde 3, 640, 360 şeklinde input shape ister. :)
			X = np.rollaxis(X, 2, 0)
			
			# Inference(Forward) Adımı
			print("\n",X.shape,"\n")

			#Inference
			pr = self.session.run(None, {self.input_name : [X]})[0]

			#TrackNet çıkışı ( net_output_height*model_output_width , n_classes ) olduğundan
			#tekrar boyutlandırmamız gerekiyor: ( net_output_height, model_output_width , n_classes(depth) )
			#.argmax( axis=2 ) => Argmax sağolsun en iyi ihtimal değerini döndürüyoruz.
			pr = pr.reshape(( self.height ,  self.width , self.n_classes ) ).argmax( axis=2 )

			# numpy.int64 -> numpy.uint8 (0-255 ya hani)
			pr = pr.astype(np.uint8) 

			#Çıkışı tekrar boyutlandırırız zaten heatmap olduğundan şekiller önemli bizim için
			heatmap = cv2.resize(pr, (output_width, output_height ))

			#Threshold
			ret, heatmap = cv2.threshold(heatmap,127,255,cv2.THRESH_BINARY)

			#Hough dönüşümü ile kapalı şekilleri(çemberleri) buluruz 2<=çap<=7
			circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT,dp=1,minDist=1,param1=50,param2=2,minRadius=2,maxRadius=7)

			if circles is not None:
				if len(circles) == 1:

					x = int(circles[0][0][0])
					y = int(circles[0][0][1])

					self.q.appendleft([x,y])   
					self.q.pop()    
				else:
					self.q.appendleft(None)
					self.q.pop()
			else:
				self.q.appendleft(None)
				self.q.pop()
			

			if draw:
				for i in range(0, self.queue_length):
					if self.q[i] is not None:
						draw_x = self.q[i][0]
						draw_y = self.q[i][1]
						print((draw_x, draw_y))
						canvas = cv2.ellipse(canvas, (draw_x, draw_y), (4, 4), 0, 0, 360, (0, 255, 255), 1)

		return (draw_x, draw_y), canvas

