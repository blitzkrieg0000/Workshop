import queue
import cv2
import numpy as np
from keras.models import *
from keras.layers import *


def TrackNet(n_classes,  input_height, input_width): # input_height = 360, input_width = 640  total_input(9,360,640)

	imgs_input = Input(shape=(9, input_height, input_width))

	#layer1
	x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(imgs_input)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer2
	x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer3
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

	#layer4
	x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer5
	x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer6
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

	#layer7
	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer8
	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer9
	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer10
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

	#layer11
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer12
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer13
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer14
	x = ( UpSampling2D( (2,2), data_format='channels_first'))(x)

	#layer15
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer16
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer17
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer18
	x = ( UpSampling2D( (2,2), data_format='channels_first'))(x)

	#layer19
	x = ( Conv2D( 128 , (3, 3), kernel_initializer='random_uniform', padding='same' , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer20
	x = ( Conv2D( 128 , (3, 3), kernel_initializer='random_uniform', padding='same' , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer21
	x = ( UpSampling2D( (2,2), data_format='channels_first'))(x)

	#layer22
	x = ( Conv2D( 64 , (3, 3), kernel_initializer='random_uniform', padding='same'  , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer23
	x = ( Conv2D( 64 , (3, 3), kernel_initializer='random_uniform', padding='same'  , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer24
	x =  Conv2D( n_classes , (3, 3) , kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	o_shape = Model(imgs_input , x ).output_shape
	print ("layer24 output shape:", o_shape[1],o_shape[2],o_shape[3])
	#layer24 output shape: 256, 360, 640

	OutputHeight = o_shape[2]
	OutputWidth = o_shape[3]

	#reshape the size to (256, 360*640)
	x = (Reshape((  -1  , OutputHeight*OutputWidth   )))(x)

	#change dimension order to (360*640, 256)
	x = (Permute((2, 1)))(x)

	#layer25
	gaussian_output = (Activation('softmax'))(x)

	model = Model( imgs_input , gaussian_output)
	model.outputWidth = OutputWidth
	model.outputHeight = OutputHeight

	#Model Özeti
	#model.summary()

	return model


class TrackNetObjectDetection(object):
	def __init__(self):
		self.n_classes = 256
		self.save_weights_path = "TF/weights/model.3"
		self.width, self.height = 640, 360
		self.img, self.img1, self.img2 = None, None, None

		#TF
		self.model = TrackNet( n_classes =self.n_classes , input_height=self.height, input_width=self.width   )
		self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
		self.model.load_weights(self.save_weights_path)
		

		self.q = queue.deque()
		for i in range(0,8):
			self.q.appendleft(None)

	def detect(self, frame):
		canvas = frame.copy()
		output_height, output_width = frame.shape[:-1]
		self.current_frame = cv2.resize(frame, ( self.width , self.height ))
		self.current_frame = self.current_frame.astype(np.float32)
		
		#Toplam 3 image olana kadar bekler ve ardışık 3 frami alır
		self.img2 = self.img1
		self.img1 = self.img
		self.img = self.current_frame

		if self.img2 is not None:
			# 3 ardışık resmi yan yana birleştir (width , height, rgb*3)
			X = np.concatenate((self.img, self.img1, self.img2), axis=2)

			# TrackNet "Değişik bir mahlukat" olduğundan yani 640, 360, 3 değilde 3, 640, 360 şeklinde input shape ister. :)
			X = np.rollaxis(X, 2, 0)
			
			# Inference(Forward) Adımı
			print("\n",X.shape,"\n")

			
			pr = self.model.predict( np.array([X]) )[0]

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
				#if only one tennis be detected
				if len(circles) == 1:

					x = int(circles[0][0][0])
					y = int(circles[0][0][1])

					#push x,y to queue
					self.q.appendleft([x,y])   
					#pop x,y from queue
					self.q.pop()    
				else:
					#push None to queue
					self.q.appendleft(None)
					#pop x,y from queue
					self.q.pop()
			else:
				#push None to queue
				self.q.appendleft(None)
				#pop x,y from queue
				self.q.pop()

			#draw current frame prediction and previous 7 frames as yellow circle, total: 8 frames
			for i in range(0,8):
				if self.q[i] is not None:
					draw_x = self.q[i][0]
					draw_y = self.q[i][1]
					print((draw_x, draw_y))
					canvas = cv2.ellipse(canvas, (draw_x, draw_y), (4, 4), 0, 0, 360, (0, 255, 255), 1)

		return canvas


if "__main__" == __name__:
	input_video_path = "video/input.mp4"
	cap = cv2.VideoCapture(input_video_path)
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	detector = TrackNetObjectDetection()

	while(True):
		ret, img = cap.read()
		
		if not ret: 
			break

		canvas = img
		canvas = detector.detect(img)

		cv2.imshow('', canvas)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	cv2.destroyWindow()
	cap.release()