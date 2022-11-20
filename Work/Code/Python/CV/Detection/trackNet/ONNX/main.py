import cv2
from libs.trackNet_onnx_v2 import TrackNetObjectDetection

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
		(draw_x, draw_y), canvas = detector.detect(img, draw=True)

		print(draw_x, draw_y)

		cv2.imshow('', canvas)
		if cv2.waitKey(0) & 0xFF == ord("q"):
			break

	cv2.destroyWindow()
	cap.release()
