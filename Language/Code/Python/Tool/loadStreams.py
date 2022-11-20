
class LoadStreams:
    def __init__(self, source=0, img_size=640, stride=32):
        
        self.source = source
        self.img_size = img_size
        self.stride = stride

        self.imgs = None
        self.thread = None
        self.fps = 0
        self.frames = 0
        
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        fps = cap.get(cv2.CAP_PROP_FPS)
        self.frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')
        self.fps = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback
        _, self.imgs = cap.read()  # guarantee first frame
        self.thread = threading.Thread(target=self.update, args=(cap, self.source), daemon=True)
        self.thread.start()

    def update(self, cap, source):
        n, f, read = 0, self.frames, 1
        while cap.isOpened() and n < f:
            n += 1

            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs = im
                else:
                    self.imgs = np.zeros_like(self.imgs)
                    cap.open(source)
            time.sleep(1 / self.fps)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not self.thread.is_alive() or cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = self.imgs.copy()
        img = self.letterbox(img0, self.img_size, stride=self.stride, auto=False)[0]

        # Convert
        img = np.expand_dims(img, axis=0)
        img = img[..., ::-1]  # BGR to RGB, BHWC to BCHW
        img = img.transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)

        return img, img0

    def __len__(self):
        return len(self.source)  # 1E12 frames = 32 streams at 30 FPS for 30 years

    def letterbox(self,im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)


