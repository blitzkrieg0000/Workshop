import logging
from pickletools import uint8
import time
from concurrent import futures

import grpc
import cv2
import numpy as np
from proto_main import ManagerService_pb2 as rc
from proto_main import ManagerService_pb2_grpc as rc_grpc
from confluent_kafka import Producer

class ManagerServicer(rc_grpc.ManagerServiceServicer):
    def __init__(self) -> None:
        super().__init__()

        self.producerClient = Producer({
            'bootstrap.servers': ",".join(["broker:9092"]),
            'message.max.bytes': 20971520,
            'linger.ms': 5,
            'queue.buffering.max.messages':200000,
            'batch.size': 3
        })

    #? Stream-Stream
    def SampleProcess(self, request_iterator, context):
        print("ENTER")

        # frame = np.ones((1520,2704,3), np.uint8)
        # frame = np.random.randint(0, 255, size=(1080,1080,3), dtype=np.uint8) #
        frame = np.random.rand(1520,2704,3) * 255
        retval, buffer = cv2.imencode('.jpg', frame)

        print(frame)
        request = next(request_iterator)
        print(request)
        print(frame.shape)
        
        say = 0
        toplam = 0
        try:
            while context.is_active():
                say +=1



                #SAVE FILE
                # f = open("dump.txt", "w")
                # f.write(str(buffer.tobytes()))
                # f.close()


                tic = time.time()

                
                try:
                    self.producerClient.produce("test", buffer.tobytes())
                    self.producerClient.poll(0)
                except BufferError as bfer:
                    logging.error(bfer)
                    self.producerClient.poll(0.1)

                # res, encodedImg = cv2.imencode('.jpg', frame)
                # byte_image = encodedImg.tobytes()
                toc = time.time()

                lag = toc - tic
                toplam += lag
                print(f"Ortalama Server:{say}: ",toplam/(say+1))

                yield rc.ManagerServiceResponse(data = "")
                request = next(request_iterator)
                print(request)
                

            print("iterate bitti")
        except Exception as e:
            print(e)

        self.producerClient.flush()

        print("EXIT")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    rc_grpc.add_ManagerServiceServicer_to_server(ManagerServicer(), server)
    server.add_insecure_port("0.0.0.0:60051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
