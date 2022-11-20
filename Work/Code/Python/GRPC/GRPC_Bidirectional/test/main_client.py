import queue
import time

import grpc

from proto_main import ManagerService_pb2 as rc
from proto_main import ManagerService_pb2_grpc as rc_grpc

#? Stream-Stream
class ClientManager():


    def __init__(self):
        MAX_MESSAGE_LENGTH = 12 * 1024 * 1024
        options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ]
        self.channel = grpc.insecure_channel(target="localhost:60051", options=options)
        self.stub = rc_grpc.ManagerServiceStub(channel=self.channel)


    #! Iterator Object
    def process_iterator(self):
        def entry_request_iterator():
            for idx in range(1, 16):
                entry_request = rc.ManagerServiceRequest(title=f"Test {idx}", code=f"T{idx}", description=f"Test {idx} description")
                yield entry_request

        for entry_response in self.stub.SampleProcess(entry_request_iterator()):

            print(entry_response)


    #! Queue Sentinel
    def process_sentinel(self):
        send_queue = queue.SimpleQueue()
        def gen(send_queue):
            while True:
                try:
                    item = send_queue.get(block=True)
                    if item is None:
                        raise queue.Empty
                    yield item
                except queue.Empty as e:
                    print("Bitti")
                    raise StopIteration

        stream_iterator = self.stub.SampleProcess(gen(send_queue))
        
        print("1.istek gönderiliyor")
        send_queue.put(rc.ManagerServiceRequest(data = ""))

        empty_message = rc.ManagerServiceRequest(data = "")
        return stream_iterator, send_queue, empty_message


if "__main__" == __name__:
    
    #! GoPro10: 5.3K/60FPS, 4K/120FPS, 2.7K/240FPS, FHD/240FPS 
    # gRPC stream-stream one-way(unary lag (localhost)---------------------
    #*> b""                               (Byte)  : 0.00028 average seconds
    #*> FHD  1920x1080x3 Resim JPG ENCODE (Bytes) : 0.00034 average seconds
    #?> 2.7K 2704x1520x3 Resim JPG ENCODE (Bytes) : 0.00035 average seconds
    #*> 4K   3840x2160x3 Resim JPG ENCODE (Bytes) : 0.00053 average seconds
    #*> 5.3K 5312x2988x3 Resim JPG ENCODE (Bytes) : 0.00077 average seconds

    manager = ClientManager()
    stream_iterator, send_queue, empty_message = manager.process_sentinel()
    
    tic = time.time()
    toplam = 0
    for i, res in enumerate(stream_iterator):
        toc = time.time()
        
        lag = toc - tic
        toplam += lag
        print(f"Ortalama:{i}: ",toplam/(i+1))

        send_queue.put(empty_message)
        tic = time.time()