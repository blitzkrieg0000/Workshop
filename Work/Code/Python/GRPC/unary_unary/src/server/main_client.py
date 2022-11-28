from __future__ import print_function

import logging
import time

import grpc
import main_pb2 as rc
import main_pb2_grpc as rc_grpc
from tqdm import tqdm

logging.basicConfig(format='%(levelname)s - %(asctime)s => %(message)s', datefmt='%d-%m-%Y %H:%M:%S', level=logging.NOTSET)

MAX_MESSAGE_LENGTH = 100*1024*1024 # 100MB
class MainClient():

    def __init__(self):    # SSL vs. olmadan bir bağlantı açıyoruz. Bağlantı özelliklerini burada belirliyoruz.
        self.channel = grpc.insecure_channel('localhost:50000',
                        options=[
                            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),       # Gönderilen ve Alınan mesaj uzunluğunu sınırlamaya yarar. 
                            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),    # Default olarak MAX_MESSAGE_LENGTH: 4 MB dır.
                        ],
                        compression=grpc.Compression.NoCompression                      # Sıkıştırma türü: NoCompression, Deflate, Gzip) 
                    )

        self.stub = rc_grpc.MainStub(self.channel)    # rc_grpc."<ProtoDosyaAdı>Stub" dan bir taslak (stub) alıyoruz.


    def Processs(self, msg:str):

        msg = msg.encode("utf-8")

        # Request paketini hazırlıyoruz.
        request = rc.ProcessRequest(data=msg)
        
        # Request gönder ve response al
        response = self.stub.Process(request)    # Serverdaki tanımladığımız fonksiyonu çağırıyoruz. "RPC" olayı bu oluyor.
        
        return response


    def disconnect(self):
        self.channel.close()


if __name__ == "__main__":
    mainClient = MainClient()

    for i in tqdm(range(10)):
        msg = f"Test: {i}"
        logging.info(f"Gönderildi: {msg}")
        response = mainClient.Processs(msg)
        logging.info(response.data.decode("utf-8")) 


