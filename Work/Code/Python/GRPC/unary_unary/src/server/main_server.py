import logging
import time
from concurrent import futures

import grpc
import main_pb2 as rc  # Paketlerle ilgili bilgi tutar.
import main_pb2_grpc as rc_grpc  # Servis ile ilgili bilgi tutar.

logging.basicConfig(format='%(levelname)s - %(asctime)s => %(message)s', datefmt='%d-%m-%Y %H:%M:%S', level=logging.NOTSET)



MAX_MESSAGE_LENGTH = 100*1024*1024 # 100MB

# Herhangi bir adla class oluşturuyoruz ve rc_grpc."<ProtoDosyaAdı>Servicer" ile extend ediyoruz.
class MainService(rc_grpc.MainServicer):
    def __init__(self):
        super().__init__()

    # Hangi fonksiyonları protobuff dosyasında tanımlamışsak onları override ediyoruz.
    def Process(self, request, context):      
        
        # Gelen Request paketlerindeki propertylere bu şekilde ulaşabiliyoruz. Byte tipinde "data" diye protobuffta tanımlamıştık.
        msg  = request.data             
        msg = msg.decode("utf-8")

        dump = f"Server Mesajı aldı: {msg}" 
        
        # Response paketimize tekrar tanımladığımız veriyi dolduruyoruz.
        response = rc.ProcessResponse(data=dump.encode("utf-8"))
        
        time.sleep(5.0)

        if not context.is_active():
            logging.warning("Client, server yanıtını beklemeden çıkış yapmış.")

        # Request gelen clienta Response mesaj paketini gönderiyoruz.
        return response


def serve():

    # Aynı anda maksimum yanıtlanacak istek sayısını (oluşacak threadi) belirtiyoruz. (10)
    # Unutmayalım ki orijinal c-python da GIL(GlobalInterpreterLock) olduğu için ne kadar thread üretirsek üretelim; oluşturulan 1 adet process içerisinde sadece CPU nun 1 adet core unu kullanacak.
    # Yani CPU bounding işlemlerde yeni bir sub-process üretip yolumuza devam etmemiz gerekiyor ki performans kaybı yaşamayalım.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                options=[
                    ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),   # Gönderilen ve Alınan mesaj uzunluğunu sınırlamaya yarar. 
                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH) # Default olarak MAX_MESSAGE_LENGTH: 4 MB dır.
                ],
                compression=grpc.Compression.NoCompression                  # Sıkıştırma türü: NoCompression, Deflate, Gzip
            )

    #Oluşturduğumuz class ı oluşturduğumuz thread havuzuna bağlıyoruz.
    rc_grpc.add_MainServicer_to_server(MainService(), server)

    # Server Ayarları Host:Port
    server.add_insecure_port('[::]:50000')

    # Server ı başlatıyoruz.
    server.start()

    # Bu process sonlanana kadar beklemesini sağlıyoruz. Yani biz sonlandırana kadar.
    server.wait_for_termination()



if __name__ == '__main__':
    serve()