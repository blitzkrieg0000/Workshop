# THREADS
# NOT: Threadler "neredeyse" eş zamanlı çalışan yapılardır.

import threading
import time
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
filehdlr = logging.FileHandler('context/Threads/log.txt', mode='w')
filehdlr.setLevel(logging.INFO)
logger.addHandler(filehdlr)


class test():
    x=0
    def arttir(self):
        self.x+=1
    def azalt(self):
        self.x-=1
veri = test()

def counter(veri:test, logger):
    id = threading.current_thread().ident

    while veri.x<=10:
        logger.info(f"{id} - counter: {veri.x}")
        veri.arttir()
        time.sleep(1)

if "__main__" == __name__:
    #MAIN THREAD ID
    id = threading.current_thread().ident
    print(f"MainThread: {id}")

    #LIST THREADS
    names = [thread.name for thread in threading.enumerate()]
    print(names) 

    threads = []
    for i in range(3):
        # Ana process işlemi bitirip kapanması gerektiğinde, Daemon threadleri beklemez.
        t = threading.Thread(name=f"CounterThread-{i}", target=counter, args=[veri, logger], daemon=True)
        threads.append(t)

    #Threadleri başlat
    for thread in threads:
        thread.start()

    # "join" ifadesi her çağrıldığı an main thread, ilgili threadin sonlanmasını bekler. 
    # Zaten oluşturulan thread, daemon değilse ana thread kapanmak için, o threadin işini bitirmesini bekleyecektir. 
    # Thread, daemon olarak kurgulanmışsa, oluşturulan yeni threadin işini yapmasını beklemek için "join" ifadesi kullanılır. 
    for i, thread in enumerate(threads):
        print(i, " joining...")
        thread.join()


    names = [thread.name for thread in threading.enumerate()]
    print(names)