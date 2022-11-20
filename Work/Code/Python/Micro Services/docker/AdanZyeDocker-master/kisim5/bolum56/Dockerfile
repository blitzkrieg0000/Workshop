FROM centos:latest
ENV TEST="Bu bir denemedir"

# Bu exec formatında $TEST i string olarak yazdırır. Ortam değişkenine ulaşamaz çünkü container içindeki shell i process olarak çalıştırmaz. Direkt komutu process olarak çalıştırır.
# CMD ["echo", "$TEST"] 

#Shell Format: Container içerisinde bir shell çalıştır ve orada komutu çalştır. Bu sayede ortam değişkenlerine erişim sağlar. Komutu shell içinde çalıştırır.Process olarak çalıştırmaz.
CMD echo $TEST