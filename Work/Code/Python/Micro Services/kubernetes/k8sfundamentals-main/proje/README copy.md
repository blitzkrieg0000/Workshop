**6:** Wordpress uygulamasını hem "test" hem de "production" namespace'lerinde deploy edin. (wordpress:latest ve mysql:5.6 imajlarından oluşturulacak)

- mysql "ClusterIp" tipi bir servisle cluster içinden erişilebilir olacak. 
- Her iki uygulamanın da uzun dönem verilerini "persistent volume"ler üzerinde saklanacak.
- Her iki uygulamanın da hiç bir hassas bilgisi "ör: şifre" uygulama ya da yaml dosyaları içerisinde tutulmayacak. 
- Her iki uygulama da aynı worker node üstünde schedule edilecek.
- Her iki uygulama için de cpu ve memory kaynak kısıtları tanımlı olacak.  

**7:** "test" namespace'inde deploy edilen wordpress uygulaması "testblog.example.com", "production" namespace'inde deploy edilen wordpress uygulaması "companyblog.example.com" olarak ingress üstünden dış dünyaya expose edilecek. 

**8:** "production" namespace'inde "ozgurozturknet/k8s:v1" imajından, 5 replikalı, update stratejisi olarak aynı anda 2 pod'un update edilebileceği bir deployment oluşturun. "/healthcheck" endpoint'ini sorgulayan "liveness probe" ve "/ready" endpoint'ini sorgulayan bir "readiness probe" tanımları da olsun. 

**9:** Bir önceki görevde oluşturduğunuz deployment'i "loadbalancer" tipi bir servisle dış dünyadan erişilebilir hale getirin. 

**10:** Bu deployment'i önce 3 replikaya indirin. Ardından 10 replikaya çıkarın. Sonrasında bu deployment'i "ozgurozturknet/k8s:v2" imajıyla güncelleyin.

**11:** "fluentd" uygulamasını bir "daemonset" olarak cluster'da deploy edin. 

**12:** 2 node'lu bir "mongodb" cluster'i "statefulset" olarak cluster'da deploy edin. "mongodb" cluster'ın çalıştığından emin olun. 

**13:** Cluster'da tüm objeler üstünde "okuma ve listeleme" haklarına sahip bir "service account" oluşturun. Bu service account’u bağladığınız bir pod oluşturun ve bağlanarak "curl" ile cluster’daki tüm podları listeleyin. 

**14:** Worker node'lardan bir tanesinin üzerindeki tüm podları tahliye edin ve ardından yeni pod schedule edilememesini sağlayın. 