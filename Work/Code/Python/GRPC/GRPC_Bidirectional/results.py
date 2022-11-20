# Pickle dumps --> 0.01    
# Pickle loads --> 0.0017  
# ToBase64     --> 0.00022 
# FrameToByte  --> 0.02   
# BytesToFrame --> 0.017  

# Kafka: 0.0045/2.7K, 0.0013/FHD

#!x FPS için Tüm işlemlerin tamamlanması gereken max süreler
#! 30  -> 0,033
#! 60  -> 0,016 (Önerilen yöntem ile tam karşılıyor.Kafka: 0.0045/2.7K yıda eklersek 60FPS verebiliyoruz. )
#! 240 -> 0,004

#! Şuan kullanılan yöntem: 0.02(FrameToJpgEncodeByte) + 0.00035(gRPC) + 0.017 (BytesToFrame) + 0.00022 (ToBase64) + 0.00019(gRPC (string)) => 0,037 seconds
#? Önerilen yöntem: 0.01(PickleDumps) + 0.00035(gRPC) + 0.0017(PickleLoads)+ 0.00022(ToBase64) + 0.00035(gRPC) => 0,012 seconds