#!pip install qrcode
import qrcode

#Versiyon = QRKod sembolünün karmaşıklığıdır. Sayı büyüdükçe okunurluk azalır. : 1-40 arası
#box_size = Kutu boyutu
#border = Kenar boşluğu
qr = qrcode.QRCode( version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
qr.add_data("Link: https://dgharge.com\n")
qr.add_data("Tel: 01234567890")

qr.make(fit=True)
img = qr.make_image(fill_color="white", back_color="black")
img.save("qrcode.png")