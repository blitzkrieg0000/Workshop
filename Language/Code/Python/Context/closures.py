
def outer(fonksiyon):

    def inner(x, y):
        print(f"ÇALIŞAN FONKSİYON {fonksiyon.__name__}")
        return fonksiyon(x, y)
        
    def cevir(x):
        x = str.upper(x)
        return fonksiyon(x)

    if fonksiyon.__name__ == "ekranaYazdir":
        return cevir

    return inner


def ekranaYazdir(x):
    print(x)

def topla(x, y):
    return x+y

def cikar(x, y):
    return x-y


inner_topla = outer(topla)
inner_cikar = outer(cikar)

toplam = inner_topla(3, 5)
cikarim = inner_cikar(5, 2)


yazici = outer(ekranaYazdir)
yazici("burakhan")


print(toplam, cikarim)