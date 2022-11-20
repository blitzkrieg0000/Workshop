
class person():
    def __init__(self): #Constructor
        self.yas = 50

    def amac(self, x):
        print(self.yas)
        return 2.5*x



class ogrenci(person):
    def __init__(self):
        super().__init__()

        pass


ogr1 = ogrenci()
print(ogr1.amac(5))