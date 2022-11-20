# GENERATORS
X = [1,2,3,4,5,6,7,8,9,10]

#! Yöntem-1
# def elemanlar(liste):
#     for item in liste:
#         yield item*item

# for k in elemanlar(X):
#     print(k)


#! Yöntem-2
# liste = (item*item for item in X)

# for x in liste:
#     print(x)


#! Yöntem-3
class MyGenerator():
    def __init__(self, X):
        self.current = -1
        self.liste = X
        self.stopFlag = False

    def stopGen(self):
        self.stopFlag=True

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if (self.current == len(self.liste)) or self.stopFlag:
            raise StopIteration

        x = self.liste[self.current]
        return x*x
        
gen = MyGenerator(X)

for x in gen:
    print(x)
    if x > 30:
        gen.stopGen()

    