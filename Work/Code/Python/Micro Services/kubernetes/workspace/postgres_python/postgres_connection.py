import collections
import psycopg2
import psycopg2.extras

class database(object):
    def __init__(self):
        self.conn = None
        self.cursor = None
        self.lasttime = None
        self.key_time = collections.defaultdict(list)
    #end
    
    def connect2(self):
        if self.conn == None:
            #Bağlantıyı kurar
            try:
                self.conn = psycopg2.connect(host="192.168.29.16", database="tenis", user="tenis", password="2sfcNavA89A294V4") 
                self.cursor = self.conn.cursor()
                print("PostgreSQL Bağlantısı Kuruldu!")
            except Exception as e:
                print("PostgreSQL Bir Sorunla Karşılaşıldı!")
                print(e)     
    #end

    def select(self):
        query = """SELECT * FROM video;"""
        self.cursor.execute(query)
        values = self.cursor.fetchall()
        return values
    #end

    def disconnect(self):
        #Bağlantıyı keser
        if self.conn:
            self.cursor.close()
            self.conn.close()
            print("PostgreSQL Bağlantısı Kesildi!")
            self.conn = None
            self.conn = None
    #end
#end::class

if __name__ == "__main__":
    db = database()
    db.connect2()
    #val = db.select()
    #print(val)