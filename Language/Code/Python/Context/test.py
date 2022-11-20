from confluent_kafka import Consumer

class ConsumerGen():
    def __init__(self, consumerGroup, offsetMethod, topics, limit):
        self.consumer = self.connectKafkaConsumer(consumerGroup, offsetMethod)
        self.consumer.subscribe(topics)
        
        #FLAG && COUNTERS
        self.stopFlag=False
        self.limit = limit
        self.limit_count=0
        self.ret_limit=0

    def connectKafkaConsumer(self, consumerGroup, offsetMethod):
        return Consumer({
                'bootstrap.servers': ",".join(["192.168.29.62:9092"]),
                'group.id': consumerGroup,
                'auto.offset.reset': offsetMethod
            })

    def closeConnection(self):
        try:
            self.consumer.close()
        except Exception as e:
            print(f"ConsumerGen: {e}")

    def stopGen(self):
        self.stopFlag = True

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if (self.limit!=-1 and self.limit_count==self.limit) or self.ret_limit>10 or self.stopFlag:
                self.closeConnection()
                raise StopIteration

            msg = self.consumer.poll(0)
            
            if msg is None:
                self.ret_limit +=1
                continue
            
            if msg.error():
                print("Consumer-error: {}".format(msg.error()))
                self.ret_limit +=1
                continue
            self.ret_limit=0
            if self.limit!=-1:
                self.limit_count = 1 + self.limit_count
            
            return msg

def gen():
    consumeGENERATOR = ConsumerGen("consumergroup-2", "earliest", ["tenis_saha_1-0-35623388544027117967351014952362451246"], 2)
    
    consumer_activity = True
    
    for msg in consumeGENERATOR:
        if not consumer_activity:
            consumeGENERATOR.stopGen()
            
        yield msg.value()

if __name__ == "__main__":
    generator = gen()
    for item in generator:
        print(str(item))