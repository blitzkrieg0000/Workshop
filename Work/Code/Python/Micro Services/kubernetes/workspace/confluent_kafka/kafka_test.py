from confluent_kafka.admin import AdminClient, NewTopic

KAFKA_BOOTSTRAP_SERVERS = ['192.168.29.58:9092'] # Broker IP-Container ortamının dışında olduğumuzdan ad çözümlemesi yapmak için hosts, ve /etc/resolv.d dosyasını ayarlamak gerekiyor.
TOPICS_NUM_PARTITION = 1

class KafkaManager():

    def __init__(self):
        self.adminC = AdminClient({'bootstrap.servers': ",".join(KAFKA_BOOTSTRAP_SERVERS)})

    def getTopics(self, prefix):
        return self.adminC.list_topics().topics.keys()
        topics = set(filter(lambda t: t.startswith(prefix), self.adminC.list_topics().topics.keys()))
        return topics

    def createTopic(self, topic_name):
        new_topics = []
        new_topics.append(NewTopic(topic_name, num_partitions=TOPICS_NUM_PARTITION))
        
        if new_topics:
            fs = self.adminC.create_topics(new_topics)
            for topic, f in fs.items():
                try:
                    f.result()
                    print("Topic {} created".format(topic))
                except Exception as e:
                    print("Failed to create topic {}: {}".format(topic, e)) 

if "__main__" == __name__:
    km = KafkaManager()
    topic_list = km.getTopics("test")
    print(topic_list)

    # km.createTopic("test")

    topic_list = km.getTopics("test")
    print(topic_list)