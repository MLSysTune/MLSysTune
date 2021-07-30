# exchange to send message to master
import logging

from kombu import Exchange, Queue, producers
import json
import kombu.mixins

from selftf.lib.common import ComputeNode
from selftf.lib.message import Message, MessageSerializer

# Different role:
# monitor
# client
# compute_node
# compute_node_tf


class KombuQueueManager(object):

    MONITOR_KOMBU_EXCHANGE_NAME = "monitor"
    MONITOR_KOMBU_QUEUE_NAME = "monitor"
    CLIENT_KOMBU_EXCHANGE_NAME = "client"

    COMPUTE_NODE_EXCHANGE = 'compute_node'

    CHIEF_KOMBU_EXCHANGE_NAME = 'chief'

    # exchange to send message to master
    monitor_exchange = Exchange(MONITOR_KOMBU_EXCHANGE_NAME, type='direct')

    # exchange to send message to compute_nodes
    compute_nodes_exchange = Exchange(COMPUTE_NODE_EXCHANGE, type='direct')

    client_exchange = Exchange(CLIENT_KOMBU_EXCHANGE_NAME, type='direct')

    chief_exchange = Exchange(CHIEF_KOMBU_EXCHANGE_NAME, type='direct')

    monitor_queue = Queue(MONITOR_KOMBU_QUEUE_NAME, monitor_exchange)

    chief_queue = Queue(CHIEF_KOMBU_EXCHANGE_NAME, chief_exchange)

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_monitor_exchange(self):
        return self.monitor_exchange

    def get_compute_nodes_exchange(self):
        return self.compute_nodes_exchange

    def get_compute_node_queue(self, compute_node):
        if not isinstance(compute_node, ComputeNode):
            raise TypeError()

        routing_key = self.get_compute_node_routing_key(compute_node)
        return Queue(routing_key, self.compute_nodes_exchange, routing_key=routing_key)

    def get_compute_node_tf_queue(self, compute_node):
        if not isinstance(compute_node, ComputeNode):
            raise TypeError()

        routing_key = self.get_compute_node_tf_routing_key(compute_node)
        return Queue(routing_key, self.compute_nodes_exchange, routing_key=routing_key)

    def get_monitor_queue(self):
        return self.monitor_queue

    def get_client_exchange(self):
        return self.client_exchange

    def get_client_queue(self, client_id):
        return Queue(client_id, self.client_exchange, routing_key=client_id)

    def get_compute_node_routing_key(self, compute_node):
        if not isinstance(compute_node, ComputeNode):
            raise TypeError()

        return compute_node.get_id()

    def get_compute_node_tf_routing_key(self, compute_node):
        return compute_node.get_id()+"_tf"

    def get_monitor_name(self):
        return self.MONITOR_KOMBU_EXCHANGE_NAME

    def send_msg(self, conn, message_obj, exchange, declare=None, routing_key=None):
        """
        :param conn:
        :param Message message_obj:
        :param exchange:
        :param declare:
        :param routing_key:
        :return:
        """
        if declare is None:
            declare = [exchange]

        is_success = False
        while not is_success:
            try:
                with producers[conn].acquire(block=True) as producer:

                    producer.publish(
                        json.dumps(message_obj, cls=MessageSerializer),
                        exchange=exchange,
                        declare=declare,
                        routing_key=routing_key
                    )
                is_success = True
            except Exception as e:
                self.logger.exception(f"Fail to send msg, retry. Error msg:{e}")

    def send_msg_to_monitor(self, conn, message_obj):
        if not isinstance(message_obj, Message):
            raise TypeError()

        self.logger.info("Send message from '%s' to '%s' type: %s" % (message_obj.get_source(),
                                                                      message_obj.get_destination(),
                                                                      message_obj.__class__.__name__))

        exchange = self.get_monitor_exchange()
        self.send_msg(conn,message_obj, exchange)

    def send_msg_to_compute_node(self, conn, compute_node_obj, message_obj):
        if not isinstance(message_obj, Message):
            raise TypeError()

        self.logger.info("Send message from '%s' to '%s' type: %s" % (message_obj.get_source(),
                                                                      message_obj.get_destination(),
                                                                      message_obj.__class__.__name__))

        exchange = self.get_compute_nodes_exchange()
        routing_key = self.get_compute_node_routing_key(compute_node_obj)
        self.send_msg(conn, message_obj, exchange, routing_key=routing_key)

    def send_msg_to_compute_node_tf(self, conn, compute_node_obj, message_obj):
        if not isinstance(message_obj, Message):
            raise TypeError()

        self.logger.info("Send message from '%s' to '%s' type: %s" % (message_obj.get_source(),
                                                                      message_obj.get_destination(),
                                                                      message_obj.__class__.__name__))

        exchange = self.get_compute_nodes_exchange()
        routing_key = self.get_compute_node_tf_routing_key(compute_node_obj)
        self.send_msg(conn, message_obj, exchange, routing_key=routing_key)

    def send_msg_to_client(self, conn, message_obj):
        if not isinstance(message_obj, Message):
            raise TypeError()

        client_id = message_obj.get_destination()

        self.logger.info("Send message from '%s' to '%s' type: %s" % (message_obj.get_source(),
                                                                      message_obj.get_destination(),
                                                                      message_obj.__class__.__name__))

        exchange = self.get_client_exchange()
        routing_key = client_id
        self.send_msg(conn, message_obj, exchange, routing_key=routing_key)

    def cleanup(self, queue, conn):
        if not isinstance(queue, Queue):
            raise TypeError()

        with conn.Consumer(queue, callbacks=[lambda a,b:None]) as consumer:
            consumer.purge()

    def get_chief_queue(self):
        return self.chief_queue


class ConsumerMixin(kombu.mixins.ConsumerMixin):
    def __init__(self, connection, message_handler, queue):
        """
        :param Connection connection:
        :param Function message_handler:
        :param Com
        """
        self.queue = queue
        self.message_handler = message_handler
        self.connection = connection

    def get_consumers(self, Consumer, channel):
        return [Consumer(self.queue,
                         callbacks=[self.message_handler],
                         prefetch_count=1)]