# for submitting request to "master" queue
import csv
import json
import logging
import uuid
import os

import sys
from kombu import Connection

from selftf.lib import util
from selftf.lib.common_conf import *
from selftf.lib.message import StartJob, ReplyStartJob, ReplyGetJobStatistic, \
    MessageSerializer, GetJobStatistic, KillJob, \
    TriggerDumpLog, ChangeConfig
from selftf.lib.queue import KombuQueueManager

# client delicated parameters
KEY_ACTION = "action"
KEY_SCRIPT = "script"

KEY_ACTION_SUBMIT_JOB = "submit_job"
KEY_ACTION_GET_JOB = "get_job"
KEY_ACTION_KILL_JOB = "kill_job"
KEY_ACTION_DUMP_LOG = "dump_log"
KEY_ACTION_CHANGE_CONFIG = "change_config"
conf_parser.add_argument("--" + KEY_ACTION)
conf_parser.add_argument("--" + KEY_SCRIPT, default="/usr/local/bin/python2.7")


class Client:
    def __init__(self, conf):
        """
        :param Config conf:
        """
        self.conn = None
        self.conf = conf
        self.logger = logging.getLogger(__name__)
        self.mq_manager = KombuQueueManager()
        self.client_id = "client_" + str(uuid.uuid1())

        self.logger.info("Init client with id: %s" % self.client_id)

    def run(self):
        with Connection('amqp://%s:%s@%s:%s//' % (self.conf.get_amqp_user(), self.conf.get_amqp_password(),
                                                  self.conf.get_amqp_host(), self.conf.get_amqp_port())) as conn:
            with conn.Consumer(self.mq_manager.get_client_queue(self.client_id),
                               callbacks=[self.message_handler]) as consumer:

                self.logger.info("Successfully connected to MQ")
                self.conn = conn
                self.action_handle()
                conn.drain_events()

    def action_handle(self):
        action_name = self.conf.get(KEY_ACTION)

        if action_name == KEY_ACTION_SUBMIT_JOB:
            self.submit_job()
        elif action_name == KEY_ACTION_GET_JOB:
            self.get_job()
        elif action_name == KEY_ACTION_KILL_JOB:
            self.kill_job()
        elif action_name == KEY_ACTION_DUMP_LOG:
            self.dump_log()
        elif action_name == KEY_ACTION_CHANGE_CONFIG:
            self.change_config()
        else:
            raise Exception("unknow action name %s" % (action_name))

    def submit_job(self):
        script = self.conf.get(KEY_SCRIPT)
        script_args_list = self.conf.get_remain()
        target_loss = float(self.conf.get_target_loss())
        ml_model = str(self.conf.get_ml_model())
        batch_size = int(self.conf.get_batch_size())
        learning_rate = float(self.conf.get_learning_rate())

        optimizer = str(self.conf.get(KEY_OPTIMIZER))
        num_worker = int(self.conf.get(KEY_NUM_WORKER))
        num_intra_thread = int(self.conf.get(KEY_NUM_INTRA_THREAD))
        n_partition = int(self.conf.get(KEY_n_partition))
        mode = str(self.conf.get(KEY_MODE))

        json_dict_pstuner_config = str(self.conf.get(KEY_JSON_DICT_PSTUNER_CONFIG))

        msg = StartJob.create(self.client_id, self.mq_manager.get_monitor_name(), script, script_args_list,
                              target_loss=target_loss, ml_model=ml_model, batch_size=batch_size, learning_rate=learning_rate,
                              optimizer=optimizer, num_worker=num_worker, num_intra_thread=num_intra_thread,
                              n_partition=n_partition, mode=mode, json_dict_pstuner_config=json_dict_pstuner_config)
        self.mq_manager.send_msg_to_monitor(self.conn, msg)

    def get_job(self):
        args_list = self.conf.get_remain()
        job_id = ""
        if not len(args_list) == 0:
            job_id = args_list[0]

        msg = GetJobStatistic.create(self.client_id, self.mq_manager.get_monitor_name(), job_id)
        self.mq_manager.send_msg_to_monitor(self.conn, msg)

    def message_handler(self, body, mq_message):
        try:
            message = json.loads(body, object_hook=MessageSerializer.message_object_hook)
            self.message_logic(message)
            mq_message.ack()
        except Exception as e:
            self.logger.exception(e)

    def message_logic(self, message_obj):
        if isinstance(message_obj, ReplyStartJob):
            self.handle_reply_start_job(message_obj)
        elif isinstance(message_obj, ReplyGetJobStatistic):
            self.handle_reply_get_job_statistic(message_obj)
        else:
            raise Exception("Can't handle message: %s" % message_obj.__class__.__name__)

    def handle_reply_start_job(self, message_obj):
        util.check_type(message_obj, ReplyStartJob)
        # print the job id
        logging.info("The job id is: %s" % message_obj.get_job_id())

    def handle_reply_get_job_statistic(self, message_obj):
        util.check_type(message_obj, ReplyGetJobStatistic)

        # print to csv
        file_name = "%s_job_statistic.csv" % message_obj.job_id
        self.pritn_job_statistic_to_csv(message_obj.get_job_statistic(), file_name)
        logging.info("statistic is printed to file: %s" % file_name)

    def pritn_job_statistic_to_csv(self, list_training_data, csv_file_name):
        """
        :param list[PSTunerTrainingData] list_training_data:
        :param file csv_file:
        :return:
        """
        logging.debug(list_training_data)
        with open(csv_file_name, 'w+') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            # write header
            header_row = [
                "epoch",
                "n_ps",
                "n_worker",
                "intra_op_parallelism_threads",
                "inter_op_parallelism_threads",
                "loss",
                "epoch_time"
            ]
            writer.writerow(header_row)
            for ps_tuner_training_data in list_training_data:
                row = [
                    ps_tuner_training_data.step,
                    ps_tuner_training_data.ps_config.ps_num,
                    ps_tuner_training_data.ps_config.worker_num,
                    ps_tuner_training_data.ps_config.intra_op_parallelism_threads,
                    ps_tuner_training_data.ps_config.inter_op_parallelism_threads,
                    ps_tuner_training_data.loss,
                    ps_tuner_training_data.elapsed_time_in_ms
                ]
                writer.writerow(row)

    def kill_job(self):
        job_id = self.conf.get_remain()[0]
        msg = KillJob.create(self.client_id, self.mq_manager.get_monitor_name(), job_id)
        self.mq_manager.send_msg_to_monitor(self.conn, msg)

    def dump_log(self):
        job_id = self.conf.get_remain()[0]
        msg = TriggerDumpLog.create(self.client_id, self.mq_manager.get_monitor_name(), job_id)
        self.mq_manager.send_msg_to_monitor(self.conn, msg)
        sys.exit(0)

    def change_config(self):
        job_id = self.conf.get_remain()[0]
        num_node = int(self.conf.get_remain()[1])
        # test
        msg = ChangeConfig.create(self.client_id, self.mq_manager.get_monitor_name(), job_id, num_node)
        self.mq_manager.send_msg_to_monitor(self.conn, msg)
        sys.exit(0)


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    client = Client(get_config())
    client.run()
