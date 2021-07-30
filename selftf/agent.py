# Process Management
# Accept connection monitor
# Thread:
# - main Thread
# - listen Thread (listen on specific queue)
import os
import re
import threading
import signal

import time

import kombu.mixins

from selftf.lib.common import Process, ComputeNode, TF_PROGRAM_JOB_NAME, \
    PSTunerTrainingData
import logging

from kombu import Connection

from selftf.lib.common_conf import Config, get_config, KEY_AGENT_NUMBER_OF_THREADS
from selftf.lib.queue import KombuQueueManager, ConsumerMixin
from selftf.lib.message import *


class AgentConsumerMixin(kombu.mixins.ConsumerMixin):
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
        return Consumer(self.queue, callbacks=[self.message_handler])


class Agent:
    _key_status = ["waiting_for_connection", "connected"]

    stdout_step_regex = "Step:\\s*(\\d+)"
    stdout_timer_regex = "Time:\\s*(\\d+.\\d+)"
    stdout_loss_regex = "Loss:\\s*(\\d+.\\d+)"
    stdout_timestamp_regex = "Timestamp:\\s*(\\d+.\\d+)"

    re_stdout_timer_regex = re.compile(stdout_timer_regex)
    re_stdout_loss_regex = re.compile(stdout_loss_regex)
    re_stdout_step_regex = re.compile(stdout_step_regex)
    re_stdout_timestamp_regex = re.compile(stdout_timestamp_regex)

    stdout_iteration_statistic_regex = "%s.*%s.*%s.*%s" % (stdout_step_regex, stdout_loss_regex,
                                                           stdout_timer_regex, stdout_timestamp_regex)
    re_stdout_iteration_statistic_regex = re.compile(stdout_iteration_statistic_regex)
    re_stdout_iteration_statistic_step_id = 1
    re_stdout_iteration_statistic_loss_id = 2
    re_stdout_iteration_statistic_timer_id = 3
    re_stdout_iteration_statistic_timestamp_id = 4

    def __init__(self, conf):

        if not isinstance(conf, Config):
            raise TypeError()

        self.mq_manager = KombuQueueManager()
        self.hostname = conf.get_hostname()
        self.tfport = conf.get_agent_tfport()
        self.conf = conf
        self.is_stop = False
        self.logger = logging.getLogger(__name__)
        self.conn = None
        self.root_workspace = conf.get_agent_workspace()
        self.compute_node_obj = ComputeNode(self.hostname, self.tfport, working_dir=conf.get_agent_workspace())

        # integrate regex
        self.process_manager = ProcessManager(self.compute_node_obj.get_id(),
                                              finish_process_handler=self.handle_process_finish,
                                              per_step_handler=self.handle_per_step,
                                              stdout_step_regex=self.stdout_step_regex,
                                              stdout_timer_regex=self.stdout_timer_regex,
                                              stdout_loss_regex=self.stdout_loss_regex,
                                              re_stdout_iteration_regex=self.re_stdout_iteration_statistic_regex)
        self.stats = self._key_status[0]

        self.update_per_step_time_limit_sec = 5

        self.consumer = None

        # hook clean up
        signal.signal(signal.SIGINT, self.exit_clean_up)
        signal.signal(signal.SIGTERM, self.exit_clean_up)

    def run(self):
        self.logger.info("Start connecting to server, %s:%s" % (self.conf.get_amqp_host(), self.conf.get_amqp_port()))
        with Connection('amqp://%s:%s@%s:%s//' % (self.conf.get_amqp_user(), self.conf.get_amqp_password(),
                                                  self.conf.get_amqp_host(), self.conf.get_amqp_port())) as conn:
            self.logger.info("Successfully connected to MQ")

            self.conn = conn

            # register itself to monitor
            self.register_to_monitor()

            self.consumer = ConsumerMixin(conn, message_handler=self.message_handler,
                                     queue=self.mq_manager.get_compute_node_queue(self.compute_node_obj))

            try:
                self.consumer.run()
            except KeyboardInterrupt:
                self.exit_clean_up()
            except:
                self.logger.exception("Something wrong")

    def exit_clean_up(self, sig=2, stack=None):
        self.logger.info("Exit clean up")
        deregister_msg = DeRegisterComputeNode.create(self.compute_node_obj.get_id(),
                                                      self.mq_manager.get_monitor_name())
        self.mq_manager.send_msg_to_monitor(self.conn, deregister_msg)
        self.stop()

        self.process_manager.monitor_thread_clean_up()

        self.consumer.should_stop = True

    def register_to_monitor(self):
        # construct message
        msg = RegisterAgent.create(self.compute_node_obj.get_id(),
                                   self.mq_manager.get_monitor_name(),
                                   self.tfport,
                                   work_dir=self.compute_node_obj.get_working_dir(),
                                   num_thread=int(self.conf.get(KEY_AGENT_NUMBER_OF_THREADS)))
        # send to monitor
        self.mq_manager.send_msg_to_monitor(self.conn, msg)

    def message_handler(self, body, mq_message):
        try:
            message = json.loads(body, object_hook=MessageSerializer.message_object_hook)
            mq_message.ack()
            self.logger.debug("Receive msg from %s type: %s" % (message.get_source(), message.get_message_type()))
            self.message_logic(message, mq_message)
        except Exception as e:
            self.logger.exception("failt to process message")

    def message_logic(self, message_obj, mq_message):
        if isinstance(message_obj, StartProcess):
            self.start_process_handler(message_obj)

        if isinstance(message_obj, KillProcessByJobId):
            self.kill_process_handler(message_obj)

    def start_process_handler(self, message_obj):
        if not isinstance(message_obj, StartProcess):
            raise TypeError()

        job_id = message_obj.get_job_id()
        args = message_obj.get_args()
        script = message_obj.get_script_location()
        training_status = message_obj.get_training_status()
        config_idx = message_obj.get_config_idx()

        self.logger.info("Start process script:'%s' args:'%s' " % (script, args))

        self.process_manager.spawn_process(job_id, script, args, training_status, config_idx=config_idx,
                                           env=message_obj.env)

    def kill_process_handler(self, message_obj):
        if not isinstance(message_obj, KillProcessByJobId):
            raise TypeError()

        job_id = message_obj.get_job_id()

        # remove monitor thread
        self.process_manager.kill_by_jobid(job_id)

        # if not process.is_ps:
        #     self.send_finish_process_msg_to_monitor(process, "")

    def stop(self):
        self.is_stop = True
        self.process_manager.stop()

    def get_create_space_by_job_id(self, job_id):
        job_path = os.path.join(self.root_workspace, job_id)
        if not os.path.exists(job_path):
            os.makedirs(job_path)
        return job_path

    def handle_process_finish(self, process):
        training_data_list = []
        # try:
        #     training_data_list = self.tranverse_proceess_stdout_to_training_data(process)
        # except Exception as e:
        #     if process.is_ps:
        #         training_data_list = []
        #     else:
        #         self.logger.exception("Error when parsing stdout of worker")
        #         raise e
        print(process.get_popen())
        json_training_data = json.dumps(training_data_list, cls=PSTunerTrainingDataSerializer)
        self.logger.debug("Finish handler collected statistic %s" % (json_training_data))

        self.send_finish_process_msg_to_monitor(process=process, json_training_data=json_training_data)

    def send_finish_process_msg_to_monitor(self, process, json_training_data=""):
        msg = FinishProcess.create(self.compute_node_obj.get_id(), self.mq_manager.get_monitor_name(),
                                   process.get_job_id(), ps_tuner_training_data_jsons=json_training_data)
        self.mq_manager.send_msg_to_monitor(self.conn, msg)

    def handle_per_step(self, monitor_thread):
        """
        :param MonitoringThread monitor_thread:
        :param steps:
        :param loss:
        :param time:
        :param timestamp:
        :return:
        """
        ps_tuner_training_data = self.polling_tranverse_proceess_stdout_to_training_data(monitor_thread.process_obj)

        if len(ps_tuner_training_data) > 0:
            msg = UpdateNumOfSteps.create(self.compute_node_obj.get_id(),
                                          self.mq_manager.get_monitor_name(), ps_tuner_training_data[-1].local_step,
                                          monitor_thread.get_job_id(),
                                          ps_tuner_training_data_jsons=json.dumps(ps_tuner_training_data,
                                                                                  cls=PSTunerTrainingDataSerializer))

            self.mq_manager.send_msg_to_monitor(self.conn, msg)

    def polling_tranverse_proceess_stdout_to_training_data(self, p):
        """
            :param Process p:
            :return:
            :rtype: list[tuner.PSTunerTrainingData]
        """
        ret = []
        begin_time = time.time()
        logging.debug("Start wait polling stdout of subprocess")
        while time.time() - begin_time < self.update_per_step_time_limit_sec:
            logging.debug("read a line")
            line = p.get_popen().readline()
            ret.append(self.line_to_ps_training_data(line, config_idx=p.config_idx))
        return ret

    def tranverse_proceess_stdout_to_training_data(self, p):
        """
        :param limit:
        :param Process p:
        :return:
        :rtype: list[tuner.PSTunerTrainingData]
        """
        ret = []
        if not isinstance(p, Process):
            raise TypeError()
        for line in p.get_popen().readlines():
            print(line)
            x = self.line_to_ps_training_data(line, config_idx=p.config_idx)
            if x is not None:
                ret.append(x)

        return ret

    def line_to_ps_training_data(self, line, config_idx):
        iter_time = None
        loss = None
        step = None
        # self.logger.debug(line)
        m_stdout = self.re_stdout_timer_regex.search(line)
        if m_stdout is not None:
            iter_time = float(m_stdout.group(1))

        m_loss = self.re_stdout_loss_regex.search(line)
        if m_loss is not None:
            loss = float(m_loss.group(1))

        m_step = self.re_stdout_step_regex.search(line)
        if m_step is not None:
            step = int(m_step.group(1))

        m_timestamp = self.re_stdout_timestamp_regex.search(line)
        if m_timestamp is not None:
            timestamp = float(m_timestamp.group(1))

        if iter_time is not None and loss is not None and step is not None:
            self.logger.debug("Extracted statistic step: %d, time: %f, loss: %f" % (step, iter_time, loss))
            return PSTunerTrainingData(
                elapsed_time_in_ms=iter_time,
                loss=loss,
                local_step=step,
                timestamp=timestamp,
                ps_config_idx=config_idx
            )
        return None


class ProcessSet(set):

    def get_processes_by_jobid(self, jobid):
        for x in self:
            if not isinstance(x, Process):
                raise TypeError()
            if x.get_job_id() == jobid:
                return x
        raise Exception("can not find process object with job_id")

    def is_finish_by_job_id(self, job_id):
        return self.get_processes_by_jobid(job_id).is_finished()


class ProcessManager():
    def __init__(self, compute_node_id, finish_process_handler, per_step_handler, stdout_step_regex, stdout_timer_regex,
                 stdout_loss_regex, re_stdout_iteration_regex):

        self.logger = logging.getLogger(__name__)
        self.process_list = ProcessSet()
        self.background_thread = None
        self.is_stop = False
        self.compute_node_id = compute_node_id

        self.finish_process_handler = finish_process_handler
        self.per_step_handler = per_step_handler

        self.stdout_step_regex = re.compile(stdout_step_regex)
        self.stdout_timer_regex = re.compile(stdout_timer_regex)
        self.stdout_loss_regex = re.compile(stdout_loss_regex)

        self.re_stdout_iteration_regex = re_stdout_iteration_regex

        self._process_background_thread_check_interval_in_sec = 1

        self.monitor_threads = set()

    def kill_by_jobid(self, job_id):
        # close the monitoring thread first
        thread = self.get_monitor_thread_by_job_id(job_id)
        thread.stop()
        thread.join()
        logging.debug("Monitor thread is joined, finish message sent: %s" % thread.sent_finish_msg)
        if not thread.sent_finish_msg:
            logging.debug("Send finish process after monitor thread join")
            self.finish_process_handler(thread.process_obj)

    def get_monitor_thread_by_job_id(self, job_id):
        """
        :param job_id:
        :return:
        :rtype: MonitoringThread
        """
        for x in self.monitor_threads:
            if x.get_job_id() == job_id:
                return x

    def spawn_process(self, job_id, script, args, training_status, config_idx, env={}):
        env_local_default = os.environ.copy()
        env.update(env_local_default)

        # logging.debug("ENV: {}".format(env))

        # add agent_id and agent_config_idx to args
        args.append("--agent_id=" + self.compute_node_id)
        args.append("--agent_config_idx=" + str(config_idx))
        args.append("--agent_job_id=" + job_id)
        p = Process.create(job_id, script, args, training_status, is_ps=self.check_args_is_ps(args),
                           config_idx=config_idx, env=env)

        self.process_list.add(p)

        monitor = MonitoringThread(job_id, p, self)
        self.monitor_threads.add(monitor)
        monitor.start()
        return p

    def check_args_is_ps(self, args):
        """
        :param list[string] args:
        :return:
        """
        for arg in args:
            if arg == "--%s=%s" % (TF_PROGRAM_JOB_NAME, "ps"):
                return True

        return False

    def monitor_thread_clean_up(self):
        pending_clean_thread = []
        for x in self.monitor_threads:
            if not x.is_alive():
                pending_clean_thread.append(x)

        for x in pending_clean_thread:
            self.monitor_threads.remove(x)
            x.join()

    # get the step from line
    def step_checker(self, monitor_thread):
        self.per_step_handler(monitor_thread)

    def stop(self):
        self.is_stop = True
        for x in self.monitor_threads:
            x.stop()


class MonitoringThread(threading.Thread):
    def __init__(self, job_id, process_obj, process_manager):
        if not isinstance(process_obj, Process) and not isinstance(process_manager, ProcessManager):
            raise TypeError()
        self.job_id = job_id
        self.process_obj = process_obj
        self.process_manager = process_manager
        self.is_stop_active = False
        self.sent_finish_msg = False  # hijacking,,
        super(MonitoringThread, self).__init__(target=self.run)

    def run(self):
        # make the process stdout as non block
        while not self.process_obj.is_finished() and not self.is_stop_active:
            # if self.process_obj.get_training_status() == TRAINING_STATUS_OPTIMAL_RUN and not self.process_obj.is_ps:
            # for online tuning phase
            # self.process_manager.step_checker(self)
            time.sleep(0.1)

        logging.debug("is_finished: %s, is_stop_active:%s" % (self.process_obj.is_finished(), self.is_stop_active))

        # ProcessSet cleanup
        try:
            logging.debug("explicit kill %d" % self.process_obj.get_popen().pid)
            self.process_obj.get_popen().terminate()

            # # debug
            # for line in self.process_obj.get_popen().stdout:
            #     logging.debug("after kill println: %s", line)  # debug

            # wait process being kill
            counter = 10
            while True:
                if counter <= 0:
                    self.process_obj.get_popen().kill()
                    break
                if self.process_obj.get_popen().poll() is not None:
                    break
                time.sleep(0.5)
                counter -= 1
        except:
            pass
        finally:
            logging.debug("job is_finished: %s" % self.process_obj.is_finished())
            self.process_manager.process_list.remove(self.process_obj)
            self.process_manager.monitor_threads.remove(self)
            self.process_manager.finish_process_handler(self.process_obj)
            self.sent_finish_msg = True

    def stop(self):
        logging.debug("Stop the thread actively before : %s, is_finished: %s" % (
        self.is_stop_active, self.process_obj.is_finished()))
        self.is_stop_active = True

    def get_job_id(self):
        return self.job_id

    def __hash__(self):
        return self.job_id.__hash__()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        )
    logging.getLogger("amqp").setLevel(logging.INFO)

    agent = Agent(get_config())
    agent.run()
