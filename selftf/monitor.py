# AMP
# Accept tensorFlow ml_program
# Engine Metrics --> Plan
# schedule Logical Plan -->  Execution operation

# communication through AMPQ (RabbitMQ)

# Logic component:
# - ComputeNode
#   - MachineId (hostname)
# - Execution operation
#   -
# - PSTunerTrainingData
#   - The attribute is not confirm yet
# - PSTunerConfiguration
#   - The attribute is not confirm yet
# - Job
#   - jobId
#   - script
#   - args
# - Process
#   - jobID
#   - execute cmd
# - Control Operation(Message) (like iTuned table2)
#   - startCmd
#       - cmd
#   - getFileToLocal
#
# - Message
#   - from
#   - to <-- destination queue

# API:
# - External:
#   - Submit Job
#   - End Job
# - Internal
#   - Pause Job
#   - Resume Job(configuration)
#       - Compare last configuration and new configuration

# Consider scenario
# - start new ML Job
#   1. user submit the script to monitor
#   2. monitor send Control Operation to agent
#   3. agent pull script from ssh to fileToLocal
#   4. (optional: if we have dataset) agent pull data
#   5. startCmd in every ComputeNode
# - reconfig existing ML Job
#   1. Given JobId, stop corresponding process
#   2. according to the new configuration, ask PS to pull the chkpt data
#   3. send startCmd to restart the ML Job

# Thread
# - Main thread
# - API thread (keep on listen to master queue)

# Queue
# - master queue
# - slave queue
import csv
import multiprocessing
import shutil
import threading

import tensorflow as tf
import signal
from kombu import Connection

import selftf.lib.common
from selftf.lib.common import *
from selftf.lib.common_conf import *
from selftf.lib.mltuner import mltuner
from selftf.lib.mltuner.mltuner import MLTuner
from selftf.lib.queue import KombuQueueManager, ConsumerMixin
from selftf.lib.tuner import TensorFlowConfigurationManager
from selftf.lib.common import PSTunerConfiguration
from selftf.lib.util import check_type, TFVariableSeralizer
from selftf.lib.message import *
from selftf.lib import web_ui, common

import os
import psutil
import subprocess
import pymysql

class SubProcess:

    def __init__(self, tf_config_manager):

        self.main_process_conn, child_conn = multiprocessing.Pipe()

        def run(tf_config_manager):
            assert isinstance(tf_config_manager, TensorFlowConfigurationManager)
            job_obj = child_conn.recv()
            logging.debug("I receive a job:%s" % str(job_obj))
            while job_obj is not None:
                try:
                    setting = tf_config_manager.get_tf_config_by_gp(job_obj)
                except:
                    logging.exception("BO checking thread fail")
                    time.sleep(0.5)
                    setting = None
                child_conn.send(setting)
                job_obj = child_conn.recv()

        self.process = multiprocessing.Process(target=run, args=[tf_config_manager])
        self.process.start()

    def join(self):
        # class by Main thread
        self.main_process_conn.send(None)
        self.process.join()

class Monitor:
    MSG_SOURCE_KEY = "monitor"

    def __init__(self, conf):

        if not isinstance(conf, Config):
            raise TypeError()

        self.conf = conf
        self.is_stop = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.conn = None

        self.compute_node_manager = ComputeNodeManager()
        self.job_manager = JobManager()
        self.mq_manager = KombuQueueManager()

        self.ml_job_finish_stragegy = ML_JOB_FINISH_STRATEGY_MOVING_AVG

        batch_size_range = conf.get_batch_size_range()
        learning_rate_range = conf.get_learning_rate_range()

        self.tf_config_manager = TensorFlowConfigurationManager(self.compute_node_manager.get_num_node,
                                                                self.compute_node_manager.get_num_thread_of_node,
                                                                learning_rate_range=learning_rate_range,
                                                                batch_size_range=batch_size_range)
        self.consumer = None

        # A thread for running
        self.dict_job_subprocess = {}
        self.dict_job_lock = {}

        self.mltuner:MLTuner = None

        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

    @property
    def ml_job_finish_threshold(self):
        return 0.9

    def run(self):
        self.logger.info("Start connecting to server, %s:%s" % (self.conf.get_amqp_host(), self.conf.get_amqp_port()))
        with Connection('amqp://%s:%s@%s:%s//' % (self.conf.get_amqp_user(), self.conf.get_amqp_password(),
                                                  self.conf.get_amqp_host(), self.conf.get_amqp_port())) as conn:
            self.logger.info("Successfully connected to MQ")
            self.conn = conn

            momnitor_queue = self.mq_manager.get_monitor_queue()
            self.consumer = ConsumerMixin(connection=conn, message_handler=self.message_handler,
                                    queue=momnitor_queue)
            while not self.is_stop:
                try:
                    self.consumer.run()
                except KeyboardInterrupt:
                    self.logger.info("Exit clean up")
                    self.stop()
                except:
                    self.logger.exception("Error")

            # cleanup
            # with conn.Consumer(self.mq_manager.get_monitor_queue(), callbacks=[self.message_handler]) as consumer:
            #     consumer.purge()
            #     while not self.is_stop:
            #         try:
            #             conn.drain_events()
            #         except KeyboardInterrupt:
            #             self.logger.info("Exit clean up")
            #             self.stop()
            #         except:
            #             self.logger.exception("Error")
            #             try:
            #                 consumer.recover()
            #             except:
            #                 pass

    def message_handler(self, body, mq_message):
        try:
            try:
                mq_message.ack()
            except:
                self.logger.exception("Can't ack the message, but keep going first")

            message = json.loads(body, object_hook=MessageSerializer.message_object_hook)
            self.logger.debug("Receive msg from %s type: %s" % (message.get_source(), message.get_message_type()))
            self.message_logic(message)
        except Exception as e:
            self.logger.exception(e)

    def message_logic(self, message_obj):
        if isinstance(message_obj, StartJob):
            self.start_job_handler(message_obj)
        elif isinstance(message_obj, RegisterAgent):
            self.register_agent_handler(message_obj)
        # elif isinstance(message_obj, AgentHeartBeat):
        #     self.agent_heart_beat_handler(message_obj)
        elif isinstance(message_obj, FinishProcess):
            self.finish_process_handler(message_obj)
        elif isinstance(message_obj, GetJobStatistic):
            self.get_job_statistic_handler(message_obj)
        elif isinstance(message_obj, UpdateNumOfSteps):
            self.update_num_of_steps_handler(message_obj)
        # elif isinstance(message_obj, FinishCopyRemoteFile):
        #     self.finish_copy_remote_file_handler(message_obj)
        elif isinstance(message_obj, KillJob):
            self.handle_kill_job(message_obj)
        elif isinstance(message_obj, DeRegisterComputeNode):
            self.handle_deregister_compute_node(message_obj)
        elif isinstance(message_obj, SendRecoveryTime):
            self.handle_send_recovery_time(message_obj)
        elif isinstance(message_obj, NaturalFinishML):
            self.handle_natural_finish_ml(message_obj)
        elif isinstance(message_obj, TriggerDumpLog):
            self.handle_dump_log(message_obj)
        elif isinstance(message_obj, FinshSubTrainingPhase):
            self.handle_finish_sub_training(message_obj)
        elif isinstance(message_obj, ChangeConfig):
            self.handle_change_config(message_obj)
        else:
            raise Exception("Can't handle message: %s" % message_obj.__class__.__name__)

    def update_num_of_steps_handler(self, message_obj):
        """
        :param UpdateNumOfSteps message_obj:
        :return:
        """
        job_id = message_obj.get_job_id()
        compute_node_id = message_obj.get_source()

        job_obj = self.job_manager.get_job(job_id)

        # add things to job training sample
        self.add_training_data_list_json_to_job(job_obj,
                                                message_obj.get_ps_tuner_training_data_jsons(),
                                                compute_node_id)

        # trigger model predition if iteration
        self.check_do_online_tuning(job_obj)

        # Check job finish
        self.check_finish_ml_job(job_obj)
        # if job_obj.check_do_reconfig():
        #     self.do_reconfig(job_obj)

    def check_do_online_tuning(self, job_obj):
        """
        :param Job job_obj:
        :return:
        """
        # if NaturalFinish is received, dun do online tuning
        if job_obj.is_receive_natural_finish():
            return

        # check current reconfig number
        if job_obj.training_status == TRAINING_STATUS_OPTIMAL_RUN:
            subprocess = self.dict_job_subprocess[job_obj]
            lock = self.dict_job_lock[job_obj]

            self.logger.debug("check_do_online_reconfig:%s" % job_obj.check_do_online_reconfig())
            if job_obj.check_do_online_reconfig():
                # add lock to job obj
                if lock.acquire(False):
                    assert isinstance(subprocess, SubProcess)
                    subprocess.main_process_conn.send(job_obj)
                else:
                    logging.debug("can't acquire lock for reconfig, do it next time")


            # to collect back best config
            if subprocess.main_process_conn.poll():
                next_conf = subprocess.main_process_conn.recv()
                if next_conf is not None:
                    job_obj.increment_counter_check_do_reconfig()
                    if job_obj.is_different_config(next_conf):

                        if job_obj.mode == MODE_VALUE_SELFTF_RECONFIG_ZERO:
                            logging.debug(
                                "get injectted config in online tuning phase")
                            try:
                                next_conf = self.tf_config_manager.get_tf_config_by_lhs(
                                    job_obj)
                            except:
                                logging.exception("Fail to inject config")
                                self.finish_online_tuning_reconfig(job_obj)
                                return

                        job_obj.increment_counter_do_reconfig()
                        logging.debug("We got a new config: %s\n"
                                      "old config: %s" % (
                                      json.dumps(next_conf.__dict__),
                                      json.dumps(
                                          job_obj.current_configuration_plan.__dict__)))
                        self.reconfig(job_obj, next_conf)
                    else:
                        self.finish_online_tuning_reconfig(job_obj)
                else:
                    logging.debug("Fail to do BO, skip it")
                    self.finish_online_tuning_reconfig(job_obj)

    def finish_online_tuning_reconfig(self, job_obj):
        lock = self.dict_job_lock[job_obj]
        try:
            lock.release()
        except:
            pass
        job_obj.done_check_online_reconfig()

    def get_ps_list_by_ps_tuner_configuration(self, ps_tuner_configuraiton):
        check_type(ps_tuner_configuraiton, PSTunerConfiguration)
        num_of_ps = ps_tuner_configuraiton.ps_num
        compute_node_list = self.compute_node_manager.get_compute_node_list()
        return compute_node_list[:num_of_ps]

    def get_worker_list_by_ps_tuner_configuration(self, ps_tuner_configuraiton):
        check_type(ps_tuner_configuraiton, PSTunerConfiguration)
        num_of_ps = ps_tuner_configuraiton.ps_num
        compute_node_list = self.compute_node_manager.get_compute_node_list()
        return compute_node_list[num_of_ps:]

    def dry_run_single_config(self, job_obj, start_job_message):
        """
        :param Job job_obj:
        :param StartJob start_job_message:
        :return:
        """
        logging.info("num_worker:{} num_node:{}".format(start_job_message.num_worker, self.compute_node_manager.get_num_node()))
        if start_job_message.num_worker > (self.compute_node_manager.get_num_node() - 1):
            raise Exception(f"The number of worker:{start_job_message.num_worker} > total number of compute node:{(self.compute_node_manager.get_num_node() - 1)}")
        learning_rate = start_job_message.learning_rate
        batch_size = start_job_message.batch_size
        optimizer_idx = get_optimizer_by_name(start_job_message.optimizer)
        number_worker = start_job_message.num_worker
        number_ps = self.compute_node_manager.get_num_node() - number_worker
        num_intra_thread = start_job_message.num_intra_thread
        num_inter_thread = self.compute_node_manager.get_num_thread_of_node() - num_intra_thread
        n_partition = start_job_message.n_partition

        job_obj.num_iter_per_os = sys.maxsize

        config = PSTunerConfiguration(
            num_ps=number_ps,
            num_worker=number_worker,
            optimizer=optimizer_idx,
            learning_rate=learning_rate,
            batch_size=batch_size,
            intra_op_parallelism_threads=num_intra_thread,
            inter_op_parallelism_threads=num_inter_thread,
            n_partition=n_partition
        )

        if start_job_message.json_dict_pstuner_config != "":
            dict_msg_config = json.loads(start_job_message.json_dict_pstuner_config)
            config.__dict__ = dict_msg_config

        self.logger.info("Dry Run: %s" % (str(config.__dict__)))

        self.start_job_by_config(ps_tuner_configuration=config,
                                 job_obj=job_obj)


    def start_job_handler(self, message_obj):
        if not isinstance(message_obj, StartJob):
            raise TypeError()
        stop_config_idx = -1
        if message_obj.mode == MODE_VALUE_TEST_ESTIMATION_FUNC_REMAINING_TIME:
            stop_config_idx = get_config().get(KEY_TEST_ESTIMATION_FUNC_CONFIG_IDX)

        job = self.job_manager.create_job(message_obj.get_script_path(), message_obj.get_args(),
                                          int(self.conf.get(KEY_RECONFIG_NUM_ITER_EACH_OS)),
                                          int(self.conf.get(KEY_RECONFIG_OS_SIZE)),
                                          target_loss=message_obj.get_target_loss(),
                                          ml_model=message_obj.get_ml_model(),
                                          batch_size=message_obj.get_batch_size(),
                                          learning_rate=message_obj.get_learning_rate(),
                                          optimizer=get_optimizer_by_name(message_obj.optimizer),
                                          online_os_size=int(self.conf.get(KEY_ONLINE_RECONFIG_OS_SIZE)),
                                          online_num_iter_per_os=int(
                                              self.conf.get(KEY_ONLINE_RECONFIG_NUM_ITER_PER_OS)),
                                          estimation_func=self.conf.get(KEY_ESTIMATION_FUNC),
                                          mode=message_obj.mode,
                                          compute_node_id_list=list(map(lambda x:x.get_id(), self.compute_node_manager.get_compute_node_list())),
                                          test_job_config_idx=stop_config_idx)

        self.dict_job_lock[job] = threading.Lock()

        self.logger.debug("Mode config: "+message_obj.mode)
        self.logger.debug(list(map(lambda x:x.get_id(), self.compute_node_manager.get_compute_node_list())))
        
        if message_obj.mode == MODE_VALUE_DRY_RUN:
            self.dry_run_single_config(job, message_obj)
        elif message_obj.mode == MODE_VALUE_TEST_RECONFIGSCHEME_ONE:
            raise Exception("Deprecated")
        elif message_obj.mode == MODE_VALUE_TEST_RECONFIGSCHEME_ZERO or message_obj.mode == MODE_VALUE_TEST_RECONFIGSCHEME_TWO:
            self.logger.info("Test mode with fixed config")
            self.dry_run_single_config(job, message_obj)
        elif message_obj.mode == MODE_VALUE_MLTUNER:
            self.logger.debug("Start job with MLTuner mode")
            self.mltuner = MLTuner(self.conf.get_repo_dir(),
                                   self.tf_config_manager,
                                   job)
            hw_info = {
                'num_node': int(os.getenv("SELFTF_NUM_COMPUTE_NODE")),
                'num_cpu_per_node': message_obj.num_cpu_per_node,
                'num_gpu_per_node': message_obj.num_gpu_per_node,
                'num_mem_per_node': message_obj.num_mem_per_node
            }
            next_config = self.mltuner.get_next_config(
                hw_info,
                self.conf.get_mysql_account(), 
                self.conf.get_mysql_password(), 
                self.conf.get_mysql_dbname()
            )

            if next_config is None:
                logging.error("Something wrong with mltuner get_next_config, we "
                              "should at least run once")
            self.start_job_by_config(next_config, job)
        else:
            init_conf = self.tf_config_manager.get_tf_config_by_lhs(job)
            self.start_job_by_config(init_conf, job)

        client_id = message_obj.get_source()
        reply_msg = ReplyStartJob.create(self.mq_manager.get_monitor_name(), client_id, job.get_id())
        self.mq_manager.send_msg_to_client(self.conn, reply_msg)

        # Send start
        web_ui.add_job(job_id=job.id)
        job.change_status(common._JOB_STATUS_initialize)
        web_ui.change_status(job.id, 0, common._JOB_STATUS_initialize)

    def register_agent_handler(self, message_obj):
        if not isinstance(message_obj, RegisterAgent):
            raise TypeError()

        hostname, tfport = ComputeNode.get_hostname_tfport_from_id(message_obj.get_source())
        self.compute_node_manager.register_compute_node(hostname, tfport, work_dir=message_obj.work_dir,
                                                        num_thread=message_obj.num_thread)

    # def agent_heart_beat_handler(self, message_obj):
    #     if not isinstance(message_obj, AgentHeartBeat):
    #         raise TypeError()
    #
    #     self.compute_node_manager.touch_compute_node(message_obj.get_source())

    # def finish_copy_remote_file_handler(self, message_obj):
    #     if not isinstance(message_obj, FinishCopyRemoteFile):
    #         raise TypeError()
    #     job_id = message_obj.get_job_id()
    #     job_obj = self.job_manager.get_job(job_id)
    #     job_obj.finish_a_ssh_task()
    #
    #     if job_obj.get_wait_for_ssh_complete() == 0:
    #         config = job_obj.next_configuraton_plan
    #         self.start_job_by_config(config, job_obj)

    def finish_process_handler(self, message_obj):
        check_type(message_obj, FinishProcess)
        job_id = message_obj.get_job_id()
        job_obj = self.job_manager.get_job(job_id)
        if job_obj is None:
            # Job disappear
            self.logger.info("Receive msg with job_id %s. ignore it" % message_obj.get_job_id())
            pass
        else:
            compute_node_id = message_obj.get_source()
            job_obj.compute_node_finish(compute_node_id)

            # batch insert training data for initialization phase
            # if job_obj.is_worker(compute_node_id):
            #     self.logger.debug("Insert training record from compute node %s, %s" % (message_obj.get_source(),
            #                                                                            message_obj.get_ps_tuner_training_data_jsons()))
            #     self.add_training_data_list_json_to_job(job_obj, message_obj.get_ps_tuner_training_data_jsons())

            # Finish Hooking
            # Check if we worker process in job done
            # If yes finish the job and clean
            # if job_obj.is_all_non_chief_worker_finish() and job_obj.get_training_status() == TRAINING_STATUS_OPTIMAL_RUN and not job_obj.is_chief_killed():
            #     # Check finish checkpointing
            #     self.logger.debug("kill chief worker")
            #     # self.trigger_chkpt(job_obj)
            #     self.kill_chief_workers_by_job(job_obj)
            #     job_obj.kill_chief()
            if job_obj.is_all_non_chief_worker_finish() and job_obj.get_training_status() == TRAINING_STATUS_TRAINING and not job_obj.is_chief_killed():
                self.logger.debug("Initialize phase, trigger do checkpoint")
                # self.trigger_chkpt(job_obj)
                job_obj.kill_chief()

            if job_obj.is_all_worker_finish() and not job_obj.is_ps_killed:
                self.logger.debug("kill ps ")
                self.kill_all_ps_job(job_obj)
                job_obj.kill_ps()

            # if job_obj.is_all_worker_ps_finish():
            #     if job_obj.get_training_status() == TRAINING_STATUS_TRAINING:
            #         if job_obj.check_do_init_reconfig():
            #             conf = self.training_draw_next_configuration(job_obj)
            #         else:
            #             job_obj.finish_training()
            #             online_os_size = self.conf.get_online_os_size()
            #             conf = self.tf_config_manager.get_tf_config_by_gp(job_obj)
            #         # self.do_reconfig_pre_process(job_obj, conf=conf) # for ssh only we use HDFS now
            #         self.start_job_by_config(conf, job_obj)
            #     else:
            #         if job_obj.has_next_configuration():
            #             self.start_job_by_config(job_obj.next_configuraton_plan, job_obj)
            #         else:
            #             # ML job finish
            #             job_obj.set_end_time()
            if job_obj.is_all_worker_ps_finish():
                # if job_obj.mode == MODE_VALUE_MLTUNER:
                #
                #     new_training_data = PSTunerTrainingData(
                #         job_obj.get_current_config(),
                #         elapsed_time_in_ms=self.mltuner.get_config_duration(),
                #         loss=job_obj.get_avg_list_final_n_machine_loss()
                #     )
                #     self.mltuner.add_config_result(
                #         new_training_data
                #     )
                #     next_config = self.mltuner.get_next_config()
                #
                #     # let mltuner to control whether the best config obtain
                #     if next_config is None:
                #         logging.info("No next configuraiton")
                #         logging.info("Total runtime: %f",
                #                      job_obj.get_duration())
                #         return
                #     job_obj.end_time = None # hack to allow execute next job
                #     self.start_job_by_config(next_config,
                #                              job_obj)
                #     return

                if job_obj.has_next_configuration():
                    self.start_job_by_config(job_obj.next_configuraton_plan, job_obj)
                    job_obj.change_status(common._JOB_STATUS_initialize)
                    web_ui.change_status(job_id=job_id,
                                         x_value=time.time() - job_obj.start_time,
                                         status=common._JOB_STATUS_initialize)
                else:
                    logging.info("No next configuraiton")
                    job_obj.finish() # record endtime
                    logging.info("Total runtime: %f", job_obj.get_duration())

                    job_obj.change_status(common._JOB_STATUS_finish)
                    web_ui.change_status(job_id, time.time() - job_obj.start_time, common._JOB_STATUS_finish)
                    self.dump_job_obj(job_obj)
                    hw_info = {
                        'num_node': int(os.getenv("SELFTF_NUM_COMPUTE_NODE")),
                        'num_cpu_per_node': message_obj.num_cpu_per_node,
                        'num_gpu_per_node': message_obj.num_gpu_per_node,
                        'num_mem_per_node': message_obj.num_mem_per_node
                    }
                    self.dump_job_obj_to_db(job_obj, hw_info)

    def add_training_data_list_json_to_job(self, job_obj, training_data_list_json, compute_node_id):
        """
        :param Job job_obj:
        :param str training_data_list_json:
        :return:
        """
        object_list = json.loads(training_data_list_json)
        for training_data_json in object_list:
            training_data = selftf.lib.common.PSTunerTrainingData()
            training_data.__dict__ = training_data_json
            training_data.ps_config = job_obj.get_history_config_by_idx(training_data.ps_config_idx)
            job_obj.get_training_statistic().add_training_statistic(training_data)
            self.logger.debug(
                "Collected iteration statistic get local_step:%d, local_iteration_time:%f, timestamp:%f, loss:%f" %
                (training_data.local_step,
                 training_data.elapsed_time_in_ms,
                 training_data.timestamp,
                 training_data.loss))
            job_obj.update_steps(compute_node_id, training_data.local_step,
                                 training_data.loss, job_obj.target_loss)

            if job_obj.mode == MODE_VALUE_MLTUNER:
                self.mltuner.updateNumOfStepsHandler(training_data.loss,
                                                     training_data.local_step)

            if job_obj.get_status() == common._JOB_STATUS_executing:
                web_ui.add_pt(job_id=job_obj.id,
                              x_value=training_data.timestamp-job_obj.start_time,
                              y_value=training_data.loss)

    def training_draw_next_configuration(self, job_obj):
        return self.tf_config_manager.get_tf_config_by_lhs(job_obj)

    def kill_job(self, job_obj):
        check_type(job_obj, Job)
        compute_node_id_list = job_obj.get_compute_node_id_list()
        for compute_node_id in compute_node_id_list:
            self.kill_process_by_job(job_obj.get_id(), compute_node_id)

    def kill_all_ps_job(self, job_obj):
        check_type(job_obj, Job)
        ps_compute_node_id_list = job_obj.get_ps_compute_node_id_list()
        for compute_node_id in ps_compute_node_id_list:
            self.logger.debug("Kill PS: %s" % compute_node_id)
            self.kill_process_by_job(job_obj.get_id(), compute_node_id)

    def kill_workers_by_job(self, job_obj):
        """
        :param Job job_obj:
        :return:
        """
        ps_compute_node_id_list = job_obj.get_worker_compute_node_id_list()
        for compute_node_id in ps_compute_node_id_list:
            self.logger.debug("Kill Worker: %s" % compute_node_id)
            self.kill_process_by_job(job_obj.get_id(), compute_node_id)

    def stop(self, sig=2, stack=None):
        self.is_stop = True
        self.consumer.should_stop = True

        self.mltuner.close()

    def kill_process_by_job(self, job_id, compute_node_id):
        compute_node_obj = self.compute_node_manager.get_compute_node_by_id(compute_node_id)
        msg = KillProcessByJobId.create(self.mq_manager.get_monitor_name(), compute_node_id, job_id)
        self.mq_manager.send_msg_to_compute_node(self.conn, compute_node_obj, msg)


    """
    - Server - If live reconfiguration
        - Server -> agent live Reconfiguration
        - Run
    - Else
        - Server -> Kill
        - agent -> FinishProcessMessage
        - Server -> agent StartJob
        - Run
    """

    def get_dnn_setting(self, current_setting):
        # learning_rate: 0.045000 ,batch_size: 10 ,optimizer: RMSProp_Imagenet ,number_worker: 5 ,number_ps: 3 ,num_intra_thread: 8 ,num_inter_thread: 8
        setting1 = PSTunerConfiguration(
            learning_rate=0.045,
            batch_size=10,
            optimizer=5,
            num_ps=3,
            num_worker=5,
            intra_op_parallelism_threads=8,
            inter_op_parallelism_threads=8
        )
        # learning_rate: 0.045000 ,batch_size: 10 ,optimizer: RMSProp_Imagenet ,number_worker: 5 ,number_ps: 3 ,num_intra_thread: 6 ,num_inter_thread: 10
        setting2 = PSTunerConfiguration(
            learning_rate=0.045,
            batch_size=10,
            optimizer=5,
            num_worker=3,
            num_ps=5,
            intra_op_parallelism_threads=14,
            inter_op_parallelism_threads=2
        )
        if current_setting is None:
            return setting1
        if current_setting == setting1:
            return setting2
        return setting1

    def reconfig(self, job_obj, new_config):
        """
        :param Job job_obj:
        :param PSTunerConfiguration new_config:
        :return:
        """
        job_obj.set_next_configuration(new_config)
        self.logger.debug("Step:%d, Reconfig to config: %s" % (job_obj.get_global_step(), str(new_config)))
        # current_config = job_obj.get_current_config()
        # if not current_config.is_data_reallocateion_invole(new_config):
        #     self.trigger_reconfig_scheme1(job_obj, config=new_config)
        # else:
        #     # Follow old path
        #     # self.kill_non_chief_workers_by_job(job_obj)
        #     # Try reconfig scheme 2
        #     self.trigger_reconfig_scheme2(job_obj, config=new_config)
        if job_obj.mode == MODE_VALUE_TEST_RECONFIGSCHEME_ZERO or \
            job_obj.mode == MODE_VALUE_SELFTF_RECONFIG_ZERO:
            self.trigger_chkpt(job_obj)
            self.kill_non_chief_workers_by_job(job_obj)

        elif job_obj.mode == MODE_VALUE_TEST_ESTIMATION_FUNC_REMAINING_TIME:
            job_obj.get_current_config_id()
            # TODO: deduce start do test or not?
            # TODO: prevent the changes of configuraiton
            pass
        else:
            self.trigger_reconfig_scheme2(job_obj, config=new_config)


    def trigger_reconfig_scheme1(self, job_obj, config):
        """
        :param Job job_obj:
        :param PSTunerConfiguration config:
        :return:
        """
        # some job object manipulation
        job_obj.clear_config_round_info()
        current_config_idx = job_obj.set_current_config_plan(config)

        if job_obj.training_status == TRAINING_STATUS_TRAINING:
            max_iteration = job_obj.get_steps_for_reconfig()
        else:
            max_iteration = -1

        # send message and done
        for compute_node_id in job_obj.get_worker_compute_node_id_list():
            compute_node = self.compute_node_manager.get_compute_node_by_id(compute_node_id)
            msg = ReconfigScheme1.create(self.mq_manager.get_monitor_name(),
                                   self.mq_manager.get_compute_node_tf_routing_key(compute_node),
                                   job_id=job_obj.get_id(),
                                   conf_id=current_config_idx,
                                   conf_dict=config.__dict__,
                                   max_iteration=max_iteration,
                                   last_step=job_obj.get_global_step())
            self.mq_manager.send_msg_to_compute_node_tf(self.conn, compute_node, msg)
        job_obj.clear_config_round_info()

    def trigger_reconfig_scheme2(self, job_obj, config):
        # calculate new vairable map
        variable_map = VariableMapUtil.get_new_variable_map(job_obj, config,
                                                            self.get_master_ps_device_name())

        # some job object manipulation
        job_obj.clear_config_round_info()
        current_config_idx = job_obj.set_current_config_plan(config)
        self.logger.debug("current_config_idx:%s "%current_config_idx)
        job_obj.set_variable_map_plain(variable_map)

        if job_obj.training_status == TRAINING_STATUS_TRAINING:
            max_iteration = job_obj.get_steps_for_reconfig()
        else:
            max_iteration = -1

        conf_dict = config.__dict__.copy()

        conf_dict[conf_dict_variable_map] = variable_map

        # conf_dict[conf_dict_non_static_ops_names] = job_obj.list_non_static_ops_name
        # send message and done
        for compute_node_id in job_obj.get_compute_node_id_list():
            compute_node = self.compute_node_manager.get_compute_node_by_id(
                compute_node_id)
            msg = ReconfigScheme2.create(self.mq_manager.get_monitor_name(),
                                         self.mq_manager.get_compute_node_tf_routing_key(
                                             compute_node),
                                         job_id=job_obj.get_id(),
                                         conf_id=current_config_idx,
                                         conf_dict=conf_dict,
                                         max_iteration=max_iteration,
                                         last_step=job_obj.get_global_step(),
                                         target_reconfig_step=job_obj.get_global_step()+5)
            self.mq_manager.send_msg_to_compute_node_tf(self.conn, compute_node,
                                                        msg)
        job_obj.clear_config_round_info()

    # Should be start from Zero / Restarting
    def start_job_by_config(self, ps_tuner_configuration, job_obj):
        """
        :param PSTunerConfiguration ps_tuner_configuration:
        :param Job job_obj:
        :return:
        """
        if job_obj.is_finish():
            return

        if not isinstance(job_obj, Job):
            raise TypeError()
        if not isinstance(ps_tuner_configuration, PSTunerConfiguration):
            raise TypeError()

        cur_config_idx = job_obj.set_current_config_plan(ps_tuner_configuration)

        # clean next config
        job_obj.clear_config_round_info()

        program = job_obj.get_script()
        script_args = job_obj.get_args()
        logging.info(f"script_args:{script_args}")
        script = script_args[0]
        args = script_args[1:]
                    
        if job_obj.training_status == TRAINING_STATUS_TRAINING:
            max_iteration = job_obj.get_steps_for_reconfig()
        else:
            max_iteration = -1

        node_list = self.compute_node_manager.get_compute_node_list()
        node_list_args = self.get_args_node_list(node_list)

        conf_dict = ps_tuner_configuration.__dict__.copy()
        conf_dict[conf_dict_mode] = job_obj.mode

        common_args = []
        common_args.extend(args)
        common_args.extend(
            [
             "--" + TF_PROGRAM_NODE_LIST + "=" + str(node_list_args),
             "--" + TF_PROGRAM_MAX_ITERATION + "=" + str(max_iteration),
             "--" + TF_PROGRAM_CONF_DICT + "=" + json.dumps(conf_dict),
             "--" + TF_PROGRAM_TARGET_LOSS + "=" + str(job_obj.get_target_loss()),
             ])

        # For all node
        for idx, x in enumerate(node_list):
            dict_env = {}
            dict_env = mltuner.get_start_job_env(
                ps_tuner_configuration,
                node_list_args,
                idx,
                job_obj.get_id()
            )
            if idx != 0 and "--host-discovery-script" in common_args:
                continue # for mxnet distribute training
            if idx == len(node_list) - 1:
                # For chief worker
                job_obj.register_compute_node(x.get_id(), Job._ROLE_WORKER, is_chief=True)
                chief_args = [script]
                chief_args.extend(common_args)
                chief_args.extend(
                    ["--" + TF_PROGRAM_WORKING_DIR + "=" + os.path.join(x.get_working_dir(), job_obj.get_id()),
                    "--" + TF_PROGRAM_IS_CHIEF + "=True", "--%s=%d" % (TF_PROGRAM_TASK_INDEX, idx)]
                )
                              
                worker_msg = StartProcess.create(self.MSG_SOURCE_KEY, x.get_id(), job_obj.get_id(), program,
                                                 chief_args, job_obj.get_training_status(), config_idx=cur_config_idx,
                                                 env=dict_env)
            else:
                role = Job._ROLE_WORKER
                if idx < ps_tuner_configuration.ps_num:
                    role = Job._ROLE_PS
                job_obj.register_compute_node(x.get_id(), role, is_chief=False)
                slave_args = [script]
                slave_args.extend(common_args)
                slave_args.extend(
                    ["--" + TF_PROGRAM_WORKING_DIR + "=" + os.path.join(x.get_working_dir(), job_obj.get_id()),
                    "--" + TF_PROGRAM_IS_CHIEF + "=False", "--%s=%d" % (TF_PROGRAM_TASK_INDEX, idx)]
                )
                worker_msg = StartProcess.create(self.MSG_SOURCE_KEY, x.get_id(), job_obj.get_id(), program,
                                                 slave_args, job_obj.get_training_status(), config_idx=cur_config_idx,
                                                 env=dict_env)
            if idx == 0 or idx == 1:
                logging.info(f"hostname:{os.uname()[1]}")
                logging.info(f"program:{program} script:{script} cur_config_idx:{cur_config_idx} args:{args} dict_env:{dict_env} ")
            
            self.mq_manager.send_msg_to_compute_node(self.conn, x, worker_msg)

    def get_args_node_list(self, compute_node_list):
        ret = ""
        for x in compute_node_list:
            ret += "%s:%d," % (x.get_hostname(), x.get_tfport())
        return ret[:len(ret) - 1]

    def get_job_statistic_handler(self, message_obj):
        check_type(message_obj, GetJobStatistic)
        job_id = message_obj.get_job_id()
        job_obj = self.job_manager.get_job(job_id)

        msg = ReplyGetJobStatistic.create(self.mq_manager.get_monitor_name(), message_obj.get_source(), job_obj)
        self.mq_manager.send_msg_to_client(self.conn, msg)

    def handle_kill_job(self, message_obj):
        job_id = message_obj.get_job_id()
        job_obj = self.job_manager.get_job(job_id)
        self.dump_job_obj(job_obj)
        self.kill_job(job_obj=job_obj)

    def handle_deregister_compute_node(self, message_obj):
        compute_node_id = message_obj.get_source()
        self.compute_node_manager.remove_compute_node(compute_node_id)

    def handle_send_recovery_time(self, mesage_obj):
        """
        :param mesage_obj:
        :return:
        """
        job_id = mesage_obj.get_job_id()
        job_obj = self.job_manager.get_job(job_id)

        #handle variable list
        if len(job_obj.variable_map) == 0:
            tf_variable_list = json.loads(mesage_obj.json_variable_str, object_hook=TFVariableSeralizer.object_hook)
            job_obj.set_variable_map_from_tf_variable_containers(tf_variable_list)
        # job_obj.set_non_static_ops_names(json.loads(mesage_obj.json_non_static_ops_str))
        last_step_timestamp = job_obj.start_time
        if len(job_obj.training_statistic.get()) > 0:
            last_step_timestamp = job_obj.training_statistic.get()[-1].timestamp
        self.job_manager.get_job(job_id).add_recovery_time_from_chief_worker(
            mesage_obj.conf_idx,
            mesage_obj.timestamp - last_step_timestamp
        )

        if mesage_obj.reconfig_finish and \
            job_obj.training_status == TRAINING_STATUS_OPTIMAL_RUN:
            self.finish_online_tuning_reconfig(job_obj)

        if job_obj.change_status(common._JOB_STATUS_executing):
            web_ui.change_status(job_obj.id, time.time()-job_obj.start_time,
                                 common._JOB_STATUS_executing)

    def handle_natural_finish_ml(self, message_obj):
        """
        :param NaturalFinishML message_obj:
        :return:
        """
        job_id = message_obj.get_job_id()
        job_obj = self.job_manager.get_job(job_id)
        compute_id = message_obj.source
        job_obj.compute_node_natural_finish(compute_id)

        logging.debug(
            "Finished natural finish: %s" % job_obj.finished_compute_id)

        job_obj.finish()

        if job_obj.is_all_worker_natural_finish():
            self.logger.info("Job %s is finished, duration:%f s" % (
            job_obj.get_id(), job_obj.get_duration()))
            self.dump_job_obj(job_obj)
            self.kill_job(job_obj)

    def dump_job_obj(self, job_obj):
        """
        :param Job job_obj:
        :param str preview_tf_config_by_lhshome_path:
        :return:
        """
        job_dir = self.conf.get_job_log_dir(job_obj.get_id())
        if not os.path.exists(job_dir):
            os.makedirs(job_dir)

        summary_file_path = os.path.join(job_dir, "summary.csv")
        self.dump_job_summary(job_obj, summary_file_path)

        # config_history_csv_file = os.path.join(job_dir, "config_history.csv")
        # self.dump_config_history(job_obj, config_history_csv_file)

        iteration_training_data_dump = os.path.join(job_dir, job_obj.get_id() + "iteration_training_data.csv")
        self.dump_iteration_training_data(job_obj, iteration_training_data_dump)

        shutil.copyfile(os.path.join(self.conf.get_tftuner_home(), "monitor.out"), os.path.join(job_dir, "monitor.out"))
    
    def extract_hw_info_from_config(self, config):
        # default value based on ssd2
        hw_info = {
            'num_cpu_per_node': 16,
            'num_gpu_per_node': 0,
            'num_mem_per_node': 67302588416,
        }
        for name in hw_info:
            if name in config:
                hw_info[name] = config[name]
        return hw_info

    def remove_unused_keys_from_config(self, config, framework, isCPU):
        def remove_keys_from_dict(d: dict, l:list):
            for key in l:
                if key in d:
                    del d[key]

        filter_out_list = [
            'num_cpu_per_node',
            'num_gpu_per_node',
            'num_mem_per_node',
            'run_mode', 
            'online_os', 
            'online_num_iter', 
            'n_partition', 
            'model', 
            'init_os', 
            'init_num_iter', 
            'estimation_func'
        ]
        framework_related_list = {
            "tensorflow": [
                'do_common_subexpression_elimination', 
                'max_folded_constant_in_bytes', 
                'do_function_inlining', 
                'global_jit_level', 
                'infer_shapes', 
                'enable_bfloat16_sendrecv', 
                'place_pruned_graph'
            ],
            "torch": [],
            "mxnet": [
                'MXNET_CPU_WORKER_NTHREADS', 
                'MXNET_CPU_NNPACK_NTHREADS', 
                'MXNET_MP_WORKER_NTHREADS', 
                'MXNET_EXEC_ENABLE_INPLACE', 
                'MXNET_EXEC_NUM_TEMP', 
                'MXNET_ENGINE_TYPE', 
                'MXNET_KVSTORE_USETREE', 
                'MXNET_UPDATE_ON_KVSTORE', 
                'MXNET_CPU_TEMP_COPY', 
                'MXNET_GPU_MEM_POOL_TYPE', 
                'MXNET_GPU_PARALLEL_RAND_COPY'
            ]
        }
        gpu_related_list = [
            'tf_gpu_thread_mode',
            'cross_device_ops',
            'num_packs',
            'allocator_type',
            'deferred_deletion_bytes',
            'polling_active_delay_usecs',
            'num_dev_to_dev_copy_streams',
            'MXNET_GPU_MEM_POOL_TYPE', 
            'MXNET_GPU_PARALLEL_RAND_COPY'
        ]
        remove_keys_from_dict(config, filter_out_list)
        for framework_name in framework_related_list:
            if framework != framework_name:
                remove_keys_from_dict(config, framework_related_list[framework_name])
        if isCPU:
            remove_keys_from_dict(config, gpu_related_list)
        return config

    def dump_job_obj_to_db(self, job_obj, hw_info):
        """
        :param Job job_obj:
        :return:
        """
        self.logger.info(f"Load model graph and collect trial information")
        
        script_path = job_obj.get_args()[0]
        # for pytorch distribute training
        if ".py" not in script_path:
            for i in job_obj.get_args():
                if ".py" in i:
                    script_path = i
                    break

        script_arr = script_path.strip().split("/")
        graph_file = script_path.replace(script_arr[-1],"graph.json")
        self.logger.info(f"job_obj.get_args():{job_obj.get_args()}, graph_file:{graph_file}")
        with open(graph_file,'r') as uf:
            graph = json.load(uf)
        if "links" not in graph:
            graph = graph["graph"]
        if 'graph' in graph and 'name' in graph['graph']:
            del graph['graph']['name']

        raw_config = vars(job_obj.get_current_config())
        
        framework = get_framework(job_obj.get_args()[0])
        if framework is None:
            framework = "mxnet"
        config_to_store = self.remove_unused_keys_from_config(raw_config, framework, hw_info['num_gpu_per_node'] == 0)
        result = {'ps_config':config_to_store, 'elapsed_time_in_ms': job_obj.get_duration()*1000, 'loss': job_obj.get_final_loss()}
        self.logger.info(f"Result to save:{result}")
        record = str(tuple([
            int(os.getenv("SELFTF_NUM_COMPUTE_NODE")),
            hw_info['num_cpu_per_node'],
            hw_info['num_gpu_per_node'],
            hw_info['num_mem_per_node'],
            framework
        ]+list(map(str,(graph,result)))))
        conn = pymysql.connect(host="ssd2",user="root",passwd="12345678",db="tftuner")
        cursor = conn.cursor()
        query_arg = f"""INSERT INTO HISTORY(num_node,num_cpu_per_node,
                        num_gpu_per_node,num_mem_per_node,framework,
                        model_information, ps_config_result_metrics)
                        VALUES {record}"""
        self.logger.info(f"Saved information in MySQL")
        cursor.execute(query_arg)
        conn.commit()
        conn.close()

    def dump_normal_job_summary(self, job_obj, file_obj):
        summary = Summary(
            job_obj.get_id(),
            job_obj.get_duration(),
            job_obj.get_avg_recovery_time(),
            job_obj.get_target_loss(),
            job_obj.get_ml_model(),
            job_obj.get_batch_size(),
            job_obj.get_learning_rate(),
            job_obj.get_final_loss(),
            job_obj.get_os_size(),
            job_obj.get_iter_per_os(),
            job_obj.get_online_os_size(),
            job_obj.get_online_num_iter_per_os(),
            job_obj.counter_check_do_reconfig,
            job_obj.counter_do_reconfig,
            job_obj.training_statistic.get_num_training_sample(),
            job_obj.get_init_duration(),
            job_obj.get_estimation_func_name(),
            job_obj.mode,
            job_obj.get_total_reconfiguration_time(),
            job_obj.get_current_config()
        )
        csv_dict_writer = csv.DictWriter(file_obj, fieldnames=
        ["job_id",
         "runtime_sec",
         "avg_recovery_time_sec",
         "target_loss",
         "ml_model",
         "batch_size",
         "learning_rate",
         "final_loss",
         "os_size",
         "n_iter_per_os",
         "online_os_size",
         "online_n_iter_per_os",
         "check_reconfig",
         "do_reconfig",
         "total_iteration",
         "init_duration",
         "estimation_func",
         "run_mode",
         "total_reconfiguration_time",
         "config"])
        csv_dict_writer.writeheader()
        csv_dict_writer.writerow(summary.__dict__)

    def dump_job_summary(self, job_obj, summary_file_path):
        """
        :param Job job_obj:
        :param summary_file_path:
        :return:
        """
        with open(summary_file_path, 'w') as f:
            if job_obj.mode == MODE_VALUE_TEST_ESTIMATION_FUNC_REMAINING_TIME:
                #TODO:
                pass
            else:
                self.dump_normal_job_summary(job_obj, f)


    # def dump_config_history(self, job_obj, config_history_csv_file):
    #     with open(config_history_csv_file, 'w+') as f:
    #         # print header
    #         f.write("iteration")
    #         f.write("%f,%f,%f" % (job_obj.get_duration(), job_obj.get_avg_recovery_time(), job_obj.get_target_loss()))

    def dump_iteration_training_data(self, job_obj, iteration_training_data_dump):
        """
        :param Job job_obj:
        :param str iteration_training_data_dump:
        :return:
        """
        with open(iteration_training_data_dump, 'w') as f:
            # print header
            f.write(
                "step,local_step,timestamp,elapsed_time_sec,loss,config_idx,ps_num,worker_num,intra_op_parallelism_threads,inter_op_parallelism_threads,"
                "optimizer,learning_rate,batch_size\n")
            for x in job_obj.training_statistic.get():
                f.write("%d,%d,%f,%f,%f,%d,%d,%d,%d,%d,%s,%f,%d\n" % (
                    x.step,
                    x.local_step,
                    x.timestamp,
                    x.elapsed_time_in_ms,
                    x.loss,
                    x.ps_config_idx,
                    x.ps_config.ps_num,
                    x.ps_config.worker_num,
                    x.ps_config.intra_op_parallelism_threads,
                    x.ps_config.inter_op_parallelism_threads,
                    optimizer_list[x.ps_config.optimizer],
                    x.ps_config.learning_rate,
                    x.ps_config.batch_size
                    ))

    def handle_dump_log(self, message_obj):
        self.logger.info(f"Enter handle_dump_log()")
        job_obj = self.job_manager.get_job(message_obj.get_job_id())
        self.dump_job_obj(job_obj)

    def kill_non_chief_workers_by_job(self, job_obj):
        """
        :param Job job_obj:
        :return:
        """
        ps_compute_node_id_list = job_obj.get_non_chief_worker_compute_node_id_list()
        for compute_node_id in ps_compute_node_id_list:
            self.logger.debug("Kill non chief Worker: %s" % compute_node_id)
            self.kill_process_by_job(job_obj.get_id(), compute_node_id)

    def kill_chief_workers_by_job(self, job_obj):
        compute_node_id = job_obj.get_chief_compute_node_id()
        self.kill_process_by_job(job_obj.get_id(), compute_node_id)

    def trigger_chkpt(self, job_obj):
        logging.debug("Trigger chief to do checkpoint")
        chief_compute_id = job_obj.get_chief_compute_node_id()
        chief_compute_node = self.compute_node_manager.get_compute_node_by_id(chief_compute_id)

        msg = TriggerChiefCheckPoint.create(source=self.mq_manager.get_monitor_name(),
                                            destination=self.mq_manager.get_compute_node_tf_routing_key(chief_compute_node),
                                            job_id=job_obj.get_id())
        self.mq_manager.send_msg_to_compute_node_tf(self.conn, chief_compute_node, msg)

        job_obj.change_status(common._JOB_STATUS_checkingpointing)
        web_ui.change_status(job_id=job_obj.id,
                             x_value=time.time() - job_obj.start_time,
                             status=common._JOB_STATUS_checkingpointing)

    def handle_finish_sub_training(self, message_obj):
        job_obj = self.job_manager.get_job(message_obj.get_job_id())
        job_obj.increment_counter_finish_sub_training_workers(message_obj.get_source())

        logging.debug("Finished sub training workers: %s" % job_obj.finish_sub_training_workers)
        if job_obj.is_finish_sub_training():

            if job_obj.mode in [MODE_VALUE_TEST_RECONFIGSCHEME_ZERO,
                    MODE_VALUE_TEST_RECONFIGSCHEME_ONE,
                    MODE_VALUE_TEST_RECONFIGSCHEME_TWO] and \
                    not job_obj.check_do_init_reconfig():
                # Testing mode here only
                job_obj.finish()
                self.dump_job_obj(job_obj)
                self.kill_job(job_obj)
                return

            logging.debug("All worker finish sub training, check whether live reconfig or restart")
            if job_obj.check_do_init_reconfig():
                next_conf = self.get_next_config(job_obj=job_obj)
                self.reconfig(job_obj=job_obj, new_config=next_conf)
            else :
                # HACK: dun want to wait for BO.... so use back last configuration first
                job_obj.finish_training()
                self.reconfig(job_obj=job_obj, new_config=job_obj.current_configuration_plan)
            # check whether the next config is non data reallocated or not
            # if job_obj.current_configuration_plan
            # if yes:
            #   send ReconfigMessage to each worker
            # if not:
            #   directly kill them
            # broadcast message to all worker
            if job_obj.get_training_status() == TRAINING_STATUS_OPTIMAL_RUN:
                # # open a new thread to do online checking
                # def _thread_run():
                #     """
                #     :param common.Job job_obj:
                #     :return:
                #     """
                #     while not job_obj.is_finish():
                #         if job_obj.check_do_online_reconfig():
                #             next_conf = self.tf_config_manager.get_tf_config_by_gp(
                #                 job_obj)
                #             job_obj.increment_counter_check_do_reconfig()
                #             if job_obj.is_different_config(next_conf):
                #                 job_obj.increment_counter_do_reconfig()
                #                 logging.debug("We got a new config: %s\n"
                #                               "old config: %s" % (
                #                               json.dumps(next_conf.__dict__),
                #                               json.dumps(
                #                                   job_obj.current_configuration_plan.__dict__)))
                #                 self.reconfig(job_obj, next_conf)
                #             job_obj.done_check_online_reconfig()
                #         else:
                #             time.sleep(1)
                if self.dict_job_subprocess.get(job_obj) is None:
                    thread_obj = SubProcess(
                        tf_config_manager=self.tf_config_manager)
                    self.dict_job_subprocess[job_obj] = thread_obj
                else:
                    self.logger.error("BO thread exist")

    def handle_change_config(self, message_obj):
        """
        :param ChangeConfig message_obj:
        :return:
        """
        job_id = message_obj.get_job_id()
        if job_id == "0":
            job = self.job_manager.job_set[0]
            job_id = job.id
        else:
            job = self.job_manager.get_job(job_id)
        config = job.get_current_config()
        new_num_node = config.ps_num + message_obj.get_num_node_change()
        old_num_node = config.ps_num
        if new_num_node < 1:
            new_num_node = 1
        if new_num_node > 35:
            new_num_node = 35
        if old_num_node == new_num_node:
            return
        else:
            delta = new_num_node - old_num_node
            config.ps_num = new_num_node
            self.reconfig(job, config)

            if delta>0:
                web_ui.change_status(job_id, time.time()-job.start_time, "Add "+str(delta)+" PS(s)")
            else:
                web_ui.change_status(job_id, time.time()-job.start_time, "Remove "+str(abs(delta))+" PS(s)")


    def get_next_config(self, job_obj):
        """
        Get config depends on the current job status
        :param Job job_obj:
        :return:
        """
        # Initialize phase
        #   - in preview next config
        # Transist between init phase and online tuning phase
        #   -
        # Online tuning phase
        #   - in job_ob: next config
        # Finish
        if job_obj.get_training_status() == TRAINING_STATUS_TRAINING:
            if job_obj.check_do_init_reconfig():
                return self.training_draw_next_configuration(job_obj)
            else:
                job_obj.finish_training()
                return self.tf_config_manager.get_tf_config_by_gp(job_obj)
        if job_obj.get_training_status() == TRAINING_STATUS_OPTIMAL_RUN:
            # assume background
            conf = job_obj.next_configuraton_plan
            if conf is None:
                raise Exception("No next configuration.....")
            return conf

    def test_reconfig_scheme1(self, job):
        init_conf = self.tf_config_manager.get_tf_config_for_test_scheme1(job)
        self.start_job_by_config(init_conf, job)

    def test_with_fixed_config(self, job, total_num_of_node=0):
        init_conf = self.tf_config_manager.get_tf_fixed_config(job, total_num_of_node)
        self.start_job_by_config(init_conf, job)

    def get_master_ps_device_name(self):
        return job_prefix+str(0)


    def check_finish_ml_job(self, job_obj):
        """
        :param Job job_obj:
        :return:
        """
        # different stragegy
        if self.ml_job_finish_stragegy == ML_JOB_FINISH_STRATEGY_WORKER_ALL_UNDER_TARGET:
            finish = job_obj.is_all_worker_converge()
        elif self.ml_job_finish_stragegy == ML_JOB_FINISH_STRATEGY_WORKER_REACH_TARGET_ONCE:
            finish = job_obj.is_all_worker_reach_target_loss_once()
        elif self.ml_job_finish_stragegy == ML_JOB_FINISH_STRATEGY_MOVING_AVG:
            finish = job_obj.is_moving_average_smaller_than_target_loss()
        else:
            finish = job_obj.is_all_worker_reach_target_loss_once(threshold=self.ml_job_finish_threshold)

        if finish and not job_obj.is_finish():
            job_obj.finish()
            thread = self.dict_job_subprocess.get(job_obj)
            lock = self.dict_job_lock.get(job_obj)
            if thread is not None:
                thread.join()
            if lock is not None:
               self.dict_job_lock[job_obj] = None
            self.dump_job_obj(job_obj)
            self.kill_job(job_obj)

class JobManager(object):
    def __init__(self):
        self.job_set = []

    #TODO take out thne batch_size and learning rate
    def create_job(self, script, args, num_iter_per_os, os_size, target_loss, ml_model="ML",
                   batch_size=1000, learning_rate=0.001, online_os_size=100, online_num_iter_per_os=200,
                   estimation_func=Job._BO_func, mode=MODE_VALUE_SELFTF, compute_node_id_list=[],
                   optimizer=0, test_job_config_idx=-1):
        if mode == MODE_VALUE_TEST_ESTIMATION_FUNC_REMAINING_TIME:
            assert test_job_config_idx > 0
            job = JobEstimationFuncTest(script, args, num_iter_per_os=num_iter_per_os, os_size=os_size, target_loss=target_loss,
                  ml_model=ml_model,
                  batch_size=batch_size, learning_rate=learning_rate, online_os_size=online_os_size,
                  online_num_iter_per_os=online_num_iter_per_os, estimation_func=estimation_func, mode=mode,
                  compute_node_id_list=compute_node_id_list,optimizer=optimizer,
                                        stop_config_idx=test_job_config_idx)
        else:
            job = Job(script, args, num_iter_per_os=num_iter_per_os, os_size=os_size, target_loss=target_loss,
                      ml_model=ml_model,
                      batch_size=batch_size, learning_rate=learning_rate, online_os_size=online_os_size,
                      online_num_iter_per_os=online_num_iter_per_os, estimation_func=estimation_func, mode=mode,
                      compute_node_id_list=compute_node_id_list,optimizer=optimizer)
        logging.debug(
            "os_size = %d, num_iter_per_os = %d, online_os_siz = %d" % (os_size, num_iter_per_os, online_os_size))
        self.job_set.append(job)
        return job

    def get_job(self, job_id=""):
        """
        :param job_id:
        :return:
        :rtype: Job
        """
        if job_id == "":
            return self.job_set[-1]
        for x in self.job_set:
            if x.get_id() == job_id:
                return x
        raise Exception("No job with id %s" % job_id)



# should allow a machine contains multi_node
# a compute_node should be identify by hostname_port
class ComputeNodeManager:
    def __init__(self):
        self.compute_node_list = []
        self.logger = logging.getLogger("ComputeNodeManager")

    def get_num_node(self):
        return len(self.compute_node_list)

    def get_num_thread_of_node(self):
        return self.compute_node_list[0].get_num_thread()

    def register_compute_node(self, hostname, tfport, work_dir="", num_thread=16):
        if self.is_dup(hostname, tfport):
            raise Exception("Duplicated registration")
        else:
            self.logger.info("%s:%s register as an agent" % (hostname, tfport))
            compute_node = ComputeNode(hostname, tfport, working_dir=work_dir, num_thread=num_thread)
            self.compute_node_list.append(compute_node)
            self.compute_node_list.sort(key=lambda x:x.get_hostname())

    def is_dup(self, hostname, tfport):
        try:
            self.get_compute_node_by_hostname_tfport(hostname, tfport)
            return True
        except:
            return False

    def get_compute_node_by_hostname_tfport(self, hostname, tfport):
        for x in self.compute_node_list:
            if x.get_hostname == hostname and x.get_tfport == tfport:
                return x
        raise Exception("No compute node with hostname \"%s\"" % hostname)

    def get_compute_node_by_id(self, id):
        for x in self.compute_node_list:
            if x.get_id() == id:
                return x
        raise Exception("No compute node with id \"%s\"" % id)

    def touch_compute_node_by_id(self, id):
        hostname, tfport = ComputeNode.get_hostname_tfport_from_id(id)

        self.get_compute_node_by_hostname_tfport(hostname, tfport) \
            .touch_compute_node()

    def get_compute_node_list(self):
        return self.compute_node_list

    def remove_compute_node(self, compute_node_id):
        compute_node_obj = self.get_compute_node_by_id(compute_node_id)
        self.compute_node_list.remove(compute_node_obj)
        logging.debug("Remove compute node %s", compute_node_id)


class VariableMapUtil(object):
    @classmethod
    def get_new_variable_map(cls, job, new_conf, master_ps_device_name):
        """

        :param Job job:
        :param PSTunerConfiguration new_conf:
        :return:
        """
        org_variable_map = job.get_variable_map()
        cur_ps_num = job.get_current_config().ps_num
        new_ps_num = new_conf.ps_num
        new_variable_map = None
        if cur_ps_num == new_ps_num:
            return org_variable_map
        if new_ps_num > cur_ps_num:
            new_variable_map = cls._calc_new_variable_map_with_more_ps(org_variable_map, new_ps_num)
        if new_ps_num < cur_ps_num:
            new_variable_map = cls._calc_new_variable_map_with_less_ps(org_variable_map, cur_ps_num, new_ps_num)

        # Hack here. Since some static optimizer variable depends on  fir var
        first_var_key = min(new_variable_map.keys())
        first_device_name = new_variable_map[first_var_key]
        if first_device_name != master_ps_device_name:
            # find a variable of master_ps_device_name to exchange
            exchange_var_name = None
            for var_name, device_name in new_variable_map.items():
                if master_ps_device_name == device_name:
                    exchange_var_name = var_name
                    break
            assert exchange_var_name is not None
            new_variable_map[exchange_var_name] = first_device_name
            new_variable_map[first_var_key] = master_ps_device_name

        return new_variable_map

    @classmethod
    def _calc_new_variable_map_with_more_ps(cls, old_variable_map, ps_num):
        """
        :param dict old_variable_map:
        :param ps_num:
        :return:
        """
        ret = {}

        avg_num_var = len(old_variable_map) / ps_num
        dict_node_vars = cls.group_var_by_node(old_variable_map)
        nodes = dict_node_vars.keys()

        pending_consolidation_vars = []


        # Missing some variable
        for node, vars in dict_node_vars.items():
            if len(vars) > avg_num_var:
                # release some vars out
                num_exceed = len(vars) - avg_num_var
                # pop out the num_exceed
                exceed_vars = vars[-num_exceed:]
                pending_consolidation_vars.extend(exceed_vars)
                # remove from the list
                for x in exceed_vars:
                    vars.remove(x)

                # put back to the ret
                for x in vars:
                    ret[x] = node
            else:
                # keep current
                for x in vars:
                    ret[x] = node

        # assign pending to the new ps
        new_ps_list = list(set(cls.get_device_list_by_num_ps(ps_num)) -
                           set(nodes))

        # RR assign
        ps_idx = 0
        try:
            for var in pending_consolidation_vars:
                ret[var] = new_ps_list[ps_idx]
                ps_idx += 1
                if ps_idx == len(new_ps_list):
                    ps_idx = 0
        except:
            logging.exception("Something wrong with _calc_new_variable_map_with_more_ps()\n"
                              "ps_idx: %s \n"
                              "new_ps_lis: %s " % (str(ps_idx),
                                                   str(new_ps_list)))
            raise
        try:
            assert len(cls.group_var_by_node(ret).keys()) == ps_num
            assert len(old_variable_map) == len(ret)
        except:
            logging.exception("Something wrong with _calc_new_variable_map_with_more_ps()\n"
                              "old_variable_map: %s \n"
                              "new_variable_map: %s \n" % (str(old_variable_map),
                                                           str(ret)))
            raise

        return ret

    @classmethod
    def sort_ps_list_by_num_var(self, dict_node_vars, ps_list):
        ps_num_var = []
        for ps in ps_list:
            num_var = 0
            existing_vars = dict_node_vars.get(ps)
            if existing_vars is not None:
                num_var = len(existing_vars)
            ps_num_var.append((ps, num_var))
        ps_num_var.sort(key=lambda tup: tup[1])
        return list(map(lambda tup: tup[0], ps_num_var))

    @classmethod
    def _calc_new_variable_map_with_less_ps(cls, old_variable_map, old_ps_num, new_ps_num):
        """
        :param VariableDict old_variable_map:
        :param ps_num:
        :return:
        """
        ret = {}

        dict_node_vars = cls.group_var_by_node(old_variable_map)

        # Find out the deleted ps list
        discard_ps_device_name = cls.get_device_list_by_idx_bound(new_ps_num, old_ps_num)

        pending_consolidation_vars = []
        for device_name in discard_ps_device_name:
            if dict_node_vars.get(device_name) is None:
                continue
            pending_consolidation_vars.extend(dict_node_vars[device_name])

        # assign pending to the new ps
        new_ps_list = cls.get_device_list_by_num_ps(new_ps_num)

        # sort by number of vars holded by new_ps
        new_ps_list = cls.sort_ps_list_by_num_var(dict_node_vars, new_ps_list)

        # add preserved variables to the list first
        for node in new_ps_list:
            for var in dict_node_vars.get(node,[]):
                ret[var] = node


        # RR assign
        ps_idx = 0
        for var in pending_consolidation_vars:
            ret[var] = new_ps_list[ps_idx]
            ps_idx += 1
            if ps_idx == len(new_ps_list):
                ps_idx = 0

        try:
            assert len(old_variable_map) == len(ret)
            assert len(cls.group_var_by_node(ret).keys()) == new_ps_num
        except:
            logging.exception("Something wrong with _calc_new_variable_map_with_less_ps()\n"
                              "old_variable_map: %s \n"
                              "new_variable_map: %s \n" % (str(old_variable_map),
                                                           str(ret)))
            raise

        return ret

    # def _rebalance_variable(self, dict_node_var, list_ps):
    #     """
    #     :param dict[str, list[str]] dict_node_var:
    #     :param list[str] list_ps:
    #     :return:
    #     """
    #
    # def merge_variable_map(self, dictA, dictB):
    #

    @classmethod
    def get_device_list_by_num_ps(cls, num_ps):
        return cls.get_device_list_by_idx_bound(0, num_ps)

    @classmethod
    def get_device_list_by_idx_bound(cls, lower_bound, upper_bound):
        ret = []
        for x in range(lower_bound, upper_bound):
            ret.append(job_prefix + str(x))
        return ret

    @classmethod
    def group_var_by_node(cls, variable_map):
        """
        :rtype: dict[str, list[str]]
        :return:
                """
        ret = {}
        for k, v in variable_map.items():
            if ret.get(v) is None:
                ret[v] = []
            ret[v].append(k)
        return ret



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        )
    logging.getLogger("amqp").setLevel(logging.INFO)
    # Fix random
    seed = 0
    numpy.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    monitor = Monitor(get_config())
    monitor.run()
