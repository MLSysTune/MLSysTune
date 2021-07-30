import logging
import threading

import time

import numpy

from selftf.lib import tf_program_util, common
import sys
import tensorflow as tf
import json

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    )

tf.logging.set_verbosity(logging.DEBUG)


class MLJobFramework:

    def print_trainable_variables(self, context):
        self.model_definition(context)
        print(json.dumps(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))

    def get_observe_loss_variable(self, context):
        """
        :param tf_program_util.TFProgramUtil context:
        :return:
        """
        return context.get_tf_variable_loss()

    def __init__(self):
        self.sess = None

    def model_definition(self, context):
        """
        :param tf_program_util.TFProgramUtil context:
        :return:
        """
        pass

    def get_feed_dict(self, context):
        """
        :param tf_program_util.TFProgramUtil context:
        :return:
        """
        return None

    def has_enqueue_func(self):
        return False

    def get_enqueue_func(self, context, sess):
        return None

    def run(self):
        tftuner = tf_program_util.TFProgramUtil()

        job_name = tftuner.get_job_name()
        task_index = tftuner.get_task_index()

        # set random seed
        tf.set_random_seed(task_index)
        numpy.random.seed(task_index)

        # for collect statistic of trainable variable
        # should be executed by Monitor only
        if tftuner.is_collect_statistic_run():
            self.print_trainable_variables(tftuner)
            sys.exit(0)

        # import pydevd_pycharm
        # pydevd_pycharm.settrace('127.0.0.1', port=4445, stdoutToServer=True,
        #                         stderrToServer=True)

        # cluster specification
        cluster = tftuner.get_tf_cluster_spec()

        server_config = tftuner.get_tf_config_proto()

        server = tf.train.Server(
            cluster,
            job_name=job_name,
            task_index=task_index,
            config=server_config)

        tftuner.set_tf_server_target(server.target)
        tftuner.set_graph_init_func(self.model_definition)
        # Redirect all global variables (e.g global step) to the master ps
        with tftuner.get_default_tf_graph().as_default():

            tf.get_variable_scope().set_partitioner(
                tf.fixed_size_partitioner(tftuner.get_num_ps())
            )

            with tf.device(tftuner.device_setter()):
                self.model_definition(tftuner)
                tftuner.init_graph() # this ine problem
                #
                # if tftuner.get_is_chief() and tftuner.conf_dict.get(common.conf_dict_non_static_ops_names) == None:
                #     # For first iteration get the static ops by chief
                #     # tftuner.set_chief_temp_static_ops(tftuner.get_static_ops())
                #     tftuner.set_chief_temp_nonstatic_ops(tftuner.get_non_static_ops())
                # tftuner.reallocate_static_ops()
                tf_program_util.SelfTFOptimizerContext.clear_all_ops_collocate()

        if self.has_enqueue_func():
            self.enqueue_sess = tf.Session(server.target, graph=tftuner.get_default_tf_graph())
            enqueue_func = self.get_enqueue_func(tftuner, self.enqueue_sess)
            enqueue_coord = tf.train.Coordinator()
            self.enqueue_coord = enqueue_coord

            def enqueue_fun_wrapper():
                while not enqueue_coord.should_stop():
                    try:
                        enqueue_func()
                    except:
                        logging.exception("Fail to do enqueue")
                        time.sleep(1)
                        pass

            self.enqueue_thread = threading.Thread(target=enqueue_fun_wrapper)
            self.enqueue_thread.start()

        local_error = False
        try:
            while not tftuner.flag_end_process:
                tftuner.pre_recovery()

                # Worker or PS
                if not tftuner.is_worker():

                    # Checking for reconfiguration staff
                    if tftuner.do_live_reconfig2_context is not None:
                        if tftuner.do_live_reconfig2_context.is_phase_1():
                            tftuner.do_live_reconfig2_context.finish_phase1()
                        elif tftuner.do_live_reconfig2_context.is_final_phase():
                            tftuner.clear_do_scheme2_reconfig()

                    logging.debug("I am ps, I sleep")
                    time.sleep(1)

                    tftuner.post_check_reconfig_or_finish()

                else:

                    logging.info("Open new tensorflow session")
                    with tftuner.get_default_tf_graph().as_default():
                        with tftuner.get_monitored_training_session() as sess:
                            try:
                                sync_should_stop = False  # flag for sync optimizer , ensure they do one more lap

                                logging.debug("Session is ready")
                                self.sess = sess

                                tftuner.post_recovery(sess=sess)

                                # if tftuner.get_is_chief():
                                #     import pydevd_pycharm
                                #     pydevd_pycharm.settrace('127.0.0.1',
                                #                             port=4445,
                                #                             stdoutToServer=True,
                                #                             stderrToServer=True)

                                batch_time = time.time()
                                tftuner.pre_do_all_iteration(sess)

                                # from tensorflow.python import \
                                #     debug as tf_debug
                                #
                                # run_options = tf.RunOptions()
                                # tf_debug.watch_graph(
                                #     run_options,
                                #     sess.graph,
                                #     debug_urls=[
                                #         "file:///tmp/tfdbg_dumps_1"])
                                #

                                def print_accumulators():
                                    for accum, num_accumulator_item in tftuner.sync_debug_grad_cur_items_ops:
                                        logging.debug("DEBUG %s, %s: accumlator current items:%d" %
                                                      (accum.name,
                                                       accum._accumulator_ref.op.get_attr("shared_name") ,
                                                       sess.run(num_accumulator_item)))

                                # run_options = tf.RunOptions(
                                #     timeout_in_ms=60000)
                                while True:

                                    feed_dict = self.get_feed_dict(tftuner)
                                    try:
                                        if feed_dict is None:
                                            # try:

                                            _, cost, step = sess.run(
                                                [tftuner.get_train_op(),
                                                 self.get_observe_loss_variable(
                                                     tftuner),
                                                 tftuner.get_tf_variable_global_step()],
                                                 options=tftuner.get_tf_run_option())
                                            # except:
                                            #     logging.exception("Fail to execute")
                                            #     if tftuner.get_is_chief():
                                            #         print_accumulators()
                                            #         for k, t in tftuner.do_live_reconfig2_context.dict_grad_accum_shared_name_op.items():
                                            #
                                            #             accum = t[0]
                                            #             take_op = t[2]
                                            #             accum_name= accum.name
                                            #
                                            #             try:
                                            #                 avg_grad = sess.run(take_op)
                                            #                 logging.debug("DEBUG run take op %s: value: %s" % (accum_name, avg_grad))
                                            #             except:
                                            #                 logging.exception("Skip %s"% accum_name)
                                            #         print_accumulators()
                                            #     sess.close()
                                        else:
                                            _, cost, step = sess.run(
                                                [tftuner.get_train_op(),
                                                 self.get_observe_loss_variable(
                                                     tftuner),
                                                 tftuner.get_tf_variable_global_step()],
                                                feed_dict=feed_dict,
                                                options=tftuner.get_tf_run_option())
                                    except:
                                        logging.exception("Fail to do tf sess.run()")
                                        # print_accumulators()
                                        # To ensure local step is updated
                                        if tftuner._is_sync_optimizer:
                                            sess.run(tftuner.sync_local_step_init_op)
                                        pass
                                    tftuner.post_do_iteration(steps=step,
                                                              loss=cost,
                                                              timestamp=time.time(),
                                                              duration=time.time() - batch_time)
                                    batch_time = time.time()

                                    if (tftuner.should_stop_iteration(step,
                                                                      cost)
                                        or tftuner.flag_do_live_reconfig2
                                        or tftuner.flag_do_checkpoint
                                        or tftuner.flag_end_process):
                                        if tftuner._is_sync_optimizer and \
                                                tftuner._is_optimal_run() and \
                                                tftuner.do_live_reconfig2_context.is_phase_0() and \
                                                step < tftuner.target_reconfig_step:
                                            continue
                                        else:
                                            break
                                        # before reconfig2 , run extra step to ensure al worker receive
                                    logging.debug(
                                        "tftuner.flag_do_live_reconfig2:%s" %tftuner.flag_do_live_reconfig2)

                            except (KeyboardInterrupt, SystemExit):
                                logging.exception(
                                    "System should be exited herfe")
                                pass
                            except:
                                logging.exception("Something Wrong")
                                local_error = True  # Turn this flag on the restart it
                            finally:
                                if not local_error:
                                    # Regular program exit
                                    tftuner.post_do_all_iteration(sess)
                    tftuner.post_check_reconfig_or_finish()
        except:
            logging.exception("Error")
        finally:
            if self.has_enqueue_func():
                self.enqueue_sess.close()
            sys.exit(0)
