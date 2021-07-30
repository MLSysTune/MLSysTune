# class BaseUtil:
#
#     def __init__(self):
#
#
#
import json
import os
import time
# from absl import flags

import tensorflow as tf
import networkx as nx
from kombu import Connection

import selftf
from selftf.lib import common
from selftf.lib.common import PSTunerTrainingDataSerializer
from selftf.lib.message import UpdateNumOfSteps
from selftf.lib.queue import KombuQueueManager

import networkx as nx


def convert_model(model,script_path):
    script_arr = script_path.strip().split("/")
    output_file = script_path.replace(script_arr[-1],"graph.json")
    print(f"[Model Matching] output file path:{output_file}")
    dot_model = tf.keras.utils.model_to_dot(
      model, show_shapes=True,  show_layer_names=True,
      rankdir='TB', expand_nested=True, dpi=96, subgraph=True)

    nxgf = nx.drawing.nx_pydot.from_pydot(dot_model)
    
    for n in nxgf.nodes:
        nxgf.nodes[n]['weight'] = 1

        arr = nxgf.nodes[n]['label'].split('\n')
        nxgf.nodes[n]['label'] = arr[0]

        inout_shape = arr[1].replace('{','').replace('}','').replace(':','').replace('[','').replace(']','')
        inout_arr = inout_shape.split('|')
        nxgf.nodes[n]['input_shape'] = inout_arr[3]
        nxgf.nodes[n]['output_shape'] = inout_arr[4]
    
    js_data = nx.readwrite.json_graph.node_link_data(nxgf)
    graph_dict = {}
    graph_dict["graph"] = js_data

    with open(output_file,'w') as dump_f:
        json.dump(graph_dict,dump_f)
    print(f"[Model Matching] Finish transferring model")


class MLTunerUtil(object):

    def compact_flag_defination(self):
        # input flags
        flags = tf.compat.v1.app.flags
        flags.DEFINE_string("job_name", "worker",
                                   "Either 'ps' or 'worker'")
        flags.DEFINE_integer("task_index", 0,
                                    "Index of task within the job")
        flags.DEFINE_float("targted_accuracy", 0.5,
                                  "targted accuracy of model")
        flags.DEFINE_string("optimizer", "Adam", "optimizer we adopted")

        # <--  Add-on by pstuner -->
        # flags.DEFINE_string("ps_list", "", "")
        # flags.DEFINE_string("worker_list", "", "")
        flags.DEFINE_string("node_list", "", "")
        flags.DEFINE_string("working_dir", "", "")
        flags.DEFINE_bool("is_chief", True, "")
        flags.DEFINE_integer("max_iteration", 20, "")
        flags.DEFINE_float("target_loss", 0.5, "")
        flags.DEFINE_string("conf_dict", "{}", "")

        # <-- From Andy Framework -->
        flags.DEFINE_string("ML_model", "LR", "ML model")
        flags.DEFINE_float("targeted_loss", 0.05,
                                  "targted accuracy of model")
        flags.DEFINE_integer("Batch_size", 1000, "Batch size")
        flags.DEFINE_integer("num_Features", 54686452,
                                    "number of features")
        flags.DEFINE_float("Learning_rate", 0.001, "Learning rate")
        flags.DEFINE_integer("Epoch", 1, "Epoch")

        # <-- ampq -->
        flags.DEFINE_string("amqp_user", "guest", "")
        flags.DEFINE_string("amqp_password", "guest", "")
        flags.DEFINE_string("amqp_host", os.getenv("AMPQ_MASTER_NODE"),
                                   "")
        flags.DEFINE_integer("amqp_port", 5672, "")
        flags.DEFINE_string("agent_id", "", "")
        flags.DEFINE_integer("agent_config_idx", -1, "")
        flags.DEFINE_string("agent_job_id", "", "")

    def __init__(self):
        # for compact define something useless here
        self.compact_flag_defination()
        flags = tf.compat.v1.app.flags
        flags.DEFINE_bool(
            'get_model', False, 'If True, record model in networkx json format in file. If False, train the model.')
        flags.DEFINE_string(
                'script_path', None, 'path for store json format model')
        if flags.FLAGS.get_model:
            self.init_for_classification()
            return

        self.agent_job_id = ""
        self.conf_dict = json.loads(os.environ.get(common.ENV_KEY_MLTUNER_CONF_DICT))

        # set ps worker
        os.environ["TF_CONFIG"] = json.dumps(self.get_tf_cluster_spec(
            self.get_node_list(),
            self.get_task_index(),
            self.get_num_ps()
        ))

        # set KMP hardware parameter
        os.environ["KMP_AFFINITY"] = "verbose,{respect},granularity={specifier},{type},{permute},{offset}".format(respect=self.get_KMP_AFFINITY_respect(),
                         specifier=self.get_KMP_AFFINITY_granularity(),
                         type=self.get_KMP_AFFINITY_type(),
                         permute=self.get_KMP_AFFINITY_permute(),
                         offset=self.get_KMP_AFFINITY_offset())
        os.environ["KMP_BLOCKTIME"] = self.get_KMP_BLOCKTIME()
        os.environ["OMP_NUM_THREADS"] = self.get_OMP_NUM_THREADS()
        os.environ["MKL_DYNAMIC"] = self.get_MKL_DYNAMIC()

        #create connection for ampq
        self.conn = None
        amqp_host = os.environ.get("AMPQ_MASTER_NODE", None)
        if amqp_host is not None:
            amqp_user = "guest"
            amqp_password = "guest"
            amqp_port = 5672
            self.agent_id = self.get_node_list()[self.get_task_index()].replace(
                ":", "_")
            self.agent_job_id = self.conf_dict[common.CONF_DICT_JOB_ID]
            self.conn = Connection('amqp://%s:%s@%s:%s//' % (
                amqp_user, amqp_password,
                amqp_host, amqp_port))
    
    def init_for_classification(self):
        self.conf_dict = {}
        self.conf_dict[common.CONF_DICT_NODE_LIST] = [1,2,3]
        self.conf_dict[common.CONF_DICT_TASK_INDEX] = 1
        self.conf_dict[common._worker_num] = 2
        self.conf_dict[common.CONF_DICT_JOB_ID] = "123"
        self.conf_dict[common._batch_size] = 16
        self.conf_dict[common._learning_rate] = 0.0001
        self.conf_dict[common._optimizer] = 1
        
        self.conf_dict[common._ps_num] = 1
        self.conf_dict[common._inter_op_parallelism_threads] = 1
        self.conf_dict[common._intra_op_parallelism_threads] = 1
        self.conf_dict[common._do_common_subexpression_elimination] = 0
        self.conf_dict[common._max_folded_constant_in_bytes] = 0
        self.conf_dict[common._do_function_inlining] = 0
        self.conf_dict[common._global_jit_level] = 0
        self.conf_dict[common._enable_bfloat16_sendrecv] = 0
        self.conf_dict[common._infer_shapes] = 0
        self.conf_dict[common._place_pruned_graph] = 0
        self.conf_dict[common._KMP_AFFINITY_granularity] = 1
        self.conf_dict[common._KMP_AFFINITY_respect] = 1
        self.conf_dict[common._KMP_AFFINITY_type] = 0
        self.conf_dict[common._KMP_AFFINITY_permute] = 0
        self.conf_dict[common._KMP_AFFINITY_offset] = 0
        self.conf_dict[common._KMP_BLOCKTIME] = 200
        self.conf_dict[common._OMP_NUM_THREADS] = 1
        self.conf_dict[common._MKL_DYNAMIC] = 0

        self.conf_dict[common._MXNET_CPU_WORKER_NTHREADS] = 1
        self.conf_dict[common._MXNET_CPU_NNPACK_NTHREADS] = 4
        self.conf_dict[common._MXNET_MP_WORKER_NTHREADS] = 1
        self.conf_dict[common._MXNET_EXEC_ENABLE_INPLACE] = 1
        self.conf_dict[common._MXNET_EXEC_NUM_TEMP] = 1
        self.conf_dict[common._MXNET_GPU_MEM_POOL_TYPE] = 0
        self.conf_dict[common._MXNET_ENGINE_TYPE] = 1
        self.conf_dict[common._MXNET_KVSTORE_USETREE] = 0
        self.conf_dict[common._MXNET_UPDATE_ON_KVSTORE] = 1
        self.conf_dict[common._MXNET_CPU_TEMP_COPY] = 4
        self.conf_dict[common._MXNET_GPU_PARALLEL_RAND_COPY] = 4

    def get_tf_cluster_spec(self, node_list, task_index, num_ps):
        workers = []
        pss = []
        for idx, node in enumerate(node_list):
            if idx < num_ps:
                pss.append(node)
            else:
                workers.append(node)
        task_type = "ps" if self.is_ps() else "worker"
        task_index = task_index if self.is_ps() else self.get_worker_index()

        return {
            "cluster": {
                "worker": workers,
                "ps": pss
            },
            "task": {"type": task_type, "index": task_index}
        }

    def get_tf_session_config(self):
        tf_config = tf.compat.v1.ConfigProto(
            inter_op_parallelism_threads=self.get_inter_op_parallelism_threads(),
            intra_op_parallelism_threads=self.get_intra_op_parallelism_threads(),
            allow_soft_placement=True,
            log_device_placement=False,
            graph_options=tf.compat.v1.GraphOptions(
                optimizer_options=tf.compat.v1.OptimizerOptions(
                    do_common_subexpression_elimination=self.get_do_common_subexpression_elimination(),
                    do_constant_folding=True,
                    max_folded_constant_in_bytes=self.get_max_folded_constant_in_bytes(),
                    do_function_inlining=self.get_do_function_inlining(),
                    opt_level=tf.compat.v1.OptimizerOptions.L0,
                    global_jit_level=self.get_global_jit_level()
                ),
                enable_bfloat16_sendrecv=self.get_enable_bfloat16_sendrecv(),
                infer_shapes=self.get_infer_shapes(),
                place_pruned_graph=self.get_place_pruned_graph()
            )
        )
        return tf_config
    
    def set_pytorch_hardware_param(self):
        # reference: https://pytorch.org/docs/stable/torch.html#parallelism
        import torch
        torch.set_num_interop_threads(self.get_inter_op_parallelism_threads())
        torch.set_num_threads(self.get_intra_op_parallelism_threads())
    
    def set_mxnet_hardware_param(self):
        # reference: https://mxnet.apache.org/versions/1.7.0/api/faq/env_var
        os.environ["MXNET_CPU_WORKER_NTHREADS"] = str(self.get_MXNET_CPU_WORKER_NTHREADS())
        os.environ["MXNET_CPU_NNPACK_NTHREADS"] = str(self.get_MXNET_CPU_NNPACK_NTHREADS())
        os.environ["MXNET_MP_WORKER_NTHREADS"] = str(self.get_MXNET_MP_WORKER_NTHREADS())
        os.environ["MXNET_EXEC_ENABLE_INPLACE"] = str(self.get_MXNET_EXEC_ENABLE_INPLACE())
        os.environ["MXNET_EXEC_NUM_TEMP"] = str(self.get_MXNET_EXEC_NUM_TEMP())
        os.environ["MXNET_GPU_MEM_POOL_TYPE"] = str(self.get_MXNET_GPU_MEM_POOL_TYPE())
        os.environ["MXNET_ENGINE_TYPE"] = str(self.get_MXNET_ENGINE_TYPE())
        os.environ["MXNET_KVSTORE_USETREE"] = str(self.get_MXNET_KVSTORE_USETREE())
        os.environ["MXNET_UPDATE_ON_KVSTORE"] = str(self.get_MXNET_UPDATE_ON_KVSTORE())
        os.environ["MXNET_CPU_TEMP_COPY"] = str(self.get_MXNET_CPU_TEMP_COPY())
        os.environ["MXNET_GPU_PARALLEL_RAND_COPY"] = str(self.get_MXNET_GPU_PARALLEL_RAND_COPY())

    def get_MXNET_CPU_WORKER_NTHREADS(self):
        return self.conf_dict[common._MXNET_CPU_WORKER_NTHREADS]
        
    def get_MXNET_CPU_NNPACK_NTHREADS(self):
        return self.conf_dict[common._MXNET_CPU_NNPACK_NTHREADS]
        
    def get_MXNET_MP_WORKER_NTHREADS(self):
        return self.conf_dict[common._MXNET_MP_WORKER_NTHREADS]
        
    def get_MXNET_EXEC_ENABLE_INPLACE(self):
        return str(self.conf_dict[common._MXNET_EXEC_ENABLE_INPLACE] == '1')
        
    def get_MXNET_EXEC_NUM_TEMP(self):
        return self.conf_dict[common._MXNET_EXEC_NUM_TEMP]
        
    def get_MXNET_GPU_MEM_POOL_TYPE(self):
        typ = int(self.conf_dict[common._MXNET_GPU_MEM_POOL_TYPE])
        if typ == 0:
            return 'Naive'
        elif typ == 1:
            return 'Round'
        elif typ == 2:
            return 'Unpooled'
        else:
            raise Exception()
        
    def get_MXNET_ENGINE_TYPE(self):
        typ = int(self.conf_dict[common._MXNET_ENGINE_TYPE])
        if typ == 0:
            return 'NaiveEngine'
        elif typ == 1:
            return 'ThreadedEngine'
        elif typ == 2:
            return 'ThreadedEnginePerDevice'
        else:
            raise Exception()
        
    def get_MXNET_KVSTORE_USETREE(self):
        return self.conf_dict[common._MXNET_KVSTORE_USETREE]
        
    def get_MXNET_UPDATE_ON_KVSTORE(self):
        return self.conf_dict[common._MXNET_UPDATE_ON_KVSTORE]
        
    def get_MXNET_CPU_TEMP_COPY(self):
        return self.conf_dict[common._MXNET_CPU_TEMP_COPY]
        
    def get_MXNET_GPU_PARALLEL_RAND_COPY(self):
        return self.conf_dict[common._MXNET_GPU_PARALLEL_RAND_COPY]
        
    def get_node_list(self):
        return self.conf_dict[common.CONF_DICT_NODE_LIST].split(',')

    def get_task_index(self):
        return self.conf_dict[common.CONF_DICT_TASK_INDEX]

    def get_num_ps(self):
        return self.conf_dict[common._ps_num]

    def get_inter_op_parallelism_threads(self):
        return self.conf_dict[common._inter_op_parallelism_threads]

    def get_intra_op_parallelism_threads(self):
        return self.conf_dict[common._intra_op_parallelism_threads]

    def get_do_common_subexpression_elimination(self):
        return self.conf_dict[common._do_common_subexpression_elimination]

    def get_max_folded_constant_in_bytes(self):
        return self.conf_dict[common._max_folded_constant_in_bytes]

    def get_do_function_inlining(self):
        return self.conf_dict[common._do_function_inlining]

    def get_global_jit_level(self):
        return self.conf_dict[common._global_jit_level]

    def get_enable_bfloat16_sendrecv(self):
        return self.conf_dict[common._enable_bfloat16_sendrecv]

    def get_infer_shapes(self):
        return self.conf_dict[common._infer_shapes]

    def get_place_pruned_graph(self):
        return self.conf_dict[common._place_pruned_graph]
    
    def get_KMP_AFFINITY_granularity(self):
        gran = int(self.conf_dict[common._KMP_AFFINITY_granularity])
        if gran == 1:
            return "core"
        elif gran == 0:
            return "fine"
        else:
            raise Exception()
    
    def get_KMP_AFFINITY_respect(self):
        respect = int(self.conf_dict[common._KMP_AFFINITY_respect])
        if respect == 1:
            return "respect"
        elif respect == 0:
            return "norespect"
        else:
            raise Exception()
    
    def get_KMP_AFFINITY_type(self):
        typ = int(self.conf_dict[common._KMP_AFFINITY_type])
        if typ == 0:
            return "compact"
        elif typ == 1:
            return "scatter"
        else:
            raise Exception()

    def get_KMP_AFFINITY_permute(self):
        return int(self.conf_dict[common._KMP_AFFINITY_permute])

    def get_KMP_AFFINITY_offset(self):
        return int(self.conf_dict[common._KMP_AFFINITY_offset])

    def get_KMP_BLOCKTIME(self):
        return str(self.conf_dict[common._KMP_BLOCKTIME])
    
    def get_OMP_NUM_THREADS(self):
        return str(self.conf_dict[common._OMP_NUM_THREADS])
    
    def get_MKL_DYNAMIC(self):
        dyna = int(self.conf_dict[common._MKL_DYNAMIC])
        if dyna == 1:
            return "TRUE"
        elif dyna == 0:
            return "FALSE"
        else:
            raise Exception()

    def get_num_worker(self):
        return self.conf_dict[common._worker_num]

    def get_worker_index(self):
        task_index = self.get_task_index()
        num_ps = self.get_num_ps()

        if self.is_ps():
            raise Exception("This node is a PS. No worker index")
        return task_index-num_ps

    def is_ps(self):
        task_index = self.get_task_index()
        num_ps = self.get_num_ps()
        return task_index < num_ps

    def report_iter_loss(self, step, loss, time_iter):
        if self.conn is not None:
            self.mq_manager = KombuQueueManager()
            ps_tuner_training_data = [selftf.lib.common.PSTunerTrainingData(
                elapsed_time_in_ms=time_iter,
                loss=float(loss),
                local_step=int(step),
                timestamp=float(time.time()),
                ps_config_idx=0)]
            msg = UpdateNumOfSteps.create(self.agent_id,
                                          self.mq_manager.get_monitor_name(),
                                          ps_tuner_training_data[-1].local_step,
                                          self.agent_job_id,
                                          ps_tuner_training_data_jsons=json.dumps(
                                              ps_tuner_training_data,
                                              cls=PSTunerTrainingDataSerializer))

            self.mq_manager.send_msg_to_monitor(conn=self.conn, message_obj=msg)

    def get_job_id(self):
        return self.agent_job_id

    def get_batch_size(self):
        return self.conf_dict[common._batch_size]

    def get_learning_rate(self):
        return self.conf_dict[common._learning_rate]

    def get_trial_budget(self):
        if 'TRIAL_BUDGET' not in self.conf_dict:
            return None
        else:
            return self.conf_dict['TRIAL_BUDGET']

    def is_chief(self):
        if self.is_ps():
            return False
        return self.get_worker_index() == 0
    
    def get_optimizer(self):
        optm = int(self.conf_dict[common._optimizer])
        if optm == 0:
            return "Adam"
        elif optm == 1:
            return "SGD"
        else:
            raise Exception()
