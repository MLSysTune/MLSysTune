import datetime
import json
import logging
import os
import shutil
import sys
import time
import tensorflow as tf
from typing import List
from pyDOE import *
import subprocess
import networkx as nx
import numpy
import csv

from selftf.lib.common import PSTunerTrainingData, \
    PSTunerTrainingDataSerializer, Job, PSTunerConfiguration, \
    random_generating_config, get_framework
from .gpr import GPRGD
from selftf.lib.tuner import TensorFlowConfigurationManager, TFConfigUtil
from .. import common
from selftf.lib.common import PSTunerConfiguration, PSTunerTrainingData, \
    _ps_num, _intra_op_parallelism_threads, _inter_op_parallelism_threads, \
    _do_common_subexpression_elimination, _max_folded_constant_in_bytes, \
    _do_function_inlining, _global_jit_level, _infer_shapes, \
    _enable_bfloat16_sendrecv, _place_pruned_graph, \
    _KMP_AFFINITY_granularity,_KMP_AFFINITY_respect,_KMP_AFFINITY_type,_KMP_AFFINITY_permute, \
    _KMP_AFFINITY_offset,_KMP_BLOCKTIME,_OMP_NUM_THREADS,_MKL_DYNAMIC, \
    _MXNET_CPU_WORKER_NTHREADS, _MXNET_CPU_NNPACK_NTHREADS, _MXNET_MP_WORKER_NTHREADS, \
    _MXNET_EXEC_ENABLE_INPLACE, _MXNET_EXEC_NUM_TEMP, _MXNET_GPU_MEM_POOL_TYPE, \
    _MXNET_ENGINE_TYPE, _MXNET_KVSTORE_USETREE, _MXNET_UPDATE_ON_KVSTORE, _MXNET_CPU_TEMP_COPY, \
    _MXNET_GPU_PARALLEL_RAND_COPY
import copy
import heapq
import subprocess
import networkx as nx
import prettytable as pt
import pymysql

MINIMUM_POINTS_TO_BUILD_MODEL = 5

class MLTuner:

    def __init__(self, path_repo:str,
                config_manager: TensorFlowConfigurationManager,
                job_obj: Job):
        self.gpr_sample_size = 10
        self.path_repo = path_repo
        self.config_manager = config_manager
        self.job_obj = job_obj
        #hack job obj here
        job_obj.num_iter_per_os = sys.maxsize

        
        self.list_training_data:List[PSTunerTrainingData] = []

        self.timestamp_last_get_config = 0

        # self.load_history()

        # init tfboard data
        if os.path.exists(self.get_tfboard_file()):
            shutil.rmtree(self.get_tfboard_file())

        self.tfsummary = tf.summary.create_file_writer(
            self.get_tfboard_file()
        )

        if not os.path.exists(self.path_repo):
            os.mkdir(self.path_repo)

    def get_tfboard_file(self):
        return os.path.join(self.path_repo, "progress")

    def get_repo_file(self):
        return os.path.join(self.path_repo, "history.csv")

    # Can return None to stop tuning
    def get_next_config(self, hw_info, mysql_account, mysql_password, mysql_dbname):
        logging.info("Classify input program and load configurations from database")
        conn = pymysql.connect(host="ssd2",user=mysql_account,passwd=mysql_password,db=mysql_dbname)
        framework = get_framework(self.job_obj.get_args()[0])
        if framework is None:
            framework = "mxnet"
        classified_result = self.classify_input_program(conn, self.job_obj, hw_info, framework)
        classified_training_data = self.query_history_from_db(model_info=classified_result, conn=conn, hw_info=hw_info)
        conn.close()
        logging.info(f"Read {len(classified_training_data)} records from database to build GPR model for BO")
        if len(classified_training_data) < MINIMUM_POINTS_TO_BUILD_MODEL:
            logging.info("Not enough matching models found for the input program. Train with a random-generated config.")
            numpy.random.seed()
            next_config = random_generating_config(framework)
            redundant_para_list = [
                'worker_num',
                'optimizer',
                'learning_rate',
                'batch_size',
                'global_jit_level'
            ]
            for para in redundant_para_list:
                if para in next_config:
                    del next_config[para]
            next_config['num_ps'] = next_config.pop('ps_num')
            return PSTunerConfiguration(
                num_worker=self.config_manager.get_num_of_node_func() - next_config['num_ps'],
                learning_rate=self.job_obj.learning_rate,
                batch_size=self.job_obj.batch_size,
                optimizer=self.job_obj.optimizer,
                **next_config
            )
        else:
            mode =  'GPU' if hw_info['num_gpu_per_node'] > 0 else 'CPU'
            next_config = self.get_tf_config_by_gp(classified_training_data, self.gpr_sample_size)
            return next_config

    def query_history_from_log(self):
        results = []
        for path in os.listdir(os.path.join(os.getenv("SELFTF_HOME"),'log')):
            if not path.startswith('vgg16'):
                continue
            with open(os.path.join(os.getenv("SELFTF_HOME"), 'log', path, 'summary.csv')) as f:
                f_csv = csv.DictReader(f)
                for row in f_csv:
                    record = json.loads(row['config'].replace("'","\""))
                    for item in ['model', 'init_os', 'init_num_iter', 'online_os', 'estimation_func', 'online_num_iter', 'run_mode']:
                        del record[item]
                    if float(row['runtime_sec']) < 3000:
                        results.append({
                            'ps_config': record,
                            'elapsed_time_in_ms': float(row['runtime_sec'])*1000.0, 
                            'loss': 0.0
                        })
        training_data = [json.loads(str(item).replace("'","\""), object_hook=PSTunerTrainingDataSerializer.object_hook) for item in results]
        return training_data

    def query_history_from_ray_summary(self):
        length = None
        summary_path = os.path.join(os.getenv("SELFTF_HOME"), "ray_table.log")
        with open(summary_path, "r") as f:
            length = len(f.readlines())
        with open(summary_path, "r") as f:
            reader = csv.reader(f)
            results = []
            for item in reader:
                if reader.line_num == 2:
                    title = [s.strip() for s in item[0].split('|') if s is not '']
                    continue
                elif reader.line_num < 4 or reader.line_num == length:
                    continue
                raw_record = [s.strip() for s in item[0].split('|') if s is not '']
                record = {}
                for i in range(3, len(title)-3):
                    record[title[i]] = raw_record[i]
                record['worker_num'] = 10-int(record['ps_num'])
                results.append({
                    'ps_config': record,
                    'elapsed_time_in_ms': float(raw_record[-1])*1000.0, 
                    'loss': 0.0
                })
        training_data = [json.loads(str(item).replace("'","\""), object_hook=PSTunerTrainingDataSerializer.object_hook) for item in results]
        return training_data

    def classify_input_program(self, conn, job_obj, hw_info, framework):

        def graph_node_equal(node1,node2):
            n1_attr = node1['label'].split(':')[1].strip()
            n2_attr = node2['label'].split(':')[1].strip()
            return n1_attr == n2_attr
        
        python_exec = job_obj.get_script()
        job_args = job_obj.get_args()

        for arg in job_args:
            if ".py" in arg:
                script_path = arg
                break

        cmd = copy.deepcopy(job_args)
        
        cmd.insert(0,python_exec)
        cmd.extend(["--get_model=True", "--script_path=%s" % script_path])
        logging.info(f"[Model Matching] cmd: {cmd}")

        proc = subprocess.Popen(cmd,stdout=subprocess.PIPE,shell=False)
        logging.info("[Model Matching] get model process start, process ID is %d" % proc.pid)
        proc.wait()
        script_arr = script_path.strip().split("/")
        graph_file = script_path.replace(script_arr[-1],"graph.json")
        with open(graph_file,'r') as uf:
            uf_dict = json.load(uf)
            logging.info("[Model Matching] load user's model in json format")
        if "links" not in uf_dict:
            uf_dict = uf_dict["graph"]
        user_graph = nx.readwrite.json_graph.node_link_graph(uf_dict)
        
        model_info = str(uf_dict)
        min_cost_graph = None
        matching_result_list = []
        cursor = conn.cursor()
        query_arg = f"SELECT DISTINCT model_information FROM HISTORY WHERE num_node='{hw_info['num_node']}' AND num_cpu_per_node='{hw_info['num_cpu_per_node']}' AND num_gpu_per_node='{hw_info['num_gpu_per_node']}' AND num_mem_per_node='{hw_info['num_mem_per_node']}' AND framework='{framework}'"
        cursor.execute(query_arg)
        results = cursor.fetchall()

        for model in results:
            load_dict = eval(model[0])
            template_graph = nx.readwrite.json_graph.node_link_graph(load_dict)
            re = list(nx.algorithms.similarity.optimize_edit_paths(user_graph, template_graph, node_match=graph_node_equal,timeout=20))
            current_min_cost = float("inf")
            for r in re:
                if r[2] < current_min_cost:
                    current_min_cost = r[2]
                    min_cost_graph = load_dict
            matching_result_list.append(current_min_cost)
            if current_min_cost <= min(matching_result_list):
                model_info = model[0]
        if len(matching_result_list) > 0:
            if min(matching_result_list) < 10:
                # regard as the same graph
                with open(graph_file,'w') as dump_f:
                    json.dump(min_cost_graph, dump_f)
            table = pt.PrettyTable()
            table.add_column('GED Cost', matching_result_list)
            logging.info(table)
        model_list = ["vgg16","NLP_transformer","resnet50","NLP_lstm","NLP_bert","mobilenetv2_1","pt_mnasnet","mxnet_resnet20","mxnet_wideresnet16","pt_vgg11","pt_mobilenet","pt_squeezenet"]
        model_to_compare = "NLP_bert" # use config from a specific model
        # model_info = results[model_list.index(model_to_compare)][0]
        return model_info

    def get_tf_config_by_gp(self, list_training_data:List[PSTunerTrainingData],
                            sample_size):
        #
        # from gevent import monkey
        # monkey.patch_all()
        # import pydevd_pycharm
        # pydevd_pycharm.settrace('127.0.0.1', port=4445, stdoutToServer=True,
        #                         stderrToServer=True)

        customize_config_sequence = list(vars(list_training_data[0].ps_config).keys())
        redundant_para_list = [
            'worker_num',
            'optimizer',
            'learning_rate',
            'batch_size',
            'global_jit_level'
        ]
        for para in redundant_para_list:
            if para in customize_config_sequence:
                customize_config_sequence.remove(para)
        isFullConfig = True
        if set(self.config_manager.config_map.keys()) != set(customize_config_sequence):
            for key in list(self.config_manager.config_map.keys()):
                if key not in customize_config_sequence:
                    self.config_manager.config_map.pop(key)
            isFullConfig = False
        self.tf_config_util:TFConfigUtil = TFConfigUtil(self.config_manager, customize_config_sequence)
        training_x, training_y = self.tf_config_util.training_data_to_config_output_vector(
            list_training_data)
        normalized_training_data = self.tf_config_util.normalize_list_config_vector_without_l(
            training_x)
        normalized_training_y = self.tf_config_util.normalize_y(training_y)
        #Train the model
        gpr_gd = GPRGD(length_scale=2.0,
              magnitude=1.0,
              max_train_size=2000,
              batch_size=100,
              num_threads=4,
              learning_rate=0.01,
              epsilon=1e-6,
              max_iter=500,
              sigma_multiplier=1.0,
              mu_multiplier=1.0,
              ridge=1.0,
              debug=False,
              hyperparameter_trainable=True)

        minX, maxX = self.tf_config_util.get_normalized_min_max_vectors()

        gpr_gd.fit(normalized_training_data, normalized_training_y,
                   minX, maxX) #TODO tune this parameter

        top10_list_training_data = heapq.nsmallest(10, list_training_data, key=lambda s: s.elapsed_time_in_ms)
        top10_x, top10_y = self.tf_config_util.training_data_to_config_output_vector(
            top10_list_training_data)
        normalized_top10_training_data = self.tf_config_util.normalize_list_config_vector_without_l(
            top10_x)

        # use the model here
        # X_sample = lhs(len(minX), samples=sample_size, criterion='center')
        # X_sample = np.random.rand(sample_size, len(minX),)
        # for entry in normalized_top10_training_data:
            # Tensorflow get broken if we use the training data points as
            # starting points for GPRGD.
            # X_sample = np.vstack((X_sample, np.array(entry) * 0.97 + 0.01))

        X_sample = np.array([item * 0.97 + 0.01 for item in normalized_top10_training_data])
        # res = gpr_gd.predict(X_sample, constraint_helper=self.tf_config_util.constraint_helper)
        res = gpr_gd.predict(X_sample)

        logging.info(f"res.minl:{res.minl.ravel()}")
        best_gpr_result_training_time = np.min(res.minl.ravel())
        best_config_idx = np.argmin(res.minl.ravel())
        logging.info(f"best_gpr_result_training_time in normalized form:{best_gpr_result_training_time:.3f}, best_config_idx:{best_config_idx} ")
        next_setting = res.minl_conf[best_config_idx, :]

        logging.info("Raw gp ret:%s " % str(next_setting))

        # Check if the config really better than history
        best_history_index, best_history_training_time, best_history_config = \
            get_best_training_time_and_config(list_training_data)
        normalized_best_history_training_time = normalized_training_y[best_history_index][0]
        logging.info(f"best_history_training_time: {best_history_training_time/1000.0}s")
        if normalized_best_history_training_time < best_gpr_result_training_time:
            next_setting = best_history_config
            logging.info("Best config is seen, {}".str(best_history_config))
        if isFullConfig:
            next_setting_config_obj = self.tf_config_util.config_vector_to_config_obj(
                self.tf_config_util.denormalize_config_vector(next_setting),
                self.job_obj
            )
        else:
            config_vector = self.tf_config_util.denormalize_config_vector(next_setting)
            next_config_dict = {}
            for key in customize_config_sequence:
                next_config_dict[key] = config_vector[customize_config_sequence.index(key)]
            next_config_dict['num_ps'] = next_config_dict.pop('ps_num')
            next_setting_config_obj = PSTunerConfiguration(
                num_worker=self.config_manager.get_num_of_node_func() - config_vector[customize_config_sequence.index(_ps_num)],
                learning_rate=self.job_obj.learning_rate,
                batch_size=self.job_obj.batch_size,
                optimizer=self.job_obj.optimizer,
                **next_config_dict
            )
        logging.info(
            "Best configuration from GPR: %s" % str(next_setting_config_obj))
        return next_setting_config_obj


    def load_history(self):
        self.load_history_from_file()

    def add_config_result(self, training_data:PSTunerTrainingData):
        self.list_training_data.append(training_data)
        self.add_config_result_to_file(training_data)

    def load_history_from_file(self):
        path_file = self.get_repo_file()
        if not os.path.exists(path_file):
            return
        with open(path_file, "r") as f:
            for line in f.readlines():
                self.list_training_data.append(
                    json.loads(line, object_hook=PSTunerTrainingDataSerializer.object_hook)
                )
    
    def query_history_from_db(self, model_info:str, conn, hw_info):
        model_info = model_info.replace("'", "\\'")
        cursor = conn.cursor()
        query_arg = f"SELECT ps_config_result_metrics FROM HISTORY WHERE model_information = '{model_info}' AND num_node='{hw_info['num_node']}' AND num_cpu_per_node='{hw_info['num_cpu_per_node']}' AND num_gpu_per_node='{hw_info['num_gpu_per_node']}' AND num_mem_per_node='{hw_info['num_mem_per_node']}'"
        cursor.execute(query_arg)
        results = cursor.fetchall()
        training_data = [json.loads(item[0].replace("'","\""), object_hook=PSTunerTrainingDataSerializer.object_hook) for item in results]
        return training_data

    def add_config_result_to_file(self, training_data:PSTunerTrainingData):
        with open(self.get_repo_file(), "a") as f:
            f.write(json.dumps(training_data, cls=PSTunerTrainingDataSerializer))
            f.write("\n")

    def get_config_duration(self):
        return time.time() - self.timestamp_last_get_config # DEBUG: no 'set' operation for self.timestamp_last_get_config

    def updateNumOfStepsHandler(self, loss, step):
        return
        summary = tf.Summary(value=[tf.Summary.Value(tag="loss",
                    simple_value=loss)])
        self.tfsummary.add_summary(summary,step)

    def close(self):
        self.tfsummary.close()

def get_best_training_time_and_config(list_training_data:List[PSTunerTrainingData]):
    min_time = sys.maxsize
    target_training_data = None
    best_index = 0
    for idx in range(len(list_training_data)):
        x = list_training_data[idx]
        if x.elapsed_time_in_ms < min_time:
            min_time = x.elapsed_time_in_ms
            target_training_data = x
            best_index = idx

    return best_index, min_time, target_training_data

def get_start_job_env(pstunerConfig:PSTunerConfiguration, node_list_str, task_index:int,
                      job_id):
    ret = pstunerConfig.__dict__.copy()

    ret[common.CONF_DICT_NODE_LIST] = node_list_str
    ret[common.CONF_DICT_TASK_INDEX] = task_index
    ret[common.CONF_DICT_JOB_ID] = job_id

    return {common.ENV_KEY_MLTUNER_CONF_DICT: json.dumps(ret)}
