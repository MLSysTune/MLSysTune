# Base class for all message between monitor and computeNode
import importlib
import json
import time
from json import JSONEncoder

import selftf.lib.util
from selftf.lib import util
from selftf.lib.common import Job, PSTunerTrainingDataSerializer
import selftf.lib.common_conf


class Message(object):
    _key_message_type = "message_type"

    def __init__(self):
        self.source = ""
        self.destination = ""
        self.collect_hw_info()

    def set_from_dest(self, source, destination):
        self.source = source
        self.destination = destination

    def get_message_type(self):
        return self.__class__.__name__

    def get_source(self):
        return self.source

    def get_destination(self):
        return self.destination
        
    def collect_hw_info(self):
        import multiprocessing
        import tensorflow as tf
        from psutil import virtual_memory
        self.num_cpu_per_node = multiprocessing.cpu_count()
        self.num_gpu_per_node = len(tf.config.experimental.list_physical_devices('GPU'))
        self.num_mem_per_node = virtual_memory().total


class StartProcess(Message):

    def __init__(self):
        self.args = []
        self.script_location = ""
        self.job_id = ""
        self.env = {}

    @classmethod
    def create(cls, source, destination, job_id, script_location, args, training_status, config_idx,
                env):
        self = cls()
        super(cls, self).set_from_dest(source, destination)

        self.script_location = script_location
        self.job_id = job_id
        self.training_status = training_status
        self.config_idx = config_idx
        self.env = env

        if not isinstance(args, list):
            raise TypeError()
        self.args = args
        return self

    def get_args(self):
        return self.args

    def get_script_location(self):
        return self.script_location

    def get_job_id(self):
        return self.job_id

    def get_training_status(self):
        return self.training_status

    def get_config_idx(self):
        return self.config_idx


class FinishProcess(Message):

        def __init__(self):
            self.job_id = ""
            self.ps_tuner_training_data_jsons = ""

        @classmethod
        def create(cls, source, destination, job_id, ps_tuner_training_data_jsons):
            self = cls()
            super(cls, self).set_from_dest(source, destination)

            self.job_id = job_id
            self.ps_tuner_training_data_jsons = ps_tuner_training_data_jsons
            self.collect_hw_info()
            return self

        def get_job_id(self):
            return self.job_id

        def get_ps_tuner_training_data_jsons(self):
            return self.ps_tuner_training_data_jsons

# class SuccessStartProcess(Message):
#
#     def __init__(self):
#         self.job_id = ""
#         self.pid = 0
#
#     @classmethod
#     def create(cls, source, destination, job_id, pid):
#         self = cls()
#         super(cls, self).set_from_dest(source, destination)
#         self.job_id = job_id
#         self.pid = pid
#         return self


class KillJob(Message):

    def __init__(self):
        self.job_id = 0

    @classmethod
    def create(cls, source, destination, job_id):
        self = cls()
        super(cls, self).set_from_dest(source, destination)
        self.job_id = job_id
        return self

    def get_job_id(self):
        return self.job_id


class TriggerDumpLog(Message):

    def __init__(self):
        self.job_id = 0

    @classmethod
    def create(cls, source, destination, job_id):
        self = cls()
        super(cls, self).set_from_dest(source, destination)
        self.job_id = job_id
        return self

    def get_job_id(self):
        return self.job_id


class KillProcessByJobId(Message):

    def __init__(self):
        self.job_id = 0

    @classmethod
    def create(cls, source, destination, job_id):
        self = cls()
        super(cls, self).set_from_dest(source, destination)
        self.job_id = job_id
        return self

    def get_job_id(self):
        return self.job_id


class RegisterAgent(Message):
    def __init__(self):
        self.tfport = 2222
        self.work_dir = ""
        self.num_thread = 16

    @classmethod
    def create(cls, source, destination, tfport, work_dir, num_thread=16):
        self = cls()
        super(cls, self).set_from_dest(source, destination)

        self.tfport = tfport
        self.work_dir = work_dir
        self.num_thread = num_thread

        return self


class AgentHeartBeat(Message):
    pass


class StartJob(Message):

    def __init__(self):
        self.args = []
        self.script_path = ""
        self.target_loss = 0.5
        self.ml_model = "ML"
        self.batch_size = 1000
        self.learning_rate = 0.001
        self.script_path = ""

        self.optimizer = "Adam"
        self.num_worker = 35
        self.num_intra_thread = 15
        self.n_partition = 1
        self.mode = selftf.lib.common_conf.MODE_VALUE_SELFTF

        self.json_dict_pstuner_config = ""

    @classmethod
    def create(cls, source, destination, script_path, args, json_dict_pstuner_config="",
                target_loss=0.5, ml_model="ML", batch_size=1000,
               learning_rate=0.001, optimizer="Adam", num_worker = 35, num_intra_thread = 15, n_partition = 1,
               mode=selftf.lib.common_conf.MODE_VALUE_SELFTF):
        self = cls()
        super(cls, self).set_from_dest(source, destination)
        if not isinstance(args, list):
            raise TypeError()
        self.args = args
        self.target_loss = target_loss
        self.ml_model = ml_model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.script_path = script_path

        self.optimizer = optimizer
        self.num_worker = num_worker
        self.num_intra_thread = num_intra_thread
        self.n_partition = n_partition
        self.mode = mode
        self.json_dict_pstuner_config = json_dict_pstuner_config
        self.collect_hw_info()
        return self

    def get_args(self):
        return self.args

    def get_script_path(self):
        return self.script_path

    def get_target_loss(self):
        return self.target_loss

    def get_ml_model(self):
        return self.ml_model

    def get_learning_rate(self):
        return self.learning_rate

    def get_batch_size(self):
        return self.batch_size


class ReplyStartJob(Message):

    def __init__(self):
        self.job_id = ""

    @classmethod
    def create(cls, source, destination, job_id):
        self = cls()
        super(cls, self).set_from_dest(source, destination)

        self.job_id = job_id
        return self

    def get_job_id(self):
        return self.job_id


# Trigger monitor to print job statistic
class GetJobStatistic(Message):

    def __init__(self):
        self.job_id = ""

    @classmethod
    def create(cls, source, destination, job_id):
        self = cls()
        super(cls, self).set_from_dest(source, destination)

        self.job_id = job_id
        return self

    def get_job_id(self):
        return self.job_id


class ReplyGetJobStatistic(Message):
    def __init__(self):
        self.job_statistic_json = ""
        self.job_id = ""

    @classmethod
    def create(cls, source, destination, job_obj):
        """
        :param source:
        :param destination:
        :param  Job job_obj:
        :return:
        """
        self = cls()
        super(cls, self).set_from_dest(source, destination)

        util.check_type(job_obj, Job)
        self.job_statistic_json = json.dumps(job_obj.get_training_statistic().get(),
                                             cls=PSTunerTrainingDataSerializer)
        self.job_id = job_obj.get_id()

        return self

    def get_job_statistic(self):
        return json.loads(self.job_statistic_json, object_hook=PSTunerTrainingDataSerializer.object_hook)


class UpdateNumOfSteps(Message):
    def __init__(self):
        self.steps = 0 #deprecated
        self.job_id = ""
        self.ps_tuner_training_data_jsons = ""

    @classmethod
    def create(cls, source, destination, steps, job_id, ps_tuner_training_data_jsons):
        self = cls()
        super(cls, self).set_from_dest(source, destination)

        self.steps = steps
        self.job_id = job_id
        self.ps_tuner_training_data_jsons = ps_tuner_training_data_jsons

        return self

    def get_steps(self):
        """
        deprecated
        """
        return self.steps

    def get_job_id(self):
        return self.job_id

    def get_ps_tuner_training_data_jsons(self):
        return self.ps_tuner_training_data_jsons


class FinishCopyRemoteFile(Message):
    def __init__(self):
        self.job_id = ""

    @classmethod
    def create(cls, source, destination, job_id):
        self = cls()
        super(cls, self).set_from_dest(source, destination)

        self.job_id = job_id

        return self

    def get_job_id(self):
        return self.job_id


class DeRegisterComputeNode(Message):
    @classmethod
    def create(cls, source, destination):
        self = cls()
        super(cls, self).set_from_dest(source, destination)

        return self


class SendRecoveryTime(Message):
    """
    For iteration 1: A SendRecoveryTime messager is still received.
    variable list is received.
    """
    def __init__(self):
        self.job_id = ""
        self.recovery_time_sec = 0

        # hijack here add variable list here
        self.json_variable_str = ""

        # hijack a list of static variable name e.g [global_step,xxxb,xxxxc]
        self.json_non_static_ops_str = ""

        self.conf_idx = -1

        self.reconfig_finish = False

        self.timestamp = time.time()

    @classmethod
    def create(cls, source, destination, job_id, recovery_time_sec,
        json_variable_str=None, json_non_static_ops_str=None, conf_idx=None,
        reconfig_finish=False):
        self = cls()
        super(cls, self).set_from_dest(source, destination)

        self.job_id = job_id
        self.recovery_time_sec = recovery_time_sec
        self.json_variable_str = json_variable_str
        self.json_non_static_ops_str = json_non_static_ops_str
        self.conf_idx = conf_idx
        self.reconfig_finish = reconfig_finish

        return self

    def get_job_id(self):
        return self.job_id

    def get_recovery_time_sec(self):
        return self.recovery_time_sec


class TriggerChiefCheckPoint(Message):
    def __init__(self):
        self.job_id = ""

    @classmethod
    def create(cls, source, destination, job_id):
        self = cls()
        super(cls, self).set_from_dest(source, destination)

        self.job_id = job_id
        return self

    def get_job_id(self):
        return self.job_id


class NaturalFinishML(Message):
    def __init__(self):
        self.job_id = ""
        self.final_cost = 0.9

    @classmethod
    def create(cls, source, destination, job_id, final_cost):
        self = cls()
        super(cls, self).set_from_dest(source, destination)

        self.job_id = job_id
        self.final_cost = final_cost
        return self

    def get_job_id(self):
        return self.job_id

    def get_final_cost(self):
        return self.final_cost


class ReconfigScheme1(Message):
    def __init__(self):
        self.job_id = ""
        self.conf_id = 0
        self.conf_dict = {}
        self.max_iteration = 0
        self.last_step = 0

    @classmethod
    def create(cls, source, destination, job_id, conf_id, conf_dict, max_iteration, last_step):
        self = cls()
        super(cls, self).set_from_dest(source, destination)

        self.job_id = job_id
        self.conf_id = conf_id
        self.conf_dict = conf_dict
        self.max_iteration = max_iteration
        self.last_step = last_step

        return self

    def get_job_id(self):
        return self.job_id

    def get_conf_id(self):
        return self.conf_id

    def get_conf_dict(self):
        return self.conf_dict

    def get_max_iteration(self):
        return self.max_iteration

    def get_last_step(self):
        return self.last_step


class ReconfigScheme2(Message):
    def __init__(self):
        self.job_id = ""
        self.conf_id = 0
        self.conf_dict = {}
        self.max_iteration = 0
        self.last_step = 0

        self.target_reconfig_step = 0

    @classmethod
    def create(cls, source, destination, job_id, conf_id, conf_dict,
        max_iteration, last_step, target_reconfig_step=0):
        self = cls()
        super(cls, self).set_from_dest(source, destination)

        self.job_id = job_id
        self.conf_id = conf_id
        self.conf_dict = conf_dict
        self.max_iteration = max_iteration
        self.last_step = last_step
        self.target_reconfig_step = target_reconfig_step

        return self

    def get_job_id(self):
        return self.job_id

    def get_conf_id(self):
        return self.conf_id

    def get_conf_dict(self):
        return self.conf_dict

    def get_max_iteration(self):
        return self.max_iteration

    def get_last_step(self):
        return self.last_step


# Message for init phase
# When worker finish the init
# 2 case
# If next config is non data reallocated
#   ReconfigScheme1
# else
#   KillThem
#
# Monitor:
#   Wait for all AskForConfig
#
class FinshSubTrainingPhase(Message):
    def __init__(self):
        self.job_id = ""

    @classmethod
    def create(cls, source, destination, job_id):
        self = cls()
        super(cls, self).set_from_dest(source, destination)

        self.job_id = job_id

        return self

    def get_job_id(self):
        return self.job_id


class ChangeConfig(Message):
    def __init__(self):
        self._num_node_change = 0
        self._job_id = ""

    @classmethod
    def create(cls, source, destination, job_id, num_node_change):
        self = cls()
        super(cls, self).set_from_dest(source, destination)

        self._job_id = job_id
        self._num_node_change = num_node_change
        return self

    def get_num_node_change(self):
        return self._num_node_change

    def get_job_id(self):
        return self._job_id


class MessageSerializer(JSONEncoder):
    def default(self, o):
        if isinstance(o, Message):
            obj_dict =  o.__dict__
            obj_dict[Message._key_message_type] = o.get_message_type()
            return obj_dict
        else:
            try:
                return super(MessageSerializer, self).default(o)
            except Exception:
                return o.__dict__

    @staticmethod
    def message_object_hook(obj):
        try:
            cls = get_message_class_from_name(obj[Message._key_message_type])
            instance = cls()
            instance.__dict__ = obj
            return instance
        except Exception:
            return obj

def get_message_class_from_name(name):
    f = __name__
    m = importlib.import_module(f)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, name)
    return c
