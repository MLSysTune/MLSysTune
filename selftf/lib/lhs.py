from pyDOE import *
from selftf.lib import tuner

def LHS_samples(n_feats, test_size):
    X_test = lhs(n_feats, samples=test_size, criterion='center')
    return X_test

class LHSAdapter(object):

    def __init__(self, tf_config_manager):
        """
        :param TensorFlowConfigurationManager tf_config_manager:
        :param dict[string, TensorFlowConfigMetaData] tf_config_meta_data_map:
        :return:
        """
        # self.tf_config_manager = tf_config_manager
        # self.tf_config_meta_data_map = tf_config_manager.config_map
        # self.config_sequence = [
        #     tuner._ps_num,
        #     tuner._intra_op_parallelism_threads
        # ]
        # self.range_matrix = self.get_range_matrix(self.tf_config_meta_data_map)
        self.tf_config_util = tuner.TFConfigUtil(tf_config_manager)
        self.tf_config_meta_data_map = tf_config_manager.config_map

        self.config_sequence = self.tf_config_util.config_sequence
        self.range_matrix = self.tf_config_util.get_range_matrix(self.tf_config_meta_data_map)

    def get_batch_lhs_config(self, num_sample, job_obj):
        """
        :param common.Job job_obj:
        :return:
        """

        x_explore = LHS_samples(len(self.config_sequence), num_sample)
        list_denormalized_config_vector = []
        for i in range(0, len(x_explore)):
            list_denormalized_config_vector.append(self.tf_config_util.denormalize_config_vector_with_range_matrix(x_explore[i]))

        ret = []
        for x in list_denormalized_config_vector:
            ret.append(self.config_vector_to_config_obj(x, job_obj))
        return ret

    def config_vector_to_config_obj(self, config_vector, job_obj):
        """
        :param common.Job job_obj:
        :param  list config_vector:
        :return:
        :rtype: PSTunerConfiguration
        """
        return self.tf_config_util.config_vector_to_config_obj(config_vector,
                                                               job_obj)

    def training_data_to_list_config_vector(self, list_tf_config_training_data):
        """
        :param list[PSTunerTrainingData] list_tf_config_training_data:
        :return:
        """
        return self.tf_config_util.training_data_to_list_config_vector(list_tf_config_training_data)

    def tf_config_to_vector(self, tf_config):
        """
        :param PSTunerConfiguration tf_config:
        :return:
        """
        return self.tf_config_util.tf_config_to_vector(tf_config)
