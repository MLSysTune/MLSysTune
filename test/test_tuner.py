from selftf.lib.tuner import PSTunerTrainingData, PSTunerConfiguration, GPTrainingModel, TensorFlowConfigurationManager, \
    TFConfigUtil
import logging
import unittest


class TestTuner(unittest.TestCase):
    def test_gp(self):
        c1 = PSTunerConfiguration(
            num_ps=2,
            num_worker=2,
            inter_op_parallelism_threads=8,
            intra_op_parallelism_threads=8
        )
        c2 = PSTunerConfiguration(
            num_ps=1,
            num_worker=3,
            inter_op_parallelism_threads=14,
            intra_op_parallelism_threads=2
        )
        c3 = PSTunerConfiguration(
            num_ps=3,
            num_worker=1,
            inter_op_parallelism_threads=2,
            intra_op_parallelism_threads=14
        )
        d1 = PSTunerTrainingData(
            ps_config=c1,
            elapsed_time_in_ms=1,
            loss=0.3,
            step=1
        )
        d2 = PSTunerTrainingData(
            ps_config=c2,
            elapsed_time_in_ms=2,
            loss=0.2,
            step=2
        )
        d3 = PSTunerTrainingData(
            ps_config=c3,
            elapsed_time_in_ms=3,
            loss=0.2,
            step=3
        )

        list_training_data = [d1, d2, d3]

        tfcm = TensorFlowConfigurationManager(lambda: 4, lambda: 16, (0.00001, 0.0001), (1000, 10000))
        tcu = TFConfigUtil(tfcm)

        gp = GPTrainingModel(tcu)
        gp.train(list_training_data)

        config_obj = gp.get_best_config()

        print(str(config_obj))

        print("Finish")

    def test_get_lastest_training_data_group_by_ps_config(self):
        c1 = PSTunerConfiguration(
            num_ps=2,
            num_worker=2,
            inter_op_parallelism_threads=8,
            intra_op_parallelism_threads=8
        )
        c2 = PSTunerConfiguration(
            num_ps=1,
            num_worker=3,
            inter_op_parallelism_threads=14,
            intra_op_parallelism_threads=2
        )
        d1 = PSTunerTrainingData(
            ps_config=c1,
            elapsed_time_in_ms=1,
            loss=0.3,
            step=1
        )
        d2 = PSTunerTrainingData(
            ps_config=c1,
            elapsed_time_in_ms=2,
            loss=0.2,
            step=2
        )
        d3 = PSTunerTrainingData(
            ps_config=c2,
            elapsed_time_in_ms=3,
            loss=0.2,
            step=3
        )
        d4 = PSTunerTrainingData(
            ps_config=c2,
            elapsed_time_in_ms=3,
            loss=0.2,
            step=4
        )
        d5 = PSTunerTrainingData(
            ps_config=c1,
            elapsed_time_in_ms=1,
            loss=0.3,
            step=5
        )
        d6 = PSTunerTrainingData(
            ps_config=c1,
            elapsed_time_in_ms=2,
            loss=0.2,
            step=6
        )

        list_training_data = [d1,d2,d3,d4,d5,d6]
        ret = TFConfigUtil.get_lastest_training_data_group_by_ps_config(list_training_data)
        assert(len(ret[c1]) == 2)
        assert(len(ret[c2]) == 2)

    def eq_pstuner_config(self):
        c1 = PSTunerConfiguration(
            num_ps=2,
            num_worker=2,
            inter_op_parallelism_threads=8,
            intra_op_parallelism_threads=8
        )
        c2 = PSTunerConfiguration(
            num_ps=2,
            num_worker=2,
            inter_op_parallelism_threads=8,
            intra_op_parallelism_threads=8
        )

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            )
    unittest.main()