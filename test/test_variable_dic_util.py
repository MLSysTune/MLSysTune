import unittest
import selftf.monitor
import selftf.lib.common


class TestVariableMapUtil(unittest.TestCase):

  def test_get_device_list_by_num_ps(self):

    expected = [selftf.lib.common.job_prefix + str(0), selftf.lib.common.job_prefix + str(1)]
    ret = selftf.monitor.VariableMapUtil.get_device_list_by_num_ps(2)
    self.assertListEqual(expected, ret)

  def test_calc_new_variable_map_with_more_ps(self):
    variable_map = {
      "a": "/job:selftf/task:0",
      "b": "/job:selftf/task:1",
      "c": "/job:selftf/task:1"
    }

    expected = {
      "a": "/job:selftf/task:0",
      "c": "/job:selftf/task:1",
      "b": "/job:selftf/task:2"
    }

    ret = selftf.monitor.VariableMapUtil._calc_new_variable_map_with_more_ps(variable_map, 3)
    self.assertDictEqual(expected, ret)

  def test_calc_new_variable_map_with_more_ps2(self):
      variable_map = {
        "a": "/job:selftf/task:0",
        "b": "/job:selftf/task:1",
        "c": "/job:selftf/task:1"
      }

      expected = {
        "a": "/job:selftf/task:0",
        "c": "/job:selftf/task:1",
        "b": "/job:selftf/task:2"
      }

      ret = selftf.monitor.VariableMapUtil._calc_new_variable_map_with_more_ps(
        variable_map, 3)
      self.assertDictEqual(expected, ret)

  def test_calc_new_variable_map_with_less_ps(self):
    variable_map = {
      "a": "/job:selftf/task:0",
      "b": "/job:selftf/task:1",
      "c": "/job:selftf/task:2"
    }

    expected = {
      "a": "/job:selftf/task:0",
      "b": "/job:selftf/task:1",
      "c": "/job:selftf/task:0"
    }

    ret = selftf.monitor.VariableMapUtil._calc_new_variable_map_with_less_ps(variable_map, 3, 2)
    self.assertDictEqual(expected, ret)


if __name__ == '__main__':
    unittest.main()