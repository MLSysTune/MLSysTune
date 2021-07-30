from json import JSONEncoder
from subprocess import call

import tensorflow as tf

from selftf.lib.common import TFVariableContainer


def check_type(obj, chk_type):
    if not isinstance(obj, chk_type):
        raise TypeError()


class TFVariableSeralizer(JSONEncoder):
    def default(self, o):
        if isinstance(o, tf.Variable):
            obj = TFVariableContainer(o)
            return obj.__dict__
        else:
            try:
                return super(TFVariableSeralizer, self).default(o)
            except Exception:
                return o.__dict__

    @staticmethod
    def object_hook(obj):
        try:
            ret = TFVariableContainer()
            ret.__dict__ = obj
            return ret
        except Exception:
            return obj