import grequests as requests

from selftf.lib import common

host = 'http://120.24.163.35:8080'
test_job_id = "cnn_1002"

session = requests.Session()

def add_job(job_id):
    return
    r = session.get("{}/add_job/{}".format(host, job_id))
    # if r.text != 'ok':
    #     print('add job fail')
    #     return False
    return True

def add_pt(job_id, x_value, y_value, type=0):  # type=0: default, type=1: live
    return
    r = session.get(
        "{}/add_point/{}/{}/{}/{}".format(host, job_id, type, x_value, y_value))
    # if r.text != 'ok':
    #     print('add point fail')
    #     return False
    return True


def change_status(job_id, x_value, status): # status can be one of 4 status or other comments
    return
    if status in [common._JOB_STATUS_finish, common._JOB_STATUS_executing, common._JOB_STATUS_initialize, common._JOB_STATUS_checkingpointing]:
        color_type = 1
    else:
        color_type = 0
    r = session.get(
        "{}/status/{}/{}/{}/{}".format(host, job_id, color_type, x_value, status))
    # if r.text != 'ok':
    #     print('add status fail')
    #     return False
    return True


def clear():
    return
    session.get("{}/clear".format(host))