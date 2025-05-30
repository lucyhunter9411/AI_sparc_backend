# shared_data.py

from collections import defaultdict
from collections import defaultdict, deque

conversations = defaultdict(list)     # robot_id → list[dict]
MAX_TOKENS = 32000

MAX_HISTORY = 5                       # or whatever value you had before
conversations = defaultdict(lambda: deque(maxlen=MAX_HISTORY))

contents = []
time_list = []
hand_raising_count = []
name_array = []
selected_student = []
local_time_vision = []
lecture_states = {}
session_id_set = []

def set_contents(value):
    global contents
    contents = value

def get_contents():
    return contents

def set_time_list(value):
    global time_list
    time_list = value

def get_time_list():
    return time_list

def set_hand_raising_count(robot_id, value):
    global hand_raising_count
    # Check if the robot_id already exists in the list
    for entry in hand_raising_count:
        if robot_id in entry:
            entry[robot_id] = value  # Update existing entry
            return
    # If not found, append a new dictionary
    hand_raising_count.append({robot_id: value})

def get_hand_raising_count(connectrobot):
    for entry in hand_raising_count:
        if connectrobot in entry:
            return entry[connectrobot]  # Return the count for the robot_id
    return 0  # Return 0 if robot_id not found

def set_name_array(robot_id, array):
    global name_array
    # Check if the robot_id already exists in the list
    for entry in name_array:
        if robot_id in entry:
            entry[robot_id] = array  # Update existing entry
            return
    # If not found, append a new dictionary
    name_array.append({robot_id: array})

def get_name_array(connectrobot):
    for entry in name_array:
        if connectrobot in entry:
            return entry[connectrobot]
    return 0  # Return 0 if robot_id not found


def set_selected_student(robot_id, value):
    global selected_student
    # Check if the robot_id already exists in the list
    for entry in selected_student:
        if robot_id in entry:
            entry[robot_id] = value  # Update existing entry
            return
    # If not found, append a new dictionary
    selected_student.append({robot_id: value})

def get_selected_student(robot_id):
    for entry in selected_student:
        if robot_id in entry:
            return entry[robot_id]  # Return the count for the robot_id
    return 0  # Return 0 if robot_id not found

def set_local_time_vision(robot_id, value):
    global local_time_vision
    # Check if the robot_id already exists in the list
    for entry in local_time_vision:
        if robot_id in entry:
            entry[robot_id] = value  # Update existing entry
            return
    # If not found, append a new dictionary
    local_time_vision.append({robot_id: value})

def get_local_time_vision(robot_id):
    for entry in local_time_vision:
        if robot_id in entry:
            return entry[robot_id]  # Return the count for the robot_id
    return 0  # Return 0 if robot_id not found

def set_session_id_set(robot_id, value):
    global session_id_set
    for entry in session_id_set:
        if robot_id in entry:
            entry[robot_id] = value
            return
    # If not found, append a new dictionary
    session_id_set.append({robot_id: value})

def get_session_id_set(robot_id):
    for entry in session_id_set:
        if robot_id in entry:
            return entry[robot_id]
    return None

def set_lecture_states(value):
    global lecture_states
    lecture_states = value

def get_lecture_states():
    return lecture_states