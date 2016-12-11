"""
README:
The appropriate numpy arrays are stored in human3.6.npz
Each npz file contains a bunch of numpy arrays. Each numpy array is shaped as follows:

    s_time: (num_samples, NUM_TIMESTEPS, 2 * NUM_S_FEATURES)
    s_inp: (num_samples, NUM_TIMESTEPS, NUM_S_FEATURES)
    s_out: (num_samples, NUM_TIMESTEPS, NUM_S_FEATURES)
    a_time: (2 * num_samples, NUM_TIMESTEPS, 2 * NUM_A_FEATURES)
    a_inp: (2 * num_samples, NUM_TIMESTEPS, NUM_A_FEATURES)
    a_out: (2 * num_samples, NUM_TIMESTEPS, NUM_A_FEATURES)
    l_time: (2 * num_samples, NUM_TIMESTEPS, 2 * NUM_L_FEATURES)
    l_inp: (2 * num_samples, NUM_TIMESTEPS, NUM_L_FEATURES)
    l_out: (2 * num_samples, NUM_TIMESTEPS, NUM_L_FEATURES)
    s_a: (num_samples, NUM_TIMESTEPS, NUM_S_FEATURES + NUM_A_FEATURES)
    s_l: (num_samples, NUM_TIMESTEPS, NUM_S_FEATURES + NUM_L_FEATURES)
    a_s: (2 * num_samples, NUM_TIMESTEPS, NUM_S_FEATURES + NUM_A_FEATURES)
    l_s: (2 * num_samples, NUM_TIMESTEPS, NUM_S_FEATURES + NUM_L_FEATURES)
    a_a: (2 * num_samples, NUM_TIMESTEPS, 2 * NUM_A_FEATURES)
    l_l: (2 * num_samples, NUM_TIMESTEPS, 2 * NUM_L_FEATURES)

    inp: (num_samples, NUM_TIMESTEPS, NUM_ALL_FEATURES)
    out: (num_samples, NUM_TIMESTEPS, NUM_ALL_FEATURES)

    training data goes from [0, val_index)
    validation data goes from [val_index, len())

For the ones with multiple edges (s-a, s-l), the multiple edges are summed together to get the final result
The s-a, s-l, a-s, l-s inputs are created by concatenating the SPINE first with the arm/leg second
The a-a and l-l inputs are created by concatenating the LEFT one first with the RIGHT one second
The _in vectors are the current time step, the _out vectors are the next time step
Even indices in the arm/leg arrays are LEFT, odd indices are RIGHT
Temporal edges are made by concatenating INPUT with OUTPUT
"""

import xml.etree.ElementTree as ET
from spacepy import pycdf
import numpy as np
import os
import math
from transforms3d import euler

NUM_JOINTS = 32
USERS = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
ACTIVITIES = ['Discussion', 'Eating', 'Smoking', 'Walking']
# USERS = ['S1']
# ACTIVITIES = ['Discussion']
NUM_A_FEATURES = 24
NUM_L_FEATURES = 15
NUM_S_FEATURES = 18
NUM_ALL_FEATURES = 96
NUM_TIMESTEPS = 150

DATA_ROOT = 'Human3.6_data'

def parse_metadata():
    tree = ET.parse('metadata.xml')
    skel_angles = tree.getroot()[3][6]
    # rotInd_array[joint_id] gives the rotInd of the joint with that id
    rotInd_array = [0] * NUM_JOINTS
    for i in range(NUM_JOINTS):
        # name = skel_angles[i][0].text (in case you wanna know)
        id = int(skel_angles[i][1].text)
        rot_ind = skel_angles[i][13].text
        if rot_ind != None:
            rot_ind = map(int, rot_ind[1:-1].split())
        else:
            rot_ind = []
        rotInd_array[id] = rot_ind
    return rotInd_array

def convert(degree):
    return math.radians(degree % 360)
def exp_map(v):
    vec, angle = euler.euler2axangle(*map(convert, v))
    return angle * vec

def downsize(array):
    return array[::2]
def extract(array, start, end):
    return array[:,:,start:end].reshape((len(array), len(array[0]), 3 * (end - start)))
def get_spine(array):
    return np.append(array[:,:,1:2], array[:,:,11:16], axis=2).reshape((len(array), len(array[0]), 18))

def get_node_features_for_one(file, rotInd_array):
    cdf = pycdf.CDF(file)
    cdf = downsize(cdf['Pose'][0])
    total_timesteps = len(cdf)
    num_pieces = total_timesteps / NUM_TIMESTEPS
    joint_array = np.zeros((num_pieces, NUM_TIMESTEPS, NUM_JOINTS, 3))
    for piece in range(num_pieces):
        for t in range(NUM_TIMESTEPS):
            for joint_id in range(NUM_JOINTS):
                assert len(rotInd_array[joint_id]) == 0 or len(rotInd_array[joint_id]) == 3
                for i in range(len(rotInd_array[joint_id])):
                    rotInd = rotInd_array[joint_id][i]
                    joint_array[piece][t][joint_id][i] = cdf[piece * NUM_TIMESTEPS + t][rotInd-1]
                joint_array[piece][t][joint_id] = exp_map(joint_array[piece][t][joint_id])
    # joint_array[piece][time][joint_id] gives the list of three rotations (z, x, y)
    # now create the node features
    everything = extract(joint_array, 0, 32)
    left_arm = extract(joint_array, 16, 24)
    right_arm = extract(joint_array, 24, 32)
    left_leg = extract(joint_array, 6, 11)
    right_leg = extract(joint_array, 1, 6)
    spine = get_spine(joint_array)
    # Each of these[piece][time] gives a list of features
    return left_arm, right_arm, left_leg, right_leg, spine, everything

def make_inp_out_time(inp, stride):
    out = np.append(inp[stride:], np.zeros((stride, NUM_TIMESTEPS, len(inp[0][0]))), axis=0)
    time = np.append(inp, out, axis=2)
    return inp, out, time

def normalize(array):
    return (array - np.mean(array)) / np.std(array)

def get_node_features(activity):
    rotInd_array = parse_metadata()
    a, l, s, e = [], [], [], []
    val_index = 0
    for user in USERS:
        if (user == 'S11'):
            val_index = len(s)
        directory = os.path.join(DATA_ROOT, activity + '/' + user + '/MyPoseFeatures/D3_Angles/')
        for file in os.listdir(directory):
            filename = os.path.join(directory, file)
            left_arm, right_arm, left_leg, right_leg, spine, everything = get_node_features_for_one(filename, rotInd_array)
            for piece in range(len(spine)):
                a.append(left_arm[piece])
                a.append(right_arm[piece])
                l.append(left_leg[piece])
                l.append(right_leg[piece])
                s.append(spine[piece])
                e.append(everything[piece])
    a_inp, a_out, a_time = make_inp_out_time(np.asarray(a), 2)
    l_inp, l_out, l_time = make_inp_out_time(np.asarray(l), 2)
    s_inp, s_out, s_time = make_inp_out_time(np.asarray(s), 1)
    inp, out, time = make_inp_out_time(np.asarray(e), 1)

    num_samples = len(s_inp)

    s_a = np.append(s_inp, a_inp[::2], axis=2)
    s_a += np.append(np.zeros((num_samples, NUM_TIMESTEPS, NUM_S_FEATURES)), a_inp[1::2], axis=2)

    s_l = np.append(s_inp, l_inp[::2], axis=2)
    s_l += np.append(np.zeros((num_samples, NUM_TIMESTEPS, NUM_S_FEATURES)), l_inp[1::2], axis=2)

    a_s = np.append(a_inp, np.repeat(s_inp, 2, axis=0), axis=2)
    l_s = np.append(l_inp, np.repeat(s_inp, 2, axis=0), axis=2)

    a_a = np.repeat(np.append(a_inp[::2], a_inp[1::2], axis=2), 2, axis=0)
    l_l = np.repeat(np.append(l_inp[::2], l_inp[1::2], axis=2), 2, axis=0)

    a_inp = normalize(a_inp)
    a_out = normalize(a_out)
    a_time = normalize(a_time)
    l_inp = normalize(l_inp)
    l_out = normalize(l_out)
    l_time = normalize(l_time)
    s_inp = normalize(s_inp)
    s_out = normalize(s_out)
    s_time = normalize(s_time)
    inp = normalize(inp)
    out = normalize(out)
    time = normalize(time)
    s_a = normalize(s_a)
    s_l = normalize(s_l)
    a_s = normalize(a_s)
    l_s = normalize(l_s)
    a_a = normalize(a_a)
    l_l = normalize(l_l)

    filename = "human3.6_" + activity + ".npz"

    with open(filename, 'w') as f:
        np.savez(f, a_inp=a_inp, a_out=a_out, a_time=a_time, l_inp=l_inp, l_out=l_out, l_time=l_time, s_inp=s_inp, 
            s_out=s_out, s_time=s_time, inp=inp, out=out, time=time, s_a=s_a, s_l=s_l, a_s=a_s, l_s=l_s, a_a=a_a, l_l=l_l, val_index=val_index)
        f.close()

if __name__ == '__main__':
    for activity in ACTIVITIES:
        get_node_features(activity)
