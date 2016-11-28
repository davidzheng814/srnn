"""
README:
The appropriate numpy arrays are stored in numpy_arrays/'10-digit-ID'.npz.
Each npz file contains a bunch of numpy arrays. Each numpy array is shaped as follows:

    h_inputs: (NUM_HUMANS, num_segments, NUM_H_FEATURES)
    ht_inputs: (NUM_HUMANS, num_segments, NUM_HT_FEATURES)
    ho_inputs: (NUM_HUMANS, num_segments, NUM_HO_FEATURES)
    h_outputs: (NUM_HUMANS, num_segments, num_act_classes)
    o_inputs: (num_objects, num_segments, NUM_O_FEATURES)
    ot_inputs: (num_objects, num_segments, NUM_OT_FEATURES)
    oo_inputs: (num_objects, num_segments, 2 * NUM_OO_FEATURES)
    oh_inputs: (num_objects, num_segments, NUM_HO_FEATURES)
    o_outputs: (num_objects, num_segments, num_afford)

For the ones with multiple edges (ho, ot, oo), the multiple edges are summed together to get the final result
The oo_inputs are created by concatenating the outgoing edge first with the ingoing edge second (and then summed together)
The h_outputs and o_outputs are one-hot vectors
"""

import numpy as np
import argparse
import os

FORMAT_ROOT = 'segments_svm_format'
DATA_ROOT = 'features_binary_svm_format'
NUMPY_ROOT = 'numpy_arrays'

NUM_O_FEATURES = 180
NUM_H_FEATURES = 630
NUM_OO_FEATURES = 200
NUM_HO_FEATURES = 400
NUM_OT_FEATURES = 40
NUM_HT_FEATURES = 160
NUM_HUMANS = 1

# Returns a dictionary that maps an activity ID to the number of segments (timesteps) in that activity
def get_activity_to_num_segments():
    activity_to_num_segments = {}
    for filename in os.listdir(FORMAT_ROOT):
        with open(os.path.join(FORMAT_ROOT, filename)) as f:
            activity_to_num_segments[filename[:-4]] = int(f.readline().split()[0])
    return activity_to_num_segments

# Given a list of lines stripped of \n from the txt file, return a list of 
def parse_segment_file(lines):
    t = iter(lines)
    num_objects_copy, num_oo, num_ho, num_afford_copy, num_act_classes_copy, segment_num = map(int, t.next().split())
    assert num_objects_copy == num_objects
    assert num_oo == num_objects * (num_objects - 1)
    assert num_ho == num_objects
    assert num_afford_copy == num_afford
    assert num_act_classes_copy == num_act_classes

    # o_nodes[object_id][feature_id]
    # <affordance_class> <object_id> <feature_num>:<feature_value> ... 
    o_nodes = np.zeros((num_objects, NUM_O_FEATURES))
    o_afford = np.zeros((num_objects, num_afford))
    for i in range(num_objects):
        feature_list = t.next().split()
        afford, object_id = map(int, feature_list[0:2])
        o_afford[object_id-1][afford-1] += 1
        for j in range(2, len(feature_list)):
            feature_id, feature_value = map(int, feature_list[j].split(':'))
            o_nodes[object_id-1][feature_id-1] = feature_value
    
    # h_nodes[feature_id]
    # <sub-activity_class> <skel_id> <feature_num>:<feature_value> ...
    h_nodes = np.zeros(NUM_H_FEATURES)
    h_act_class = np.zeros(num_act_classes)
    feature_list = t.next().split()
    act_class, skel_id = map(int, feature_list[0:2])
    assert skel_id == 1
    h_act_class[act_class-1] += 1
    for j in range(2, len(feature_list)):
        feature_id, feature_value = map(int, feature_list[j].split(':'))
        h_nodes[feature_id-1] = feature_value

    # oo_edges[object_id_1][object_id_2][feature_id]
    # <affordance_class_1> <affordance_class_2> <object_id_1> <object_id_2> <feature_num>:<feature_value> ...
    oo_edges = np.zeros((num_objects, num_objects, NUM_OO_FEATURES))
    for i in range(num_oo):
        feature_list = t.next().split()
        afford_1, afford_2, object_id_1, object_id_2 = map(int, feature_list[0:4])
        assert o_afford[object_id_1-1][afford_1-1] == 1
        assert o_afford[object_id_2-1][afford_2-1] == 1
        for j in range(4, len(feature_list)):
            feature_id, feature_value = map(int, feature_list[j].split(':'))
            oo_edges[object_id_1-1][object_id_2-1][feature_id-1] = feature_value

    # ho_edges[object_id][feature_id]
    # <affordance_class> <sub-activity_class> <object_id> <feature_num>:<feature_value> ...
    ho_edges = np.zeros((num_objects, NUM_HO_FEATURES))
    for i in range(num_ho):
        feature_list = t.next().split()
        afford, act_class, object_id = map(int, feature_list[0:3])
        assert o_afford[object_id-1][afford-1] == 1
        assert h_act_class[act_class-1] == 1
        for j in range(3, len(feature_list)):
            feature_id, feature_value = map(int, feature_list[j].split(':'))
            ho_edges[object_id-1][feature_id-1] = feature_value

    return o_nodes, h_nodes, oo_edges, ho_edges, o_afford, h_act_class

def parse_between_segment_file(lines, h_outputs, o_outputs):
    t = iter(lines)
    num_ot, num_ht, sn1, sn2 = map(int, t.next().split())
    assert num_ot == num_objects
    assert num_ht == 1

    # ot_edges[object_id][feature_id]
    # <affordance_class_sn1> <affordance_class_sn2> <object_id> <feature_num>:<feature_value> ...
    ot_edges = np.zeros((num_objects, NUM_OT_FEATURES))
    for i in range(num_ot):
        feature_list = t.next().split()
        afford_1, afford_2, object_id = map(int, feature_list[0:3])
        assert o_outputs[object_id-1][sn1-1][afford_1-1] == 1
        assert o_outputs[object_id-1][sn2-1][afford_2-1] == 1
        for j in range(3, len(feature_list)):
            feature_id, feature_value = map(int, feature_list[j].split(':'))
            ot_edges[object_id-1][feature_id-1] = feature_value

    # ht_edges[feature_id]
    # <sub-activity_class_sn1> <sub-activity_class_sn2> <skel_id> <feature_num>:<feature_value> ...
    ht_edges = np.zeros(NUM_HT_FEATURES)
    feature_list = t.next().split()
    act_class_1, act_class_2, skel_id = map(int, feature_list[0:3])
    assert h_outputs[0][sn1-1][act_class_1-1] == 1
    assert h_outputs[0][sn2-1][act_class_2-1] == 1
    assert skel_id == 1
    for j in range(3, len(feature_list)):
        feature_id, feature_value = map(int, feature_list[j].split(':'))
        ht_edges[feature_id-1] = feature_value

    return ot_edges, ht_edges

def get_metadata(activity):
    with open(os.path.join(DATA_ROOT, activity + "_1.txt")) as f:
        metadata = map(int, f.readline().rstrip('\n').split())
    return metadata[0], metadata[3], metadata[4]

# ht_inputs, ot_inputs, ho_inputs, oo_inputs, h_inputs, o_inputs, h_outputs, o_outputs
# Given an activity ID, outputs a list of tensors
def get_activity_tensors(activity, num_segments):
    global num_objects, num_afford, num_act_classes
    num_objects, num_afford, num_act_classes = get_metadata(activity)

    # Initialize the tensors
    h_inputs = np.zeros((NUM_HUMANS, num_segments, NUM_H_FEATURES))
    ht_inputs = np.zeros((NUM_HUMANS, num_segments, NUM_HT_FEATURES))
    ho_inputs = np.zeros((NUM_HUMANS, num_segments, NUM_HO_FEATURES))
    h_outputs = np.zeros((NUM_HUMANS, num_segments, num_act_classes))
    o_inputs = np.zeros((num_objects, num_segments, NUM_O_FEATURES))
    ot_inputs = np.zeros((num_objects, num_segments, NUM_OT_FEATURES))
    oo_inputs = np.zeros((num_objects, num_segments, 2 * NUM_OO_FEATURES))
    oh_inputs = np.zeros((num_objects, num_segments, NUM_HO_FEATURES))
    o_outputs = np.zeros((num_objects, num_segments, num_afford))

    # Deal with segment files (nodes and spatial edges)
    for i in range(num_segments):
        filename = activity + "_" + str(i+1) + ".txt"
        lines = [line.rstrip('\n') for line in open(os.path.join(DATA_ROOT, filename))]
        # o_nodes[object_id][feature_id], h_nodes[feature_id], oo_edges[object_id_1][object_id_2][feature_id], ho_edges[object_id][feature_id]
        # o_afford[object_id][afford], h_act_class[act_class]
        o_nodes, h_nodes, oo_edges, ho_edges, o_afford, h_act_class = parse_segment_file(lines)
        for object_id in range(num_objects):
            o_inputs[object_id][i] = o_nodes[object_id]
        h_inputs[0][i] = h_nodes
        for object_id in range(num_objects):
            for object_id_2 in range(num_objects):
                # TODO: might change how this is concatenated in the future
                oo_inputs[object_id][i][:NUM_OO_FEATURES] += oo_edges[object_id][object_id_2]
                oo_inputs[object_id][i][NUM_OO_FEATURES:] += oo_edges[object_id_2][object_id]
        for object_id in range(num_objects):
            ho_inputs[0][i] += ho_edges[object_id]
            oh_inputs[object_id][i] = ho_edges[object_id]
            o_outputs[object_id][i] = o_afford[object_id]
        h_outputs[0][i] = h_act_class

    # Deal with between segment files (temporal edges)
    for i in range(num_segments-1):
        filename = activity + "_" + str(i+1) + "_" + str(i+2) + ".txt"
        lines = [line.rstrip('\n') for line in open(os.path.join(DATA_ROOT, filename))]
        # ot_edges[object_id][feature_id], ht_edges[feature_id]
        ot_edges, ht_edges = parse_between_segment_file(lines, h_outputs, o_outputs)
        for object_id in range(num_objects):
            ot_inputs[object_id][i] = ot_edges[object_id]
        ht_inputs[0][i] = ht_edges

    filename = activity + ".npz"
    with open(os.path.join(NUMPY_ROOT, filename), 'w') as f:
        np.savez(f, h_inputs=h_inputs, ht_inputs=ht_inputs, ho_inputs=ho_inputs, h_outputs=h_outputs, o_inputs=o_inputs,
            ot_inputs=ot_inputs, oo_inputs=oo_inputs, oh_inputs=oh_inputs, o_outputs=o_outputs)
        f.close()

def load_numpy_array(activity):
    with open(os.path.join(NUMPY_ROOT, activity) + ".npz") as f:
        return np.load(f)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    activity_to_num_segments = get_activity_to_num_segments()
    for activity, num_segments in activity_to_num_segments.iteritems():
        get_activity_tensors(activity, num_segments)
