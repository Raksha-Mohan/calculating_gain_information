import numpy as np
def ent(class_cnt):
    total = sum(class_cnt)
    if total == 0:
        return 0
    probs = [count / total for count in class_cnt]
    return -sum(p * np.log2(p) for p in probs if p > 0)


def information_gain(parent_counts, splits):
    # Entropy of the parent node
    parent_ent = ent(parent_counts)

    # Entropy of the child nodes
    total_instances = sum(parent_counts)
    weighted_child_ent = 0

    for split in splits:
        split_ent = ent(split)
        weighted_child_ent += (sum(split) / total_instances) * split_ent

    # Information gain is the difference
    return parent_ent - weighted_child_ent


# Given cases
# (1) C0: 1 and C1: 9 + C0: 7 and C1: 3
case_1_parent = [8, 12]
case_1_splits = [[1, 9], [7, 3]]

# (2) C0: 3 and C1: 3 + C0: 3 and C1: 11
case_2_parent = [6, 14]
case_2_splits = [[3, 3], [3, 11]]

# Calculate information gain for both cases
gn_case_1 = information_gain(case_1_parent, case_1_splits)
gn_case_2 = information_gain(case_2_parent, case_2_splits)

print(f"Information Gain for Case (1): {gn_case_1:.3f}")
print(f"Information Gain for Case (2): {gn_case_2:.3f}")

