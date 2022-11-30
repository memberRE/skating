import numpy as np
import csv


test_len = 628
label_len = 30
sub = 'submission.csv'


def weight(results, weight):
    assert len(results) == len(weight)
    w = [[0 for j in range(label_len)] for i in range(test_len)]
    for l in range(len(results)):
        res = np.load(results[l], allow_pickle=True)
        for i in range(len(res)):
            for j in range(len(res[i])):
                w[i][j] += weight[l] * res[i][j]

    r = [0 for i in range(test_len)]
    for i in range(len(w)):
        label = 0
        max_w = 0
        for j in range(len(w[i])):
            if w[i][j] > max_w:
                max_w = w[i][j]
                label = j
        r[i] = label

    with open(sub, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'result'])
        for i in range(len(r)):
            writer.writerow([i, r[i]])


if __name__ == '__main__':
    weight(
        ['result.pkl'],   # 预测结果文件
        [1]               # 权重
    )
