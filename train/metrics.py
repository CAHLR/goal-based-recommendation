__author__ = 'jwj'
import numpy as np
import torch


def accuracy(output, out_len, label):
    # first get the last semester of each sequence
    output_last = output[range(output.shape[0]), out_len - 1]
    label_last = label[range(label.shape[0]), out_len - 1]

    # second mask the unenrolled courses in output vector
    tran_label = label_last.contiguous().view(-1, 4)
    tran_label1 = torch.sum(tran_label, dim=1)
    masked_label = torch.nonzero(tran_label1).view(-1)
    real_label = tran_label[masked_label]

    tran_output = output_last.contiguous().view(-1, 4)
    real_output = tran_output[masked_label]

    # identify whether >=B<B or Credit/No credit
    real_label1 = real_label[:, :2]
    real_output1 = real_output[:, :2]
    l1 = torch.sum(real_label1, dim=1)
    l1 = torch.nonzero(l1).view(-1)

    real_output1 = real_output1[l1]
    real_output1 = real_output1.max(dim=1)[1]
    real_label1 = torch.nonzero(real_label1)[:, 1]
    #print(real_label1.size(), real_output1.size())
    hit1 = len(l1) - len(torch.nonzero(real_label1 - real_output1))
    accuracy1 = hit1 / float(len(l1))

    real_label2 = real_label[:, 2:]
    real_output2 = real_output[:, 2:]
    l2 = torch.sum(real_label2, dim=1)
    l2 = torch.nonzero(l2).view(-1)

    real_output2 = real_output2[l2]
    real_output2 = real_output2.max(dim=1)[1]
    real_label2 = torch.nonzero(real_label2)[:, 1]
    hit2 = len(l2) - len(torch.nonzero(real_label2 - real_output2))
    accuracy2 = hit2 / float(len(l2))
    accuracy = (hit1 + hit2) / float(len(l1) + len(l2))
    return accuracy1, accuracy2, accuracy, len(l1), len(l2)


# true-positive, false-positive, true-negative, false-negative rate of each grade
def sensitivity(output, out_len, label):
    # first get the last semester of each sequence
    output_last = output[range(output.shape[0]), out_len - 1]
    label_last = label[range(label.shape[0]), out_len - 1]

    # second mask the unenrolled courses in output vector
    tran_label = label_last.contiguous().view(-1, 4)
    tran_label1 = torch.sum(tran_label, dim=1)
    masked_label = torch.nonzero(tran_label1).view(-1)
    real_label = tran_label[masked_label]

    tran_output = output_last.contiguous().view(-1, 4)
    real_output = tran_output[masked_label]

    # identify whether >=B or Credit/No credit
    real_label1 = real_label[:, :2]
    real_output1 = real_output[:, :2]
    l1 = torch.sum(real_label1, dim=1)
    l1 = torch.nonzero(l1).view(-1)

    real_output1 = real_output1[l1]
    real_output1 = real_output1.max(dim=1)[1].data.cpu().numpy()
    real_label1 = torch.nonzero(real_label1)[:, 1].data.cpu().numpy()
    tp = np.zeros(2)
    tn = np.zeros(2)
    true = np.zeros(2)
    false = np.zeros(2)
    predict_true = np.zeros(2)
    predict_false = np.zeros(2)

    # >=B or not
    hit1 = np.where(real_label1 - real_output1 == 0)[0]
    tp[0] = len(np.where(real_label1[hit1] == 0)[0])
    tn[0] = len(np.where(real_label1[hit1] == 1)[0])
    true[0] = len(np.where(real_label1 == 0)[0])
    false[0] = len(np.where(real_label1 == 1)[0])
    predict_true[0] = len(np.where(real_output1 == 0)[0])
    predict_false[0] = len(np.where(real_output1 == 1)[0])

    # credit / uncredit
    real_label2 = real_label[:, 2:]
    real_output2 = real_output[:, 2:]
    l2 = torch.sum(real_label2, dim=1)
    l2 = torch.nonzero(l2).view(-1)

    real_output2 = real_output2[l2]
    real_output2 = real_output2.max(dim=1)[1].data.cpu().numpy()
    real_label2 = torch.nonzero(real_label2)[:, 1].data.cpu().numpy()
    hit2 = np.where(real_label2 - real_output2 == 0)[0]
    tp[1] = len(np.where(real_label2[hit2] == 0)[0])
    tn[1] = len(np.where(real_label2[hit2] == 1)[0])
    true[1] = len(np.where(real_label2 == 0)[0])
    false[1] = len(np.where(real_label2 == 1)[0])
    predict_true[1] = len(np.where(real_output2 == 0)[0])
    predict_false[1] = len(np.where(real_output2 == 1)[0])
    print(tp, tn, true, false, predict_true, predict_false)
    return tp, tn, true, false, predict_true, predict_false






