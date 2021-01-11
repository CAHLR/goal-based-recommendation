__author__ = 'jwj'
import torch
from torch.autograd import Variable
import myLSTM as LSTM
import pickle
import numpy as np
import pandas as pd
import torch.utils.data as Data
from data_process import get_data_from_condense_seq, process_data
from evaluate import evaluate_loss
from torch.nn.utils import clip_grad_norm
from utils import *
torch.manual_seed(1)    # reproducible


def train(model, optimizer, loader, train_data, epoch):

    model.train()
    summ = []
    for step, (batch_x, batch_y) in enumerate(loader):  # batch_x: index of batch data
        print('Epoch: ', epoch, ' | Iteration: ', step+1)
        processed_data = process_data(batch_x.numpy(), train_data, batchsize, dim_input_course, dim_input_grade, dim_input_major)
        padded_input = Variable(torch.Tensor(processed_data[0]), requires_grad=False).cuda()
        seq_len = processed_data[1]
        padded_label = Variable(torch.Tensor(processed_data[2]), requires_grad=False).cuda()

        # clear gradients and hidden state
        optimizer.zero_grad()
        model.hidden = model.init_hidden()
        model.hidden[0] = model.hidden[0].cuda()
        model.hidden[1] = model.hidden[1].cuda()
        y_pred = model(padded_input, seq_len).cuda()
        loss = model.loss(y_pred, padded_label).cuda()
        print('Epoch ' + str(epoch) + ': ' + 'The '+str(step+1)+'-th interation: loss'+str(loss.item())+'\n')
        loss.backward()
        if clip_gradient > 0:
            clip_grad_norm(model.parameters(), clip_gradient)
        optimizer.step()
        summ.append(loss.item())

    average_loss = np.mean(summ)
    return average_loss


def train_and_evaluate(model, train_data, vali_data, optimizer, tolerance=5):

    best_vali_loss = None  # set a large number for validation loss at first
    best_vali_accu = 0
    epoch = 0
    training_loss_epoch = []
    testing_loss_epoch = []

    # training data on mini batch
    train_data_index = torch.IntTensor(range(train_data.shape[0]))
    torch_train_data_index = Data.TensorDataset(train_data_index, train_data_index)
    train_loader = Data.DataLoader(dataset=torch_train_data_index, batch_size=batchsize, shuffle=True, num_workers=2, drop_last=True)

    # validation data on mini batch
    vali_data_index = torch.IntTensor(range(vali_data.shape[0]))
    torch_vali_data_index = Data.TensorDataset(vali_data_index, vali_data_index)
    vali_loader = Data.DataLoader(dataset=torch_vali_data_index, batch_size=batchsize, shuffle=True, num_workers=2, drop_last=True)

    # apply early stopping
    while True:
        epoch += 1
        train_loss = train(model, optimizer, train_loader, train_data, epoch)
        training_loss_epoch.append(train_loss)
        print('The average loss of training set for the first ' + str(epoch) + ' epochs: ' + str(training_loss_epoch))

        # evaluate on validation set
        evaluation_loss = evaluate_loss(model, vali_loader, vali_data, batchsize, dim_input_course, dim_input_grade, dim_input_major)
        testing_loss_epoch.append(evaluation_loss)
        print('The average loss of validation set for the first ' + str(epoch) + ' epochs: ' + str(testing_loss_epoch))

        # save best model so far
        if best_vali_loss is None:
            best_vali_loss = evaluation_loss
            torch.save(model, args.model_path+model_name)
        elif evaluation_loss < best_vali_loss:
            best_vali_loss = evaluation_loss
            torch.save(model, args.model_path+model_name)

        if epoch >= 5:
            # early stopping with tolerance
            near_loss = testing_loss_epoch[-tolerance:]
            if near_loss == sorted(near_loss):  # loss increases for 5 consecutive epochs
                print("Best model found! Stop training, saving loss!")
                loss_train_vali = {'epoch': range(1, epoch+1), 'training loss': training_loss_epoch, 'testing loss': testing_loss_epoch}
                loss_train_vali = pd.DataFrame.from_dict(loss_train_vali)
                loss_train_vali.to_csv(args.model_path+pack_loss, index=False)
                break


if __name__ == '__main__':
    # load hyper parameters
    args = parse_args()
    print(args)

    model_type = args.model_type  # RNN, GRU
    drop_out = args.drop_out  # If no dropout, set to 0
    weight_decay = args.weight_decay  # If no weight_decay, set to 0
    clip_gradient = args.clip_gradient  # If no clip_gradient, set to 0
    nb_lstm_units = args.nb_lstm_units
    nb_lstm_layers = args.nb_lstm_layers
    batchsize = args.batch_size
    learning_rate = args.learning_rate

    # model name
    model_name = model_type + str(nb_lstm_layers) + '_' + str(nb_lstm_units) + 'lr' + str(learning_rate) + 'drp' + str(drop_out) + 'wd' + str(weight_decay) + 'clp' + str(clip_gradient) + '.pkl'
    pack_loss = model_type + str(nb_lstm_layers) + '_' + str(nb_lstm_units) + 'lr' + str(learning_rate) + 'drp' + str(drop_out) + 'wd' + str(weight_decay) + 'clp' + str(clip_gradient) + '.csv'

    evaluated_semester = args.evaluated_semester

    # load data
    print("Loading data...")
    subtrain_data, vali_data = get_data_from_condense_seq(evaluated_semester)

    f = open(args.course_id_path, 'rb')
    course_id = pickle.load(f)['course_id']
    f = open(args.major_id_path, 'rb')
    major_id = pickle.load(f)['major_id']
    f.close()
    # input is the concat of grade and course in the next semester
    dim_input_course = len(course_id)
    dim_input_grade = len(course_id) * 4
    dim_input_major = len(major_id)

    #  training
    print("Training Begin")
    model = LSTM.GradePrediction(model_type, dim_input_course, dim_input_grade, dim_input_major, nb_lstm_layers, nb_lstm_units, batchsize, drop_out).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_and_evaluate(model, subtrain_data, vali_data, optimizer)







