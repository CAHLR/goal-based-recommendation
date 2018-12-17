__author__ = 'jwj'
import torch
from torch.autograd import Variable
import myLSTM as LSTM
import pickle
import numpy as np
import torch.utils.data as Data
from data_process import get_data_from_condense_seq, process_data
from evaluate_cat_cat import evaluate_loss, evaluate_accuracy
from torch.nn.utils import clip_grad_norm
torch.manual_seed(1)    # reproducible


def train(model, optimizer, loader, train_data, epoch, weight1, weight2):

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
        loss = model.loss(y_pred, padded_label, weight1, weight2).cuda()
        print('Epoch ' + str(epoch) + ': ' + 'The '+str(step+1)+'-th interation: loss'+str(loss.data[0])+'\n')
        loss.backward()
        if clip_gradient > 0:
            clip_grad_norm(model.parameters(), clip_gradient)
        optimizer.step()
        summ.append(loss.data[0])

    average_loss = np.mean(summ)
    return average_loss


def train_and_evaluate(model, train_data, vali_data, optimizer, weight1=torch.Tensor([1,1]).cuda(), weight2=torch.Tensor([1,1]).cuda(), weight3=torch.Tensor([1,1]).cuda(), weight4=torch.Tensor([1,1]).cuda(), tolerance=5):

    best_vali_loss = 100  # set a large number for validation loss at first
    best_vali_accu = 0
    epoch = 0
    training_loss_epoch = []
    testing_loss_epoch = []

    # training data on mini batch
    train_data_index = torch.IntTensor(range(train_data.shape[0]))
    torch_train_data_index = Data.TensorDataset(data_tensor=train_data_index, target_tensor=train_data_index)
    train_loader = Data.DataLoader(dataset=torch_train_data_index, batch_size=batchsize, shuffle=True, num_workers=2, drop_last=True)

    # validation data on mini batch
    vali_data_index = torch.IntTensor(range(vali_data.shape[0]))
    torch_vali_data_index = Data.TensorDataset(data_tensor=vali_data_index, target_tensor=vali_data_index)
    vali_loader = Data.DataLoader(dataset=torch_vali_data_index, batch_size=batchsize, shuffle=True, num_workers=2, drop_last=True)

    # apply early stopping
    while True:
        epoch += 1
        train_loss = train(model, optimizer, train_loader, train_data, epoch, weight1, weight2)
        training_loss_epoch.append(train_loss)
        print('The average loss of training set for the first ' + str(epoch) + ' epochs: ' + str(training_loss_epoch))

        # validate by loss or accuracy
        if validation_metric == 'loss':
            # evaluate on validation set
            evaluation_loss = evaluate_loss(model, vali_loader, vali_data, batchsize, dim_input_course, dim_input_grade, dim_input_major, weight3, weight4)
            testing_loss_epoch.append(evaluation_loss)
            print('The average loss of validation set for the first ' + str(epoch) + ' epochs: ' + str(testing_loss_epoch))

            # save best model so far
            if evaluation_loss < best_vali_loss:
                best_vali_loss = evaluation_loss
                torch.save(model, model_name)

            if epoch >= 5:
                # early stopping with tolerance
                near_loss = testing_loss_epoch[-tolerance:]
                if near_loss == sorted(near_loss):  # loss increases for 5 consecutive epochs
                    print("Best model found! Stop training, saving loss!")
                    loss_train_vali = {'training loss': training_loss_epoch, 'testing loss': testing_loss_epoch}
                    f = open(pack_loss, 'wb')
                    pickle.dump(loss_train_vali, f)
                    f.close()
                    break
        elif validation_metric == 'accuracy':
            # evaluate on validation set
            evaluation_accu = evaluate_accuracy(model, vali_loader, vali_data, batchsize, dim_input_course, dim_input_grade)
            testing_loss_epoch.append(evaluation_accu)
            print('The average accuracy of validation set for the first ' + str(epoch) + ' epochs: ' + str(testing_loss_epoch))

            # save best model so far
            if evaluation_accu > best_vali_accu:
                best_vali_accu = evaluation_accu
                torch.save(model, model_name)

            if epoch >= 5:
                # early stopping with tolerance
                near_accu = testing_loss_epoch[-tolerance:]
                if near_accu == sorted(near_accu, reverse=True):  # accuracy decreases for 5 consecutive epochs
                    print("Best model found! Stop training, saving loss and accuracy!")
                    loss_train_vali = {'training loss': training_loss_epoch, 'testing accuracy': testing_loss_epoch}
                    f = open(pack_loss, 'wb')
                    pickle.dump(loss_train_vali, f)
                    f.close()
                    break


if __name__ == '__main__':
    # only consider grade higher than B or not, pass or not pass

    # set hyper parameters
    validation_metric = 'loss'  # accuracy or loss
    weighted_loss = False  # whether use weighted loss for training
    model_type = 'LSTM'  # RNN, GRU
    drop_out = 0.2  # If no dropout, set to 0
    weight_decay = 1e-07  # If no weight_decay, set to 0
    clip_gradient = 0  # If no clip_gradient, set to 0
    nb_lstm_units = 50
    nb_lstm_layers = 1

    batchsize = 32
    learning_rate = 0.001
    hidden_out_activation = "sigmoid"  # SELU, RELU

    # model name
    model_name = 'nw_' + model_type + '_cat_cat_' + str(nb_lstm_layers) + '_' + str(nb_lstm_units) + 'lr' + str(learning_rate) + 'drp' + str(drop_out) + 'wd' + str(weight_decay) + 'clp' + str(clip_gradient) + '.pkl'
    pack_loss = 'nw_Loss_' + model_type + '_cat_cat_' + str(nb_lstm_layers) + '_' + str(nb_lstm_units) + 'lr' + str(learning_rate) + 'drp' + str(drop_out) + 'wd' + str(weight_decay) + 'clp' + str(clip_gradient) + '.pkl'

    # subtrain or train
    train_or_subtrain = 'subtrain'

    # subtrain or train
    if train_or_subtrain == 'subtrain':
        time = 22
    else:
        time = 25

    # load data
    print("Load data")
    subtrain_data, vali_data = get_data_from_condense_seq(time)

    f = open('../../RNN/data_preprocess/course_id.pkl', 'rb')
    course_id = pickle.load(f)['course_id']
    f = open('../../RNN2/data_preprocess/major_id.pkl', 'rb')
    major_id = pickle.load(f)['major_id']
    f.close()
    # input is the concat of grade and course in the next semester
    dim_input_course = len(course_id)
    dim_input_grade = len(course_id) * 4
    dim_input_major = len(major_id)

    #  training
    print("Training Begin")
    model = LSTM.LSTM_cat_cat(model_type, dim_input_course, dim_input_grade, dim_input_major, nb_lstm_layers, nb_lstm_units, batchsize, drop_out).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if weighted_loss:
        #count_sub = count_vali = torch.Tensor(np.array([0.8, 0.2, 0.6, 0.4])).cuda()
        # count the proportion of each label in subtraining set
        count_sub = np.zeros(4)
        for i in range(subtrain_data.shape[0]):
            for j in range(subtrain_data.shape[1]):
                if subtrain_data[i, j] != {}:
                    for k in subtrain_data[i, j]:
                        if k[1] <= 2:
                            count_sub[0] += 1
                        elif k[1] <= 5:
                            count_sub[1] += 1
                        elif k[1] == 6:
                            count_sub[2] += 1
                        elif k[1] == 7:
                            count_sub[3] += 1
        count_sub[:2] /= np.sum(count_sub[:2])
        count_sub[2:] /= np.sum(count_sub[2:])
        print(count_sub)
        count_sub = torch.Tensor(count_sub).cuda()

        # count the proportion of each label in validation set
        count_vali = np.zeros(4)
        for i in range(vali_data.shape[0]):
            for j in range(vali_data.shape[1]):
                if vali_data[i, j] != []:
                    for k in vali_data[i, j]:
                        if k[1] <= 2:
                            count_vali[0] += 1
                        elif k[1] <= 5:
                            count_vali[1] += 1
                        elif k[1] == 6:
                            count_vali[2] += 1
                        elif k[1] == 7:
                            count_vali[3] += 1
        count_vali[:2] /= np.sum(count_vali[:2])
        count_vali[2:] /= np.sum(count_vali[2:])
        print(count_vali)
        count_vali = torch.Tensor(count_vali).cuda()

        train_and_evaluate(model, subtrain_data, vali_data, optimizer, count_sub[:2], count_sub[2:], count_vali[:2], count_vali[2:])
    else:
        train_and_evaluate(model, subtrain_data, vali_data, optimizer)







