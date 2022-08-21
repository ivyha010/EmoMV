import subprocess
import numpy as np
import argparse
import os
import time
import gc
from collections import Mapping, Container
from sys import getsizeof
import h5py
from torch.utils.data import DataLoader, Dataset
from pytorchtools import EarlyStopping
from sklearn import metrics
from DS1_model_classification_single_task import *
from sklearn.preprocessing import StandardScaler
import csv
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, roc_auc_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import random

def deep_getsizeof(o, ids):
    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str) or isinstance(0, np.unicode):
        return r

    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())

    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)

    return r

# Memory check
def memoryCheck():
    ps = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv'],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    print(ps.communicate(), '\n')
    os.system("free -m")

# Free memory
def freeCacheMemory():
    torch.cuda.empty_cache()
    gc.collect()


# Build dataloaders
def myDataloader(videoFeatures, audioFeatures, emoValues, args, shuffleBool=False):
    class my_dataset(Dataset):
        def __init__(self, videoData, audioData, emo):
            self.videoData = videoData
            self.audioData = audioData
            self.emo = emo

        def __getitem__(self, index):
            return self.videoData[index], self.audioData[index], self.emo[index]

        def __len__(self):
            return len(self.videoData)

    # Build dataloaders
    my_dataloader = DataLoader(dataset=my_dataset(videoFeatures, audioFeatures, emoValues), batch_size=args.batch_size, shuffle=shuffleBool)
    return my_dataloader


# Train
def train_func(train_loader, validate_loader, the_model, optimizer, criter, device, n_epochs, patience):
    start_time = time.time()
    # to track the training loss as the model trains
    train_losses = []
    valid_losses = []
    # to track the validation loss as the model trains
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1, n_epochs + 1):
        #epoch_acc = 0
        # Adjust learning rate
        # adjust_learning_rate(optimizer, epoch)
        #####################
        ## train the model ##
        #####################
        the_model.train()  # prep model for training
        count_batches = 0
        for (video_feature, audio_feature, labels) in train_loader:
            video_feature, audio_feature, labels = video_feature.to(device), audio_feature.to(device), labels.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            out = the_model.forward(video_feature, audio_feature)
            loss =  criter(out, labels)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward(retain_graph=True)
            # perform a single optimization step (parameter update)
            optimizer.step()
            #epoch_acc += acc.item()
            # record training loss
            train_losses.append(loss.item())

            if (count_batches % 100) == 0:
                print('Batch: ', count_batches)
            count_batches += 1

            # Free catch memory
            del video_feature, audio_feature, labels
            freeCacheMemory()

        ######################
        # validate the model #
        ######################
        the_model.eval()  # prep model for evaluation
        #val_epoch_acc = 0
        for (v_video_feature, v_audio_feature, vLabels) in validate_loader:
            v_video_feature, v_audio_feature, vLabels = v_video_feature.to(device), v_audio_feature.to(device), vLabels.to(device)
            vout = the_model(v_video_feature, v_audio_feature)
            # validation loss:
            batch_valid_losses = criter.forward(vout, vLabels)
            valid_losses.append(batch_valid_losses.item())

            del vout, v_video_feature, v_audio_feature, vLabels
            freeCacheMemory()

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        avg_train_losses.append(train_loss)
        valid_loss = np.average(valid_losses)
        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}]' +
                     f' train_loss: {train_loss:.8f} ' +
                     f' valid_loss: {valid_loss:.8f} ')
        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the (1-valid_pearson) to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss.item(), the_model)
        print('Epoch[{}/{}]: Training time: {} seconds '.format(epoch, n_epochs, time.time() - start_time))

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    the_model.load_state_dict(torch.load('checkpoint.pt'))

    return the_model, avg_train_losses, avg_valid_losses


# VALIDATE
# LOAD TEST DATA TO GPU IN BATCHES
def validate_func(validate_loader, the_model, device):
    the_model.eval()
    all_vout_labels = []
    all_vlabels = []
    all_y_prob = []
    for (v_video_feature,  v_audio_feature, v_labels) in validate_loader:
        v_video_feature, v_audio_feature, v_labels = v_video_feature.to(device), v_audio_feature.to(device), v_labels.to(device)
        vout = the_model(v_video_feature, v_audio_feature)
        _, vout_labels = torch.max(vout, 1)
        y_prob = torch.nn.functional.softmax(vout)
        all_y_prob.append(y_prob)
        all_vout_labels.append(vout_labels.detach().cpu().detach().numpy())
        all_vlabels.append(v_labels.detach().cpu().detach().numpy())

        del v_video_feature, v_audio_feature
        freeCacheMemory()

    all_vout_labels = np.concatenate(all_vout_labels, axis=0)
    all_vlabels = np.concatenate(all_vlabels, axis=0)
    all_y_prob = torch.cat(all_y_prob, dim=0)

    # Classification accuracy
    val_acc = np.sum(all_vout_labels == all_vlabels)
    # Compute the average accuracy
    val_acc = np.float32(val_acc/len(all_vlabels))
    print('Accuracy: ', val_acc)
    return all_vout_labels, all_vlabels, all_y_prob


# Load extracted featurefiles
def loadingfiles(feature_file, label_file):
    print('\n')
    print('Loading h5 files containing extracted features......')
    h5file = h5py.File(feature_file, mode='r')
    getKey = list(h5file.keys())[0]
    getData = h5file.get(getKey)
    features = np.asarray(getData)
    features = torch.from_numpy(features)
    h5file.close()

    labelValues = []
    with open(label_file,'r') as csvfile:
        csvReader = csv.reader(csvfile)
        for row in csvReader:
            labelValues.append(np.int((row[2])))
    labelValues = np.asarray(labelValues)
    labelValues = torch.from_numpy(labelValues)
    csvfile.close()

    return features, labelValues


def plot_cfs_matrix(cfs):
    df_cm = pd.DataFrame(cfs, lbl_range, lbl_range)
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, cmap="Reds", annot=True, fmt = ".5g", annot_kws={"size": 13})
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()

def comput_roc(y_prob, y_truth):
    # from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    # https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/
    y_truth = torch.from_numpy(y_truth) # y_true is a tensor
    y_prob = y_prob.cpu().detach().numpy()
    y_true_one_hot = np.array([[1,0] if l==1 else [0,1] for l in y_truth]) # for 2 classes
    n_classes = y_true_one_hot.shape[1]

     # ROC curve for the model
    fpr, tpr, _ = roc_curve(y_truth, y_prob[:, 1], pos_label=1)
    # roc curve for tpr = fpr
    random_probs = [0 for i in range(len(y_truth))]
    p_fpr, p_tpr, _ = roc_curve(y_truth, random_probs, pos_label=1)
    plt.plot(fpr, tpr, linestyle='--', color='orange', label='single task learning classif model')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    plt.show()

    # auc score
    auc_score = roc_auc_score(y_truth, y_prob[:, 1])
    print("ROC AUC score: ", auc_score)


# Main
def main(args):
    # Device configuration
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # Manual seed
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.random.initial_seed()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device: ', device)

    # Data
    # for model training
    video_train_features, train_labelValues = loadingfiles(video_feature_file_train, label_file_train)
    audio_train_features,_ = loadingfiles(audio_feature_file_train, label_file_train)

    memoryCheck()

    video_val_features, val_labelValues  = loadingfiles(video_feature_file_validate, label_file_validate)
    audio_val_features,_ = loadingfiles(audio_feature_file_validate, label_file_validate)

    video_test_features, test_labelValues = loadingfiles(video_feature_file_test, label_file_test)
    audio_test_features, _ = loadingfiles(audio_feature_file_test, label_file_test)

    # standardize:
    scaler_1 = StandardScaler()
    video_train_features = torch.from_numpy(scaler_1.fit_transform(video_train_features)).float()
    video_val_features = torch.from_numpy(scaler_1.transform(video_val_features)).float()
    video_test_features = torch.from_numpy(scaler_1.transform(video_test_features)).float()

    scaler_2 = StandardScaler()
    audio_train_features = torch.from_numpy(scaler_2.fit_transform(audio_train_features)).float()
    audio_val_features = torch.from_numpy(scaler_2.transform(audio_val_features)).float()
    audio_test_features = torch.from_numpy(scaler_2.transform(audio_test_features)).float()
    memoryCheck()

    train_dataset = myDataloader(video_train_features, audio_train_features, train_labelValues, args, True)
    validate_dataset = myDataloader(video_val_features, audio_val_features, val_labelValues, args, False)
    test_dataset = myDataloader(video_test_features, audio_test_features, test_labelValues, args, False)
    memoryCheck()

    # input_size for the model
    video_dim = video_train_features.shape[1]
    audio_dim = audio_train_features.shape[1]

    # Build the model
    model = embedding_network(video_dim, audio_dim).to(device)
    model = model.to(device)
    memoryCheck()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.wd)

    model, train_losses, valid_losses = train_func(train_dataset, validate_dataset, model, optimizer, criterion, device, args.num_epochs, args.patience)
    # Validation
    print("Validation set: ")
    val_voutput, val_grdtruth, val_pred_prob = validate_func(validate_dataset, model, device)
    # Test
    print("Testing set: ")
    test_voutput, test_grdtruth, test_pred_prob = validate_func(test_dataset, model, device)

    # Confusion matrix
    cfs_matrix = confusion_matrix(list(np.asarray(test_grdtruth)), list(np.asarray(test_voutput)))
    plot_cfs_matrix(cfs_matrix)
    # compute F-1 score and roc_auc_score
    f1 = f1_score(np.asarray(val_grdtruth), np.asarray(val_voutput))  # , average="macro")
    print("F1-score:  ", f1)
    # compute ROC
    comput_roc(test_pred_prob, test_grdtruth)

    # Save the model
    model_name = "DS1_classif_model_single_task.pth"
    torch.save(model.state_dict(), os.path.join(args.model_path, model_name))

    # Save the predicted values:
    afilename = "DS1_classif_model_predicted_labels.h5"
    h5file = h5py.File(os.path.join(pred_path, afilename), mode='w')
    h5file.create_dataset('data', data=np.array(test_voutput, dtype=np.int))
    h5file.close()

    os.remove('./checkpoint.pt')


if __name__ == "__main__":
    dir_path = "./DS1_EmoMV_A"   # path to extracted features
    model_path = os.path.join(dir_path, 'models')  # path to save models => rember to create this folder
    pred_path = os.path.join(dir_path, 'predicted_values')  # path to save predicted outputs => rember to create this folder
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=model_path, help='path for saving trained models')
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=20,
                        help='early stopping patience; how long to wait after last time validation loss improved')
    parser.add_argument('--batch_size', type=int, default=256, help='number of feature vectors loaded per batch')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', type=float, default=0.1, help='weight decay')
    #parser.add_argument('--mm', type=float, default=0.9, help='momentum')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 123)')

    args = parser.parse_args()
    print(args)

    # for training
    label_file_train = os.path.join(dir_path + "/" + "annotation", "DS1_TRAIN_MATCH_MISMATCH_labels.csv")
    video_feature_file_train = os.path.join(dir_path + "/" + "extracted_features", "SlowFast_DS1_TRAIN_MATCH_MISMATCH.h5")
    audio_feature_file_train = os.path.join(dir_path + "/" + "extracted_features", "VGGish_DS1_TRAIN_MATCH_MISMATCH.h5")

    # for validation
    label_file_validate = os.path.join(dir_path + "/" + "annotation", "DS1_VAL_MATCH_MISMATCH_labels.csv")
    video_feature_file_validate = os.path.join(dir_path + "/" + "extracted_features", "SlowFast_DS1_VAL_MATCH_MISMATCH.h5")
    audio_feature_file_validate = os.path.join(dir_path + "/" + "extracted_features", "VGGish_DS1_VAL_MATCH_MISMATCH.h5")

    # for testing
    label_file_test = os.path.join(dir_path + "/annotation", "DS1_TEST_MATCH_MISMATCH_labels.csv")
    video_feature_file_test = os.path.join(dir_path + "/" + "extracted_features", "SlowFast_DS1_TEST_MATCH_MISMATCH.h5")
    audio_feature_file_test = os.path.join(dir_path + "/" + "extracted_features", "VGGish_DS1_TEST_MATCH_MISMATCH.h5")

    # -------------------------------------------------------------------------------------------------------------------
    main_start_time = time.time()
    lbl_range = ["Mismatched", "Matched"]  # mismatched: 0, matched: 1
    main(args)
    print('Total running time: {:.5f} seconds'.format(time.time() - main_start_time))



