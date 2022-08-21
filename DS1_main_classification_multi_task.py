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
from DS1_model_classification_multi_task import *
from sklearn.preprocessing import StandardScaler
import csv
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, roc_auc_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

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
def myDataloader(videoFeatures, audioFeatures, labels, labels_vid, labels_aud, args, shuffleBool=False):
    class my_dataset(Dataset):
        def __init__(self, videoData, audioData, label, label_vid, label_aud):
            self.videoData = videoData
            self.audioData = audioData
            self.label = label
            self.label_vid = label_vid
            self.label_aud = label_aud

        def __getitem__(self, index):
            return self.videoData[index], self.audioData[index], self.label[index], self.label_vid[index], self.label_aud[index]

        def __len__(self):
            return len(self.videoData)

    # Build dataloaders
    my_dataloader = DataLoader(dataset=my_dataset(videoFeatures, audioFeatures,labels, labels_vid, labels_aud), batch_size=args.batch_size, shuffle=shuffleBool)
    return my_dataloader


# Train
def train_func(train_loader, validate_loader, the_model, optimizer, criter, criter_vid, criter_aud, device, n_epochs, patience):
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

    # Note: each batch is a batch of sequences(chunks)
    for epoch in range(1, n_epochs + 1):
        #epoch_acc = 0
        # Adjust learning rate
        # adjust_learning_rate(optimizer, epoch)
        #####################
        ## train the model ##
        #####################
        the_model.train()  # prep model for training
        count_batches = 0
        for (video_feature, audio_feature, labels, labels_vid, labels_aud) in train_loader:
            video_feature, audio_feature, labels, labels_vid, labels_aud = video_feature.to(device), audio_feature.to(device), labels.to(device), labels_vid.to(device), labels_aud.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            out, out_vid, out_aud = the_model.forward(video_feature, audio_feature)
            loss = criter(out, labels) + criter_vid(out_vid, labels_vid) + criter_aud(out_aud, labels_aud)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward(retain_graph=True)
            # perform a single optimization step (parameter update)
            optimizer.step()

            # record training loss
            train_losses.append(loss.item())

            if (count_batches % 100) == 0:
                print('Batch: ', count_batches)
            count_batches += 1

            # Free catch memory
            del video_feature, audio_feature, labels_vid, labels_aud
            freeCacheMemory()

        ######################
        # validate the model #
        ######################
        the_model.eval()  # prep model for evaluation
        for (v_video_feature, v_audio_feature, vLabels, vLabels_vid, vLabels_aud) in validate_loader:
            v_video_feature, v_audio_feature, vLabels, vLabels_vid, vLabels_aud = v_video_feature.to(device), v_audio_feature.to(device), vLabels.to(device), vLabels_vid.to(device), vLabels_aud.to(device)
            vout, vout_vid, vout_aud = the_model(v_video_feature, v_audio_feature)

            # validation loss:
            batch_valid_losses = criter.forward(vout, vLabels) + criter_vid.forward(vout_vid, vLabels_vid) + criter_aud.forward(vout_aud, vLabels_aud)
            valid_losses.append(batch_valid_losses.item())

            del v_video_feature, v_audio_feature, vLabels
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
        start_time = time.time()

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    the_model.load_state_dict(torch.load('checkpoint.pt'))

    return the_model, avg_train_losses, avg_valid_losses


# VALIDATE
def validate_func(validate_loader, the_model, device):
    the_model.eval()

    all_vout_labels = []
    all_vlabels = []
    all_y_prob = []

    all_vout_labels_vid = []
    all_vlabels_vid = []
    all_y_prob_vid = []

    all_vout_labels_aud = []
    all_vlabels_aud = []
    all_y_prob_aud = []

    for (v_video_feature,  v_audio_feature, v_labels, v_labels_vid, v_labels_aud) in validate_loader:
        v_video_feature, v_audio_feature, v_labels = v_video_feature.to(device), v_audio_feature.to(device), v_labels.to(device)

        vout, vout_vid, vout_aud = the_model(v_video_feature, v_audio_feature)

        _, vout_labels = torch.max(vout, 1)
        y_prob = torch.nn.functional.softmax(vout)
        all_y_prob.append(y_prob)

        _, vout_labels_vid = torch.max(vout_vid, 1)
        y_prob_vid = torch.nn.functional.softmax(vout_vid)
        all_y_prob_vid.append(y_prob_vid)

        _, vout_labels_aud = torch.max(vout_aud, 1)
        y_prob_aud = torch.nn.functional.softmax(vout_aud)
        all_y_prob_aud.append(y_prob_aud)

        all_vout_labels.append(vout_labels.detach().cpu().detach().numpy())
        all_vlabels.append(v_labels.detach().cpu().detach().numpy())

        all_vout_labels_vid.append(vout_labels_vid.detach().cpu().detach().numpy())
        all_vlabels_vid.append(v_labels_vid.detach().cpu().detach().numpy())

        all_vout_labels_aud.append(vout_labels_aud.detach().cpu().detach().numpy())
        all_vlabels_aud.append(v_labels_aud.detach().cpu().detach().numpy())

        del v_video_feature, v_audio_feature
        freeCacheMemory()

    all_vout_labels = np.concatenate(all_vout_labels, axis=0)
    all_vlabels = np.concatenate(all_vlabels, axis=0)
    all_y_prob = torch.cat(all_y_prob, dim=0)

    all_vout_labels_vid = np.concatenate(all_vout_labels_vid, axis=0)
    all_vlabels_vid = np.concatenate(all_vlabels_vid, axis=0)
    all_y_prob_vid = torch.cat(all_y_prob_vid, dim=0)

    all_vout_labels_aud = np.concatenate(all_vout_labels_aud, axis=0)
    all_vlabels_aud = np.concatenate(all_vlabels_aud, axis=0)
    all_y_prob_aud = torch.cat(all_y_prob_aud, dim=0)

    # Classification accuracy
    val_acc = np.sum(all_vout_labels == all_vlabels)
    # Compute the average accuracy and loss over all validate dataset
    val_acc = np.float32(val_acc/len(all_vlabels))

    val_acc_vid = np.sum(all_vout_labels_vid == all_vlabels_vid)
    # Compute the average accuracy and loss over all validate dataset
    val_acc_vid = np.float32(val_acc_vid/ len(all_vlabels_vid))

    val_acc_aud = np.sum(all_vout_labels_aud == all_vlabels_aud)
    # Compute the average accuracy and loss over all validate dataset
    val_acc_aud = np.float32(val_acc_aud / len(all_vlabels_aud))

    print('Accuracy (matched/mismatched):  ', val_acc * 100)
    print('Accuracy (video): ', val_acc_vid * 100)
    print('Accuracy (audio): ', val_acc_aud * 100)

    return all_vout_labels, all_vlabels, all_y_prob, all_vout_labels_vid, all_vlabels_vid, all_y_prob_vid, all_vout_labels_aud, all_vlabels_aud, all_y_prob_aud


def loadingfiles(feature_file, label_file):
    # Load extracted features
    print('\n')
    h5file = h5py.File(feature_file, mode='r')
    getKey = list(h5file.keys())[0]
    getData = h5file.get(getKey)
    features = np.asarray(getData)
    features = torch.from_numpy(features)
    h5file.close()

    labelValues = []
    labels_vid = []
    labels_aud = []
    with open(label_file,'r') as csvfile:
        csvReader = csv.reader(csvfile)
        for row in csvReader:
            labelValues.append(np.int((row[2])))
            labels_vid.append(np.int(row[5]))
            labels_aud.append(np.int(row[6]))
    labelValues = np.asarray(labelValues)
    labelValues = torch.from_numpy(labelValues)

    labels_vid = np.asarray(labels_vid)
    labels_vid= torch.from_numpy(labels_vid)

    labels_aud = np.asarray(labels_aud)
    labels_aud = torch.from_numpy(labels_aud)

    csvfile.close()

    return features, labelValues, labels_vid, labels_aud


def plot_cfs_matrix(cfs):
    df_cm = pd.DataFrame(cfs, lbl_range, lbl_range)
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, cmap="Reds", annot=True, fmt = ".5g", annot_kws={"size": 13})  # font size
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
    plt.plot(fpr, tpr, linestyle='--', color='orange', label='our model')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    plt.show()

    # auc score
    auc_score = roc_auc_score(y_truth, y_prob[:, 1])
    print("ROC AUC score: ", auc_score)
    print()



# Main
def main(args):
    # Device configuration
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Manual seed
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device: ', device)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Data
    # for model training
    video_train_features, train_labelValues, train_labelValues_vid, train_labelValues_aud = loadingfiles(video_feature_file_train, label_file_train)
    audio_train_features, _, _,_ = loadingfiles(audio_feature_file_train, label_file_train)

    memoryCheck()

    video_val_features, val_labelValues, val_labelValues_vid, val_labelValues_aud = loadingfiles(video_feature_file_validate, label_file_validate)
    audio_val_features, _, _, _ = loadingfiles(audio_feature_file_validate, label_file_validate)

    video_test_features, test_labelValues, test_labelValues_vid, test_labelValues_aud = loadingfiles(
        video_feature_file_test, label_file_test)
    audio_test_features, _, _, _ = loadingfiles(audio_feature_file_test, label_file_test)

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

    train_dataset = myDataloader(video_train_features, audio_train_features, train_labelValues, train_labelValues_vid, train_labelValues_aud, args, True)
    validate_dataset = myDataloader(video_val_features, audio_val_features, val_labelValues, val_labelValues_vid, val_labelValues_aud, args, False)
    test_dataset = myDataloader(video_test_features, audio_test_features, test_labelValues, test_labelValues_vid, test_labelValues_aud, args, False)

    memoryCheck()
    # input_size for the model
    video_dim = video_train_features.shape[1]
    audio_dim = audio_train_features.shape[1]

    m_start_time = time.time()
    # Build the model
    model = embedding_network(video_dim, audio_dim).to(device)
    model = model.to(device)
    memoryCheck()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion_video = nn.CrossEntropyLoss()
    criterion_audio = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.wd)

    model, train_losses, valid_losses = train_func(train_dataset, validate_dataset, model, optimizer, criterion, criterion_video, criterion_audio, device, args.num_epochs, args.patience)
    print('Training time: ', time.time() - m_start_time)

    test_voutput, test_grdtruth, test_pred_prob, test_voutput_vid, test_grdtruth_vid, test_pred_prob_vid, test_voutput_aud, test_grdtruth_aud, test_pred_prob_aud = validate_func(test_dataset, model, device)

    print("MATCHED/MISMATCHED CLASSIFICATION: ")
    # Confusion matrix
    cfs_matrix = confusion_matrix(list(np.asarray(test_grdtruth)), list(np.asarray(test_voutput)))
    plot_cfs_matrix(cfs_matrix)
    # compute F-1 score and roc_auc_score
    f1 = f1_score(np.asarray(test_grdtruth), np.asarray(test_voutput))  # , average="macro")
    print("F1-score for matched/mismatched classification:  ", f1)
    # compute ROC
    comput_roc(test_pred_prob, test_grdtruth)


    print("VIDEO EMOTION CLASSIFICATION: F1-score and AUC")
    # compute F-1 score and roc_auc_score
    f1_vid = f1_score(np.asarray(test_grdtruth_vid), np.asarray(test_voutput_vid), average="macro")
    print("F1-score: ", f1_vid)
    # compute ROC
    print("ROC: ", roc_auc_score(test_grdtruth_vid, test_pred_prob_vid.detach().numpy(), multi_class="ovo", average="macro"))


    print("AUDIO EMOTION CLASSIFICATION: F1-score and AUC")
    # compute F-1 score and roc_auc_score
    f1_aud = f1_score(np.asarray(test_grdtruth_aud), np.asarray(test_voutput_aud), average="macro")
    print("F1-score: ", f1_aud)
    # compute ROC
    print("ROC: ", roc_auc_score(test_grdtruth_aud, test_pred_prob_aud.detach().numpy(), multi_class="ovo", average="macro"))

    os.remove('./checkpoint.pt')


if __name__ == "__main__":
    dir_path = "./DS1_EmoMV_A"  # path to extracted features
    model_path = os.path.join(dir_path, 'models')  # path to save models => remember to create this folder
    pred_path = os.path.join(dir_path, 'predicted_values')  # path to save predicted values => rember to create this folder
    # ------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=model_path, help='path for saving trained models')
    # -------------------------------------------------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------------------------------------------------
    # for training
    label_file_train = os.path.join(dir_path + "/" + "annotation", "DS1_TRAIN_MATCH_MISMATCH_labels.csv")
    video_feature_file_train = os.path.join(dir_path + "/" + "extracted_features", "SlowFast_DS1_TRAIN_MATCH_MISMATCH.h5")
    audio_feature_file_train = os.path.join(dir_path + "/" +  "extracted_features", "VGGish_DS1_TRAIN_MATCH_MISMATCH.h5")

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






