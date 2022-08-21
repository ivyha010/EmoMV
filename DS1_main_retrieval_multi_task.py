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
from DS1_model_retrieval_multi_task import *
from sklearn.preprocessing import StandardScaler
import csv
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
from loss_functions import *

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
            return self.videoData[index], self.audioData[index], self.label[index], self.label_vid[index], \
                   self.label_aud[index]

        def __len__(self):
            return len(self.videoData)

    # Build dataloaders
    my_dataloader = DataLoader(dataset=my_dataset(videoFeatures, audioFeatures, labels, labels_vid, labels_aud),
                               batch_size=args.batch_size, shuffle=shuffleBool)
    return my_dataloader


def myDataloader_retrieval(videoFeatures, audioFeatures, emoValues, args, shuffleBool=False):
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
    my_dataloader = DataLoader(dataset=my_dataset(videoFeatures, audioFeatures, emoValues), batch_size=args.batch_size,
                               shuffle=shuffleBool)
    return my_dataloader


# Train
def train_func(train_loader, validate_loader, the_model, optimizer, criter, criter_vid, criter_aud, device, n_epochs,
               patience):
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
        # epoch_acc = 0
        # Adjust learning rate
        # adjust_learning_rate(optimizer, epoch)
        #####################
        ## train the model ##
        #####################
        the_model.train()  # prep model for training
        count_batches = 0
        for (video_feature, audio_feature, labels, labels_vid, labels_aud) in train_loader:
            video_feature, audio_feature, labels, labels_vid, labels_aud = video_feature.to(device), audio_feature.to(
                device), labels.to(device), labels_vid.to(device), labels_aud.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model
            sim, out_vid, out_aud = the_model.forward(video_feature, audio_feature)
            dist = 1.0 - sim

            loss = criter(dist, labels.float()) + criter_vid(out_vid, labels_vid) + criter_aud(out_aud, labels_aud)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward(retain_graph=True)
            # perform a single optimization step (parameter update)
            optimizer.step()

            # epoch_acc += acc.item()
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
        # val_epoch_acc = 0
        # data, label_for_model, target_disc, target_cont)
        for (v_video_feature, v_audio_feature, vLabels, vLabels_vid, vLabels_aud) in validate_loader:
            v_video_feature, v_audio_feature, vLabels, vLabels_vid, vLabels_aud = v_video_feature.to(
                device), v_audio_feature.to(device), vLabels.to(device), vLabels_vid.to(device), vLabels_aud.to(device)

            vsim, vout_vid, vout_aud = the_model(v_video_feature, v_audio_feature)
            vdist = 1.0 - vsim

            # validation loss:
            batch_valid_losses = criter.forward(vdist, vLabels.float()) + criter_vid.forward(vout_vid, vLabels_vid) + criter_aud.forward(vout_aud, vLabels_aud)
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

        # early_stopping needs the loss to check if it has decreased,
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


# Load extracted features and arousal/valence files
def loadingfiles(feature_file, label_file):
    # Load extracted features and arousal .h5 files
    print('\n')
    print('Loading h5 files containing extracted features......')
    loading_time = time.time()
    h5file = h5py.File(feature_file, mode='r')
    getKey = list(h5file.keys())[0]
    getData = h5file.get(getKey)
    features = np.asarray(getData)
    features = torch.from_numpy(features)
    h5file.close()

    labelValues = []
    labels_vid = []
    labels_aud = []
    with open(label_file, 'r') as csvfile:
        csvReader = csv.reader(csvfile)
        for row in csvReader:
            labelValues.append(np.int((row[2])))
            labels_vid.append(np.int(row[5]))
            labels_aud.append(np.int(row[6]))
    labelValues = np.asarray(labelValues)
    labelValues = torch.from_numpy(labelValues)

    labels_vid = np.asarray(labels_vid)
    labels_vid = torch.from_numpy(labels_vid)

    labels_aud = np.asarray(labels_aud)
    labels_aud = torch.from_numpy(labels_aud)

    csvfile.close()

    return features, labelValues, labels_vid, labels_aud


def loadingfiles_retrieval(feature_file, csv_filename):
    print('Loading h5 files containing extracted features......')
    loading_time = time.time()

    h5file = h5py.File(feature_file, mode='r')
    getKey = list(h5file.keys())[0]
    getData = h5file.get(getKey)
    features = np.asarray(getData)
    features = torch.from_numpy(features)
    h5file.close()

    print('Time for loading extracted features: ', time.time() - loading_time)

    all_filenames = []
    with open(csv_filename, 'r') as csvfile:
        csvReader = csv.reader(csvfile)
        for row in csvReader:
            all_filenames.append(row[2])  # 1: filename,  2: emotion label
    csvfile.close()
    return features, all_filenames


def get_audio_video_views(validate_loader, model_path, device):
    model = embedding_network().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    video_view = []
    audio_view = []
    with torch.no_grad():
        for (video_features, audio_features, _) in validate_loader:
            video_emb = model.video_projection(model.video_br(video_features))
            audio_emb = model.audio_projection(model.audio_br(audio_features))
            video_view.append(video_emb)
            audio_view.append(audio_emb)
    return torch.cat(video_view), torch.cat(audio_view)


def compute_similarity(query, data):
    cosine_sim = nn.CosineSimilarity()
    similarity = cosine_sim(query.unsqueeze(0), data)
    return similarity


def AvgP(y_pred_rank, queries_label, label):
    score = 0.0
    count = 0.0

    lab_rel = list(queries_label).count(label)
    for i, p in enumerate(y_pred_rank):
        if int(p) == int(label):
            count += 1
            score += count / (i + 1.0)
    if lab_rel == 0:
        avgP = 0.0
    else:
        avgP = (score * 1.0) / lab_rel
    return avgP


def prec_rec(RelN, y_pred, queries_label, label):
    prec_list = []
    rec_list = []
    for _id in RelN:
        count = 0.0
        lab_retrieved = _id
        lab_rel = list(queries_label).count(label)
        for i, p in enumerate(y_pred[:_id]):
            if int(p) == int(label):
                count += 1
        if lab_retrieved != 0:
            prec = (count * 1.0) / lab_retrieved
        else:
            prec = 0.0
        if lab_rel != 0:
            rec = (count * 1.0) / lab_rel
        else:
            rec = 0.0
        prec_list.append(prec)
        rec_list.append(rec)
    return np.asarray(prec_list), np.asarray(rec_list)


def metric(queries, view, queries_label):
    Ap_sum_view = 0.0
    query_num = queries.shape[0]
    prec_all, rec_all = [], []

    acc_count_1 = 0
    acc_count_3 = 0
    acc_count_5 = 0
    acc_count_10 = 0

    pre_final, rec_final = [], []
    for _idx in range(query_num):
        label = queries_label[_idx]
        sim_vector = compute_similarity(queries[_idx], view)
        rank_view_index = torch.argsort(sim_vector, dim=- 1, descending=True)
        pred_view_label = [queries_label[index] for index in rank_view_index]
        prec_list, rec_list = prec_rec(RelN, pred_view_label, queries_label, label)
        AP_view = AvgP(pred_view_label, queries_label, label)
        Ap_sum_view += AP_view
        prec_all.append(prec_list)
        rec_all.append(rec_list)

        # TopK
        values_top1, indices_top1 = torch.topk(sim_vector, 1)  # top1
        recommend_label_top1 = [queries_label[idx_top1] for idx_top1 in indices_top1]

        values_top3, indices_top3 = torch.topk(sim_vector, 3)  # top3
        recommend_label_top3 = [queries_label[idx_top3] for idx_top3 in indices_top3]

        values_top5, indices_top5 = torch.topk(sim_vector, 5)  # top5
        recommend_label_top5 = [queries_label[idx_top5] for idx_top5 in indices_top5]

        values_top10, indices_top10 = torch.topk(sim_vector, 10)  # top10
        recommend_label_top10 = [queries_label[idx_top10] for idx_top10 in indices_top10]

        if label in recommend_label_top1:
            acc_count_1 += 1
        if label in recommend_label_top3:
            acc_count_3 += 1
        if label in recommend_label_top5:
            acc_count_5 += 1
        if label in recommend_label_top10:
            acc_count_10 += 1

    mAp_view = float("{:.5f}".format((Ap_sum_view * 1.0) / query_num))
    print("mAP_view={}".format(mAp_view))

    prec_all, rec_all = np.array(prec_all), np.array(rec_all)
    [pre_final.append(np.mean(prec_all[:, i])) for i in range(prec_all.shape[1])]
    [rec_final.append(np.mean(rec_all[:, i])) for i in range(rec_all.shape[1])]

    print("accuracy_top1 (%): ", acc_count_1 / len(queries_label) * 100)
    print("accuracy_top3 (%): ", acc_count_3 / len(queries_label) * 100)
    print("accuracy_top5 (%): ", acc_count_5 / len(queries_label) * 100)
    #print("accuracy_top10 (%): ", acc_count_10 / len(queries_label) * 100)

    return mAp_view, pre_final, rec_final


def plot_precsion_recall_retrieval_task(rec, prec, query2retrieval=""):
    # Read this: https://ils.unc.edu/courses/2013_spring/inls509_001/lectures/10-EvaluationMetrics.pdf => Note: When plotting a PR curve, we use the best precision for a level of recall or greater!
    rec_np = np.array(rec)
    prec_np = np.array(prec)
    indices = np.argsort(-prec_np)  # decreaing order
    new_prec_np = [prec_np[id] for id in indices]
    new_rec_np = [rec_np[id] for id in indices]
    init = new_rec_np[0]
    get_rec = []
    get_prec = []
    for i in range(1, len(new_rec_np)):
        if new_rec_np[i] > init:
            get_rec.append(new_rec_np[i])
            get_prec.append(new_prec_np[i])
            init = new_rec_np[i]

    plt.plot(get_rec, get_prec, label=query2retrieval + " audio-visual retrieval")
    # plt.plot(rec_cca_va, prec_cca_va, label="CCA")

    plt.title('')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    leg = plt.legend(bbox_to_anchor=(0.65, 1), ncol=1, mode=None, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.65)
    plt.grid(True)
    # plt.savefig("./DS1_retrieval.png")
    plt.show()


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
    video_train_features, train_labelValues, train_labelValues_vid, train_labelValues_aud = loadingfiles(
        video_feature_file_train, label_file_train)
    audio_train_features, _, _, _ = loadingfiles(audio_feature_file_train, label_file_train)

    memoryCheck()

    video_val_features, val_labelValues, val_labelValues_vid, val_labelValues_aud = loadingfiles(
        video_feature_file_validate, label_file_validate)
    audio_val_features, _, _, _ = loadingfiles(audio_feature_file_validate, label_file_validate)

    # standardize:
    scaler_1 = StandardScaler()
    video_train_features = torch.from_numpy(scaler_1.fit_transform(video_train_features)).float()
    video_val_features = torch.from_numpy(scaler_1.transform(video_val_features)).float()

    scaler_2 = StandardScaler()
    audio_train_features = torch.from_numpy(scaler_2.fit_transform(audio_train_features)).float()
    audio_val_features = torch.from_numpy(scaler_2.transform(audio_val_features)).float()

    memoryCheck()
    train_dataset = myDataloader(video_train_features, audio_train_features, train_labelValues, train_labelValues_vid,
                                 train_labelValues_aud, args, True)
    validate_dataset = myDataloader(video_val_features, audio_val_features, val_labelValues, val_labelValues_vid,
                                    val_labelValues_aud, args, False)

    memoryCheck()
    # ------------------------------------------------------------------------------------------------
    # input_size for the model
    video_dim = video_train_features.shape[1]
    audio_dim = audio_train_features.shape[1]

    m_start_time = time.time()
    # Build the model
    model = embedding_network(video_dim, audio_dim).to(device)
    model = model.to(device)
    memoryCheck()
    # Loss and optimizer
    # Cross Entropy Loss
    criterion = ContrastiveLoss()
    criterion_video = nn.CrossEntropyLoss()
    criterion_audio = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.wd)

    model, train_losses, valid_losses = train_func(train_dataset, validate_dataset, model, optimizer, criterion,
                                                   criterion_video, criterion_audio, device, args.num_epochs,
                                                   args.patience)
    print('Training time: ', time.time() - m_start_time)

    # save model
    saved_model = "_DS1_model_retrieval_multi_task.pth"
    torch.save(model.state_dict(), os.path.join(args.model_path, saved_model))
    print("Saved the best model!")

    # Find matches
    print("FOR TESTING: ")
    retrieval_video_test_features, retrieval_test_labelValues = loadingfiles_retrieval(video_feature_retrieval_test,
                                                                                       label_retrieval_test)
    retrieval_audio_test_features, _ = loadingfiles_retrieval(audio_feature_retrieval_test, label_retrieval_test)
    retrieval_video_test_features = torch.from_numpy(scaler_1.transform(retrieval_video_test_features)).float()
    retrieval_audio_test_features = torch.from_numpy(scaler_2.transform(retrieval_audio_test_features)).float()

    retrieval_test_dataset = myDataloader_retrieval(retrieval_video_test_features, retrieval_audio_test_features,
                                                    retrieval_test_labelValues, args, False)

    vid_view, aud_view = get_audio_video_views(retrieval_test_dataset, os.path.join(args.model_path, saved_model),
                                               device)
    print("Video query => Retrieve music: ")
    vid2aud_mAP, vid2aud_precision_list, vid2aud_recall_list = metric(vid_view, aud_view, retrieval_test_labelValues)
    print("Music query => Retrieve videos: ")
    aud2vid_mAP, aud2vid_precision_list, aud2vid_recall_list = metric(aud_view, vid_view, retrieval_test_labelValues)
    plot_precsion_recall_retrieval_task(vid2aud_recall_list, vid2aud_precision_list, query2retrieval="Video to music:")
    plot_precsion_recall_retrieval_task(aud2vid_recall_list, aud2vid_precision_list, query2retrieval="Music to video:")

    #os.remove('./checkpoint.pt')


if __name__ == "__main__":
    dir_path = "./DS1_EmoMV_A"  # path to extracted features
    model_path = os.path.join(dir_path, 'models')  # path to save models => rember to create this folder
    pred_path = os.path.join(dir_path, 'predicted_values')  # path to save predicted values => rember to create this folder
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=model_path, help='path for saving trained models')
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=20,
                        help='early stopping patience; how long to wait after last time validation loss improved')
    parser.add_argument('--batch_size', type=int, default=256, help='number of feature vectors loaded per batch')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', type=float, default=0.1, help='weight decay')
    # parser.add_argument('--mm', type=float, default=0.9, help='momentum')

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
    dir_path_retrieval = "./DS1_EmoMV_A"
    label_retrieval_test = os.path.join(dir_path_retrieval + "/" + "for_retrieval", "DS1_from_MVED_test_set_5_classes_for_retrieval.csv")
    video_feature_retrieval_test = os.path.join(dir_path_retrieval + "/" + "for_retrieval", "SlowFast_DS1_from_MVED_test_set_5_classes_for_retrieval.h5")
    audio_feature_retrieval_test = os.path.join(dir_path_retrieval + "/" + "for_retrieval", "VGGish_DS1_from_MVED_test_set_5_classes_for_retrieval.h5")

    # -------------------------------------------------------------------------------------------------------------------
    main_start_time = time.time()
    lbl_range = ["Mismatched", "Matched"]  # mismatched: 0, matched: 1
    RelN = [i for i in range(1, 250)] # for EmoMV-B, EmoMV-C, remember to update this number
    main(args)
    print('Total running time: {:.5f} seconds'.format(time.time() - main_start_time))
