import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import pickle
from pathlib import Path
import glob
import re
from g2p_en import G2p
import numpy as np
import scipy
import os

PHONE_DEF = [
        'AA', 'AE', 'AH', 'AO', 'AW',
        'AY', 'B',  'CH', 'D', 'DH',
        'EH', 'ER', 'EY', 'F', 'G',
        'HH', 'IH', 'IY', 'JH', 'K',
        'L', 'M', 'N', 'NG', 'OW',
        'OY', 'P', 'R', 'S', 'SH',
        'T', 'TH', 'UH', 'UW', 'V',
        'W', 'Y', 'Z', 'ZH'
    ]
PHONE_DEF_SIL = PHONE_DEF + ['SIL']

CHAR_DEF = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    "'", '-', '_'
    ]

CHAR_DEF_SIL = CHAR_DEF + ['SIL']

def idToChar(i):
    # zero is blank
    if i == 0:
        return "<BLANK>"
    elif CHAR_DEF_SIL[i - 1] == 'SIL':
        return " "
    else:
        return CHAR_DEF_SIL[i - 1]

def charToId(c):
    """
    Convert a character to its corresponding index in the PHONE_DEF list
    """
    return CHAR_DEF.index(c)


def phoneToId(p):
    """
    Convert a phone to its corresponding index in the PHONE_DEF list
    """
    return PHONE_DEF_SIL.index(p)


def loadFeaturesAndNormalize(sessionPath, full_data=False):
    
    dat = scipy.io.loadmat(sessionPath)

    input_features = []
    transcriptions = []
    frame_lens = []
    n_trials = dat['sentenceText'].shape[0]

    #collect area 6v tx1 and spikePow features
    for i in range(n_trials):    
        #get time series of TX and spike power for this trial
        #first 128 columns = area 6v only
        if full_data:
            features = np.concatenate([dat['tx1'][0,i], dat['spikePow'][0,i]], axis=1)
        else:
            features = np.concatenate([dat['tx1'][0,i][:,0:128], dat['spikePow'][0,i][:,0:128]], axis=1)

        sentence_len = features.shape[0]
        sentence = dat['sentenceText'][i].strip()

        input_features.append(features)
        transcriptions.append(sentence)
        frame_lens.append(sentence_len)

    #block-wise feature normalization
    blockNums = np.squeeze(dat['blockIdx'])
    blockList = np.unique(blockNums)
    blocks = []
    for b in range(len(blockList)):
        sentIdx = np.argwhere(blockNums==blockList[b])
        sentIdx = sentIdx[:,0].astype(np.int32)
        blocks.append(sentIdx)

    for b in range(len(blocks)):
        feats = np.concatenate(input_features[blocks[b][0]:(blocks[b][-1]+1)], axis=0)
        feats_mean = np.mean(feats, axis=0, keepdims=True)
        feats_std = np.std(feats, axis=0, keepdims=True)
        for i in blocks[b]:
            input_features[i] = (input_features[i] - feats_mean) / (feats_std + 1e-8)

    #convert to tfRecord file
    session_data = {
        'inputFeatures': input_features,
        'transcriptions': transcriptions,
        'frameLens': frame_lens
    }

    return session_data


def getDataset(fileName, cfg=None):

    g2p = G2p()

    session_data = loadFeaturesAndNormalize(fileName, False)
        
    allDat = []
    trueSentences = []
    seqElements = []
    
    for x in range(len(session_data['inputFeatures'])):
        allDat.append(session_data['inputFeatures'][x])
        
        thisTranscription = str(session_data['transcriptions'][x]).strip()
        thisTranscription = re.sub(r'[^a-zA-Z\- \']', '', thisTranscription)
        thisTranscription = thisTranscription.replace('--', '').lower()
        session_data['transcriptions'][x] = thisTranscription
        trueSentences.append(session_data['transcriptions'][x])

        print(thisTranscription)
        addInterWordSymbol = True

        phonemes = []

        if cfg.model.tokenizer == 'char':
            char_list = list(thisTranscription)
            # replace white space with 'SIL'
            char_list = [c if c != ' ' else 'SIL' for c in char_list]
            char_list.append('SIL')
            phonemes = char_list
        elif cfg.model.tokenizer == 'phoneme':
            for p in g2p(thisTranscription):
                if addInterWordSymbol and p==' ':
                    phonemes.append('SIL')
                p = re.sub(r'[0-9]', '', p)  # Remove stress
                if re.match(r'[A-Z]+', p):  # Only keep phonemes
                    phonemes.append(p)

            #add one SIL symbol at the end so there's one at the end of each word
            if addInterWordSymbol:
                phonemes.append('SIL')
        elif cfg.model.tokenizer == 'bpe':
            import sentencepiece as spm
            s = spm.SentencePieceProcessor(model_file=cfg.model.tokenizer_path)
            encoded = s.encode_as_ids(thisTranscription)
            phonemes = encoded

        print(phonemes)

        seqLen = len(phonemes)
        maxSeqLen = 500
        seqClassIDs = np.zeros([maxSeqLen]).astype(np.int32)
        if cfg.model.tokenizer == 'char':
            seqClassIDs[0:seqLen] = [charToId(p) + 1 for p in phonemes]
        elif cfg.model.tokenizer == 'phoneme':
            seqClassIDs[0:seqLen] = [phoneToId(p) + 1 for p in phonemes]
        elif cfg.model.tokenizer == 'bpe':
            seqClassIDs[0:seqLen] = [p + 1 for p in phonemes]
        seqElements.append(seqClassIDs)

    newDataset = {}
    newDataset['sentenceDat'] = allDat
    newDataset['transcriptions'] = trueSentences
    newDataset['phonemes'] = seqElements
    
    timeSeriesLens = []
    phoneLens = []
    for x in range(len(newDataset['sentenceDat'])):
        timeSeriesLens.append(newDataset['sentenceDat'][x].shape[0])
        
        zeroIdx = np.argwhere(newDataset['phonemes'][x]==0)
        phoneLens.append(zeroIdx[0,0])
    
    newDataset['timeSeriesLens'] = np.array(timeSeriesLens)
    newDataset['phoneLens'] = np.array(phoneLens)
    newDataset['phonePerTime'] = newDataset['phoneLens'].astype(np.float32) / newDataset['timeSeriesLens'].astype(np.float32)
    return newDataset

from tqdm import tqdm

def preprocess_neural_data(data_dir='./data/competitionData/', output_dir='./data/preprocessed', cfg=None):

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Get the list of all the session names
    all_train_test_comp_files = glob.glob(str(data_dir / "*" / "*.mat"))
    session_names = set([Path(f).stem for f in all_train_test_comp_files])
    session_names_sorted = sorted(list(session_names))

    trainDatasets = []
    testDatasets = []
    competitionDatasets = []

    for dayIdx in tqdm(range(len(session_names_sorted))):
        trainDataset = getDataset(data_dir.as_posix() + '/train/' + session_names_sorted[dayIdx] + '.mat', cfg=cfg)
        testDataset = getDataset(data_dir.as_posix() + '/test/' + session_names_sorted[dayIdx] + '.mat', cfg=cfg)

        trainDatasets.append(trainDataset)
        testDatasets.append(testDataset)

        if os.path.exists(data_dir.as_posix() + '/competitionHoldOut/' + session_names_sorted[dayIdx] + '.mat'):
            dataset = getDataset(data_dir.as_posix() + '/competitionHoldOut/' + session_names_sorted[dayIdx] + '.mat', cfg=cfg)
            competitionDatasets.append(dataset)

    allDatasets = {}
    allDatasets['train'] = trainDatasets

    #save all transcriptions to a file
    allTranscriptions = []
    for x in range(len(trainDatasets)):
        allTranscriptions.extend(trainDatasets[x]['transcriptions'])
    with open('./data/true_train_Transcriptions.txt', 'w') as f:
        for x in allTranscriptions:
            f.write(x + '\n')

    allDatasets['test'] = testDatasets
    # save all dev transcriptions to a file
    allTranscriptions = []
    for x in range(len(testDatasets)):
        allTranscriptions.extend(testDatasets[x]['transcriptions'])
    with open('./data/true_dev_Transcriptions.txt', 'w') as f:
        for x in allTranscriptions:
            f.write(x + '\n')
    allDatasets['competition'] = competitionDatasets

    print('Saving preprocessed data to: ', output_dir)
    with open(output_dir, 'wb') as handle:
        pickle.dump(allDatasets, handle)
    

class SpeechDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.n_days = len(data)
        self.n_trials = sum([len(d["sentenceDat"]) for d in data])

        self.neural_feats = []
        self.phone_seqs = []
        self.neural_time_bins = []
        self.phone_seq_lens = []
        self.days = []
        self.transcriptions = []
        for day in range(self.n_days):
            for trial in range(len(data[day]["sentenceDat"])):
                self.neural_feats.append(data[day]["sentenceDat"][trial])
                self.phone_seqs.append(data[day]["phonemes"][trial])
                self.neural_time_bins.append(data[day]["sentenceDat"][trial].shape[0])
                self.phone_seq_lens.append(data[day]["phoneLens"][trial])
                self.days.append(day)
                self.transcriptions.append(data[day]["transcriptions"][trial])

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        neural_feats = torch.tensor(self.neural_feats[idx], dtype=torch.float32)

        if self.transform:
            neural_feats = self.transform(neural_feats)

        return (
            neural_feats,
            torch.tensor(self.phone_seqs[idx], dtype=torch.int32),
            torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
            torch.tensor(self.phone_seq_lens[idx], dtype=torch.int32),
            torch.tensor(self.days[idx], dtype=torch.int64),
            self.transcriptions[idx]
        )


def getDatasetLoaders(datasetName, batchSize):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days, transcriptions = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
            transcriptions
        )

    train_data = SpeechDataset(loadedData["train"], transform=None)
    dev_data = SpeechDataset(loadedData["test"], transform=None)
    test_data = SpeechDataset(loadedData["competition"], transform=None)

    train_loader = DataLoader(
        train_data,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, dev_loader, test_loader, loadedData