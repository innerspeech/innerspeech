from pathlib import Path
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from dataset import getDatasetLoaders
import lightning as L
from lightning.pytorch import seed_everything
from train_phoneme_pl import Lightning_GRUDecoder
import jiwer
import neural_decoder.lmDecoderUtils as lmDecoderUtils

from tqdm import tqdm

# Parse config with hydra
@hydra.main(version_base=None, config_path="configs", config_name="exp_config")
def main(cfg: DictConfig) -> None:

    # ------------------------------
    # Start the experiment
    # ------------------------------
        
    # Seed everything
    seed_everything(cfg.trainer.seed)

    # Load dataloaders
    trainLoader, devLoader, testLoader, loadedData = getDatasetLoaders(
        cfg.dataset.preprocessed_path,
        cfg.trainer.batch_size,
    )


    # get the last checkpoint in directory
    model_dir = cfg.base.exp_dir
    last_checkpoint = sorted(Path(model_dir).glob("*.ckpt"))[-1]

    print(f"Loading model from {last_checkpoint}")

    model = Lightning_GRUDecoder.load_from_checkpoint(last_checkpoint)
    
    print("Model args", model.args) 

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        devices=cfg.trainer.gpus,
        accelerator="auto",
        strategy=cfg.trainer.strategy,
        num_nodes=cfg.cluster.nodes,
        precision=cfg.trainer.precision,
    )

    model.cuda()
    model.eval()

    decoder = model.decoder

    # LM decoding hyperparameters
    acoustic_scale = 0.5
    blank_penalty = np.log(7)
    llm_weight = 0.5


    llm, llm_tokenizer = lmDecoderUtils.build_opt(device="auto")
    # 5gram
    # lmDir = "./data/speech_5gram/lang_test/"
    # 3gram
    # lmDir = "data/languageModel"
    # train
    if cfg.model.lm_version == "train":
        lmDir = "/home/ubuntu/neural_seq_decoder/speechBCI/AnalysisExamples/lm_model/data/lang_test"
    elif cfg.model.lm_version == "3-gram":
        lmDir = "/home/ubuntu/innerspeech/examples/utah_speechbci/data/languageModel"

    ngramDecoder = lmDecoderUtils.build_lm_decoder(
        lmDir, acoustic_scale=0.5, nbest=100, beam=18
    )

    all_decodedSeq_text = []
    all_decodedSeq_text_nbest = []

    rnn_outputs = {
        "logits": [],
        "logitLengths": [],
        "trueSeqs": [],
        "transcriptions": [],
    }

    portion = cfg.eval.portion

    if portion == "competition":
        loader = testLoader
    else:
        loader = devLoader

    # loop the examples in tesloader one by one
    for i, batch in enumerate(tqdm(loader)):
        # get the input and target
        X, y, X_len, y_len, dayIdx, _ = batch
        
        # Hotfix for dayIdx in test
        if portion == "competition":
            for x in dayIdx:
                if x < 7:
                    x += 4
                elif x < 12:
                    x += 5
                else:
                    x += 6
        
        pred = model.model.forward(X.cuda(), dayIdx.cuda())

        adjustedLens = ((X_len - model.args["model"]["kernelLen"]) / model.args["model"]["strideLen"]).to(
            torch.int32
        )

        for iterIdx in range(pred.shape[0]):

            trueSeq = np.array(y[iterIdx][0 : y_len[iterIdx]].cpu().detach())

            logits = pred[iterIdx].cpu().detach().numpy()
            logits = np.concatenate(
                    [logits[:, 1:], logits[:, 0:1]], axis=-1
                ) 
            logits = lmDecoderUtils.rearrange_speech_logits(logits[None, :, :], has_sil=True)

            # print(logits.shape)

            nbest_outputs = lmDecoderUtils.lm_decode(
                ngramDecoder,
                logits[0],
                blankPenalty=blank_penalty,
                returnNBest=True,
                rescore=True,
            )

            # Get only the nbests
            n_best_texts = [nbest[0] for nbest in nbest_outputs]

            # print(n_best_texts)

            decodedTranscriptions, confidence = lmDecoderUtils.gpt2_lm_decode(model=llm,
                                          tokenizer=llm_tokenizer,
                                          nbest=nbest_outputs,
                                          acousticScale=acoustic_scale,
                                          lengthPenlaty=0.0,
                                          alpha=llm_weight,
                                          returnConfidence=True)

            print(decodedTranscriptions)

            all_decodedSeq_text.append(decodedTranscriptions)

            # rnn_outputs["trueSeqs"].append(trueSeq)

            # logits = torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :])
            # decodedSeq = torch.argmax(logits, dim=-1)
            # decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
            # decodedSeq = decodedSeq.cpu().detach().numpy()
            # decodedSeq = np.array([i for i in decodedSeq if i != 0])
            # # 1 best - by pyctcdecode
            # #
            # text = decoder.decode(logits.cpu().detach().numpy())

            # print("trueSeq", trueSeq)
            # print("decodedSeq", decodedSeq)

            # # print('1-best', text)
            # all_decodedSeq_text.append(text)

            # nbests = decode_nbest(decoder, logits.cpu().detach().numpy(), nbest=10)
            # rescored_nbests = rescore_nbest(nbests, None, llm_weight=0.5)
            # # sort by final score
            # rescored_nbests_sorted = sorted(rescored_nbests, key=lambda x: x["final_score"], reverse=True)
            # # print('nbest', rescored_nbests_sorted[0]['text'])
            # top_best = rescored_nbests_sorted[0]
            # all_decodedSeq_text_nbest.append(top_best['text'])

    # write the list to a file
    if portion == 'test':
        # get the ground truth
        transcript_path = "/home/ubuntu/innerspeech/examples/utah_speechbci/data/true_dev_Transcriptions.txt"
        with open(transcript_path, "r") as f:
            true_transcripts = f.readlines()
        # calculate the WER
        wer = jiwer.wer(true_transcripts, all_decodedSeq_text)
        print(f"WER: {wer}")
    elif portion == 'competition':
        wer = 0.0
    

    with open(f"results/decodedSeq_{portion}_{cfg.base.exp_name}_{cfg.model.lm_version}_{wer:.4f}.txt", "w") as f:
        for item in all_decodedSeq_text:
            f.write(f"{item.strip()}\n")
        
    # with open(f"results/decodedSeq_nbest_{cfg.base.exp_name}.txt", "w") as f:
    #     for item in all_decodedSeq_text_nbest:
    #         f.write(f"{item.strip()}\n")


def decode_nbest(decoder, logits, nbest=3):
    # Decode nbest
    nbests_with_meta = decoder.decode_beams(logits)
    nbests = []
    for nbest_with_meta in nbests_with_meta[:nbest]:
        # nbest_with_meta[0] = text, 
        # nbest_with_meta[-2] = logit_score: float  # Cumulative logit score
        # nbest_with_meta[-1] = lm_score: float  # Cumulative language model + logit score
        text = nbest_with_meta[0]
        logit_score = nbest_with_meta[-2]
        logit_plus_arpa_score = nbest_with_meta[-1]
        arpa_score = logit_plus_arpa_score - logit_score

        nbests.append({"text": text,
                      "am_score": logit_score, 
                      "logit_plus_arpa_score": logit_plus_arpa_score,
                      "arpa_score": arpa_score})
    return nbests


def rescore_nbest(nbests, lm_model, llm_weight=0.0):

    nbest_final_scores = []
    
    # Rescore nbest
    for idx, nbest in enumerate(nbests):
        text = nbest['text']
        # get scroring from LLM
        llm_lm_score = -1

        final_score = nbest["am_score"] + (nbest['arpa_score'] * (1 - llm_weight)) + (llm_lm_score * llm_weight)

        nbests[idx]["final_score"] = final_score
    return nbests

if __name__ == "__main__":
    # USAGE: export CUDA_VISIBLE_DEVICES=0 && python predict.py --config-path=outputs/2024-03-17/19-52-20/
    main()