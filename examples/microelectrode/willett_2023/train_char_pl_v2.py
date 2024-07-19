import os
from datetime import datetime
from pathlib import Path
import logging

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from edit_distance import SequenceMatcher
from dataset import getDatasetLoaders
from model import GRUDecoder

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
import sentencepiece as spm

import jiwer

# define the LightningModule
class Lightning_GRUDecoder(L.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.args = args
        self.save_hyperparameters()
        if self.args["model"]["tokenizer"] == 'bpe':
            self.tokenizer = spm.SentencePieceProcessor(model_file=self.args["model"]["tokenizer_path"])
        

    def training_step(self, batch, batch_idx):
        X, y, X_len, y_len, dayIdx = batch

        # OPTIONAL: add white noise and constant offset
        if self.args["model"]["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device="cuda") * self.args["model"]["whiteNoiseSD"]
        if self.args["model"]["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device="cuda")
                * self.args["model"]["constantOffsetSD"]
            )

        # Compute prediction error
        pred = self.model.forward(X, dayIdx)

        loss = self.loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - self.args["model"]["kernelLen"]) / self.args["model"]["strideLen"]).to(torch.int32),
            y_len,
        )

        loss = torch.sum(loss)
        self.log("train_loss", loss)

        adjustedLens = ((X_len - self.args["model"]["kernelLen"]) / self.args["model"]["strideLen"]).to(
            torch.int32
        )

        total_edit_distance = 0
        total_seq_length = 0

        for iterIdx in range(pred.shape[0]):
            decodedSeq = torch.argmax(
                torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                dim=-1,
            )  # [num_seq,]
            decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
            decodedSeq = decodedSeq.cpu().detach().numpy()
            decodedSeq = np.array([i for i in decodedSeq if i != 0])

            trueSeq = np.array(y[iterIdx][0 : y_len[iterIdx]].cpu().detach())

            matcher = SequenceMatcher(a=trueSeq.tolist(), b=decodedSeq.tolist())
            total_edit_distance += matcher.distance()
            total_seq_length += len(trueSeq)
        cer = total_edit_distance / total_seq_length

        self.log("train_cer", cer, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y, X_len, y_len, dayIdx = batch

        # Compute prediction error
        pred = self.model.forward(X, dayIdx)

        loss = self.loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - self.args["model"]["kernelLen"]) / self.args["model"]["strideLen"]).to(torch.int32),
            y_len,
        )

        loss = torch.sum(loss)
        self.log("dev_loss", loss, sync_dist=True)

        adjustedLens = ((X_len - self.args["model"]["kernelLen"]) / self.args["model"]["strideLen"]).to(
            torch.int32
        )

        trueSeq_decodedSeq_text = []

        for iterIdx in range(pred.shape[0]):
            decodedSeq = torch.argmax(
                torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                dim=-1,
            )  # [num_seq,]
            decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
            decodedSeq = decodedSeq.cpu().detach().numpy()
            decodedSeq = np.array([i for i in decodedSeq if i != 0])

            trueSeq = np.array(y[iterIdx][0 : y_len[iterIdx]].cpu().detach())

            trueSeq_text = "".join(map(idToChar, trueSeq))
            decodedSeq_text = "".join(map(idToChar, decodedSeq))

            trueSeq_decodedSeq_text.append(
                [
                    self.current_epoch,
                    iterIdx,
                    jiwer.cer(trueSeq_text, decodedSeq_text),
                    jiwer.wer(trueSeq_text, decodedSeq_text),
                    trueSeq_text,
                    decodedSeq_text,
                ]
            )
        # get the list of all trueSeq_text
        all_trueSeq_text = [x[-2] for x in trueSeq_decodedSeq_text]
        all_decodedSeq_text = [x[-1] for x in trueSeq_decodedSeq_text]

        # calculate the CER and WER with jiwer
        ji_cer = jiwer.cer(all_trueSeq_text, all_decodedSeq_text)
        ji_wer = jiwer.wer(all_trueSeq_text, all_decodedSeq_text)

        self.logger.log_text(
            key="samples",
            columns=["EPOCH", "IDX", "JICER", "JIWER", "trueSeq", "decodedSeq"],
            data=trueSeq_decodedSeq_text,
        )
        # cer = total_edit_distance / total_seq_length

        # self.log("dev_cer", cer, sync_dist=True)
        self.log("dev_cer", ji_cer, sync_dist=True)
        self.log("dev_wer", ji_wer, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args["model"]["lrStart"],
            betas=(0.9, 0.999),
            eps=0.1,
            weight_decay=self.args["model"]["l2_decay"],
        )

        # return optimizer
    
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=self.args["model"]["lrEnd"] / self.args["model"]["lrStart"],
            total_iters=self.args["trainer"]["max_epochs"],
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch) 


# Define the phone set
PHONE_DEF = [
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]
PHONE_DEF_SIL = PHONE_DEF + ["SIL"] + ["_"]


def phoneToId(p):
    return PHONE_DEF_SIL.index(p)


def idToPhone(i):
    # zero is blank
    if i == 0:
        return "<BLANK>"
    else:
        return PHONE_DEF_SIL[i - 1]

CHAR_DEF = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    "'", 
    '-',
    'SIL'
]

CHAR_DEF_SIL = CHAR_DEF + ["_"]

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


# Parse config with hydra
@hydra.main(version_base=None, config_path="configs", config_name="config_bpe")
def main(cfg: DictConfig) -> None:

    # ------------------------------
    # Congif and local logging setup
    # ------------------------------

    # Logging
    # A logger for this file
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    # Log all the configs
    logging.info(OmegaConf.to_yaml(cfg))

    # Add unique config parameters for experiment tracking
    cfg.base.date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    cfg.base.exp_name = f"{cfg.base.date}_{cfg.base.model_name}_{cfg.base.note}"
    cfg.base.exp_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Create the output directory if it does not exist
    os.makedirs(cfg.base.exp_dir, exist_ok=True)

    # Save modified cfg to yaml file
    OmegaConf.save(config=cfg, f=Path(cfg.base.exp_dir) / cfg.base.config_file)

    # Resolve the config for wandb logging
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
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

    gru = GRUDecoder(
        neural_dim=cfg.model.nInputFeatures,
        n_classes=cfg.model.nClasses,
        hidden_dim=cfg.model.nUnits,
        layer_dim=cfg.model.nLayers,
        nDays=len(loadedData["train"]),
        dropout=cfg.model.dropout,
        device="cuda",
        strideLen=cfg.model.strideLen,
        kernelLen=cfg.model.kernelLen,
        gaussianSmoothWidth=cfg.model.gaussianSmoothWidth,
        bidirectional=cfg.model.bidirectional,
    )

    model = Lightning_GRUDecoder(gru, wandb.config)

    # Configure environment variables for DDP
    if cfg.cluster.nodes > 1:
        os.environ["MASTER_PORT"] = cfg.cluster.master_port
        os.environ["MASTER_ADDR"] = cfg.cluster.master_addr
        os.environ["WORLD_SIZE"] = cfg.cluster.world_size
        os.environ["NODE_RANK"] = cfg.cluster.node_rank

    # Configure the ML logger
    if cfg.trainer.ml_logging == "mlflow":
        import mlflow
        from lightning.pytorch.loggers import MLFlowLogger
        #TODO: test mlflow locally
        mlflow.pytorch.autolog()
        ml_logger = MLFlowLogger(
            experiment_name=cfg.base.exp_name,
            tracking_uri="file:./ml-runs",
        )
    elif cfg.trainer.ml_logging == "wandb":
        from lightning.pytorch.loggers import WandbLogger

        ml_logger = WandbLogger(
            log_model=True,
            name=cfg.base.exp_name,
            project=cfg.trainer.project_name,
            save_dir=cfg.base.exp_dir,
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="dev_cer",
            mode="min",
            save_top_k=3,
            auto_insert_metric_name=True,
            dirpath=cfg.base.exp_dir,
        )

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        devices=cfg.trainer.gpus,
        accelerator="auto",
        strategy=cfg.trainer.strategy,
        num_nodes=cfg.cluster.nodes,
        precision=cfg.trainer.precision,
        callbacks=[checkpoint_callback],
        logger=ml_logger,
    )
    # Start training
    trainer.fit(model=model, train_dataloaders=trainLoader, val_dataloaders=devLoader)

if __name__ == "__main__":
    # USAGE: export CUDA_VISIBLE_DEVICES=1; python train_bpe_pl.py --config-name=config_bpe_64.yaml base.note=testing
    main()