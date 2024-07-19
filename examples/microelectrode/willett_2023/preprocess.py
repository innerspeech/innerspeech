from dataset import preprocess_neural_data
import yaml
import hydra
from omegaconf import DictConfig

def parse_config(args):
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    return configs

@hydra.main(version_base=None, config_path="configs", config_name="config_phoneme")
def main(cfg: DictConfig) -> None:
    preprocess_neural_data(data_dir=cfg.dataset.raw_path, 
                           output_dir=cfg.dataset.preprocessed_path,
                           cfg=cfg)

if __name__ == '__main__':
    # USAGE: python preprocess.py [--config-name=config_bpe]
    main()