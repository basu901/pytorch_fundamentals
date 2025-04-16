from omegaconf import DictConfig, OmegaConf
import logging
import os

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__file__)

def main():
    config = OmegaConf.load("./working_with_hydra/config.yaml")

    #'optimizer' has been defined as mandatory in the YAML file. 
    # If you try to access 'optimizer' without setting it, you will receive an error.

    #Following will be an error
    #logger.info(config.optimizer)

    #Setting value before access
    config.optimizer = "adam"

    os.environ["USER"] = "shaunak.basu"
    os.environ["PASSWORD1"] = "hello_password"

    logger.info(config.optimizer)

    logger.info(OmegaConf.to_yaml(config,resolve=True))

    #You can merge multiple config files as well using OmegaConf
    #config_1 = OmegaConf.load('\path\to\first\config')
    #config_2 = OmegaConf.load('\path\to\second\config')

    #config3 = OmegaConf.merge(config_1,config_2)

    #Overriding values passed in CLI
    #config3.merge_with_cli()

if __name__=="__main__":
    main()