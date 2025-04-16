#import hydra
import logging
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__file__)

"""
@hydra.main(config_path=".", config_name="config")
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
    pass
"""

def main()-> None:
    #Creating a YAML file
    config = OmegaConf.create({
        "some_key" : "some_value",
        "item_list" : [1,"2",{"nested_dict":{"key1":"param","key2": 3}, "outside_key": 7}]
    })
    logger.info(OmegaConf.to_yaml(config))

    #Loading a YAML file
    config_2 = OmegaConf.load("./working_with_hydra/config.yaml") #Note: unlike hydra, the '.' refers to the current directory of execution and NOT w.r.t the file
    logger.info(OmegaConf.to_yaml(config_2))

    #Creating YAML file from a list of items
    item_list = ["training.batch_size=1024", "training.nrof_epochs=30", "model.lr=5e-4"]
    config_3 = OmegaConf.from_dotlist(item_list)
    logger.info(OmegaConf.to_yaml(config_3))

    #If you want to use cli for creating the args of the Config
    #config_4 = OmegaConf.from_cli()

    #Updating the learning rate
    config_2.training.lr = 5e-10
    
    #Adding new values
    config_2.training.new_key = "blah"

    #Accessing the config values
    learing_rate = config_2["training"]["lr"]
    training_size = config_2.training.batch_size

    logger.info(learing_rate)
    logger.info(training_size)

    logger.info(OmegaConf.to_yaml(config_2))
    
    

if __name__=="__main__":
    main()