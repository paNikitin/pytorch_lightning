import fmnist_classifier
import fmnist_data
import pytorch_lightning as pl
import hydra
from config import FMNISTConfig
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="fmnist_config", node=FMNISTConfig)

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: FMNISTConfig):
    model = fmnist_classifier.LightningFMNISTClassifier(cfg.params.optimizer, cfg.params.lr)
    trainer = pl.Trainer( max_epochs=cfg.params.max_epochs)
    data_module = fmnist_data.FMNISTDataModule(cfg.params.batch_size)
    trainer.fit(model, data_module)
    #model.eval()
    #trainer.predict(model, data_module)
    #print(model.get_accuracy())
    
if __name__ == "__main__":
    main()