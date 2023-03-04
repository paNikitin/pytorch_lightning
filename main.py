import fmnist_classifier
import fmnist_data
import pytorch_lightning as pl
import hydra
import omegaconf

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: omegaconf.DictConfig):
    model = fmnist_classifier.LightningFMNISTClassifier(cfg.opt)
    trainer = pl.Trainer( max_epochs=cfg.params.max_epochs)
    data_module = fmnist_data.FMNISTDataModule(cfg.params.batch_size)
    trainer.fit(model, data_module)
    model.eval()
    #trainer.predict(model, data_module)
    #print(model.get_accuracy())
if __name__ == "__main__":
    main()