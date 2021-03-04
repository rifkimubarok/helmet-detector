from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="helmet")
trainer.setTrainConfig(object_names_array=["With Helmet", "Without Helmet"], batch_size=1, num_experiments=2, train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()