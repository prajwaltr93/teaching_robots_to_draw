from global_model import getGlobalModel, checkModel
# from local_model import getLocalModel

weights_path = "./saved_model_weights/"

model = getGlobalModel()

model.load_weights(weights_path + "global_model_weights" )

checkModel(model)
