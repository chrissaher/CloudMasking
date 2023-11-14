from cloudmasking.models.unet import get_model as get_unet_model

models = {
    'unet': get_unet_model
}

def get_model(model_name):
    if model_name not in models.keys():
        print(f"{model_name} is an unsupported architecture. Supported values are: {models.keys()}")
        exit(0)
    model_fn =  models[model_name]
    return model_fn()
