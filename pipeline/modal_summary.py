# This python script is for summarizing the model's architecture. To test them we include then in kaggle notebooks
import baseline_model
import resnet_sconns_model
import skconn_model

def get_summary():
    model = baseline_model()
    model = resnet_sconns_model.resnet_model().get_model()
    model = resnet_sconns_model()
    # model = skconn_model()
    model.summary()

if __name__ == '__main__':
    get_summary()
