from model import Tagger 

if __name__ == "__main__":
  
    corpus = ['ba', 'b', 'a', 'baa', 'a', 'baaa', 'aa', 'b', 'abaa', 'abb', 'bb']
    labels = ['11', '1', '1', '110', '1', '1100', '10', '1', '1010', '101', '11']
    corpus_ = [ "e"+x+"z"*(4-len(x)) for x in corpus]
    states = []

    filepath = "weigths/model_weights.h5"

    trained_model = Tagger(3, 10, 10)
    trained_model.load_weights(filepath)
    """Mettre un indicatif pour que le modèle ne se mette à à jour"""
    train_results = trained_model(corpus_, y_train)

    train_preds = train_results["predictions"]

    representations = train_results["states"]