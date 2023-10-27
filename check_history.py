import matplotlib.pyplot as plt

def show_all_history(history):
    print(f"Training accuracy: {history.history['auc']}")
    print(f"Training loss: {history.history['loss']}")  
    print(f"Validation accuracy: {history.history['val_auc']}")
    print(f"Validation loss: {history.history['val_loss']}")

    plt.figure(1)
    plt.plot(history.history['auc'])   
    plt.plot(history.history['val_auc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
