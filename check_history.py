import matplotlib.pyplot as plt

def show_all_history(history):
    print(f'KEYS: {history.history.keys()}')
    print(f"Training accuracy: {history.history['accuracy']}")
    print(f"Training loss: {history.history['loss']}")  
    print(f"Validation accuracy: {history.history['val_accuracy']}")
    print(f"Validation loss: {history.history['val_loss']}")

    plt.figure(1)

    plt.subplot(121)
    plt.plot(history.history['accuracy'])   
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.subplot(122)
    plt.plot(history.history['loss'])   
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.tight_layout()
    plt.show()

