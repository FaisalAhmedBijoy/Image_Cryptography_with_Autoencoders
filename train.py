import matplotlib.pyplot as plt
from data_loader import load_data
from auto_encoder import autoencoder_architecture

def visualize_training_performance(history,save_loss_graph_filepath):
    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    # plt.show()
    plt.savefig(save_loss_graph_filepath, dpi=500)


if __name__=='__main__':
    
    epochs = 50
    batch_size = 32
    input_image_shape = (256, 256, 1)
    dataset_dir='datasets/test'
    saved_model_path = 'logs/autoencoder_model.h5'
    saved_architecture_path = "images/model_architecture_and_performances/autoencoder_architecture.png"
    save_loss_graph_filepath='images/model_architecture_and_performances/loss_graph.png'
    
    train_images,val_images=load_data(dataset_dir)
    autoencoder=autoencoder_architecture(input_image_shape, saved_architecture_path)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    history=autoencoder.fit(train_images, 
                            train_images, 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            validation_data=(val_images, val_images))

    # Save the trained model
    autoencoder.save(saved_model_path)
    print("Model training saved to " + saved_model_path)
    # visualize_training_performance(history,save_loss_graph_filepath)
  
   