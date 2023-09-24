# Image_Cryptography_with_Autoencoders
Deep learning in image Cryptography

Image encryption and decryption using a chaotic map sequence and an autoencoder. The repository includes code for generating chaotic map sequences, shuffling and deshuffling images, preparing datasets, and computing performance metrics.

## Environment setup
```bash
conda env create --file environment.yml
```
**Install packages**

```bash
keygen==0.0.3
matplotlib==3.3.3
numpy==1.23.1
numpy==1.23.2
opencv_python==4.4.0.46
scikit_learn==0.23.2
skimage==0.0
tensorflow==2.3.0
```

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```
## 1. Chaotic map sequence generation
To generate the chaotic map sequence, run the following command:
```bash
python generate_chaotic_map_sequence.py
```
## 2. Shuffle and Deshuffle image
To shuffle and deshuffle an image, run the following command:
```bash 
python shuffling_deshuffling_images.py
```
**Input image**

<img src="images/input_samples/lena_gray.gif" width="256" height="256">
<!-- ![](images/input_samples/lena_gray.gif) -->

**Shuffle image** 

![](images/shuffled_deshuffled_image/Lena_shuffled_image.png)

**Deshuffle image**

![](images/shuffled_deshuffled_image/Lena_deshuffled_image.png)

## 3. Encryption and Decryption of image using chaotic map sequence
To encrypt and decrypt an image using chaotic map sequence, run the following command:
```bash
python image_encryption_decryption.py
```
**Encryption of image**

![](images/encrypted_decrypted_images/Lena_encrypted_image.png)

**Decryption of image**

![](images/encrypted_decrypted_images/Lena_decrypted_image.png)

## 4. Dataset preparation 
To prepare the dataset, run the following command:
```bash
python data_loader.py
```

## 5. Autoencoder for image encryption and decryption
To use the autoencoder for image encryption and decryption, run the following command:
```bash
python auto_encoder.py
```

**Model architecture**

![](images/model_architecture_and_performances/autoencoder_architecture.png)


**Model training**
```bash
python train.py
```
**Model loss performance graph**

![](images/model_architecture_and_performances/loss_graph.png)

## 6. Inferencing the model

*original_vs_compressed_vs_reconstruction*

```bash
python inference.py
```
**Original image**

![](images/model_architecture_and_performances/original_image.png)

**encoder compressed image**

![](images/model_architecture_and_performances/compressed_encoded_image.png)

**Decoder decompressed image**

![](images/model_architecture_and_performances/decompressed_decoded_image.png)

## 7. Performance metrics

- Structural Similarity Index (SSIM)

```bash
python structural_similarity_SSIM_calculation.py
```
- Number of pixel change rate (NPCR)

```bash
python nnumber_of_pixel_change_rate_NPCR_comparision.py
```
- NPCR computation

**image 1 output**

![](images/NPCR_images/NPCR_difference_1_LENA.png)

**image 2 output**

![](images/NPCR_images/NPCR_difference_2_LENA.png)

- Unified Average Changing Intensity (UACI)

```bash
python unified_average_changing_intensity_UACI_comparision.py
```
![](images/UACI_images/UACI_difference_LENA.png)

## 8. Salt and paper noise
    
 ```bash
   python salt_and_pepper_noise.py
```

**Noisy image**

![](images/noisy_images/lena_noisy.png)
