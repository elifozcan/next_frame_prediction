# Next Frame Prediction
Next Frame prediction is an application of AI which involves predicting the next few frames of a video given the previous frames. This prediction is about getting the history of images or understanding the sequence while the input being the previous few frames, and prediction are the next frames.

These predictions can be anything, for example a video frame predictor can be shown several movies of a specific genre, such as romance movies or action thrillers. The video frame predictor can learn the probability distribution of the frames from the second half of a movie given the frames from the first half. But predicting the other half of the still far from what is achieved today, still many scenes in real life can be predicted since they satisfy physical laws, such as ball parabola prediction for a ping-pong robot.

In this project, we will try different deep learning models for predicting the next frames and continue our path with the best model, discussing ideas for predicting and performance metrics. Also, try to understand which fields next frame prediction can be used.

## Datasets
### Circle Dataset
For the next frame prediction application, we created our video data set. This data set contains 1000 videos for training and 200 for testing. Each video is 4 seconds and has 24 frames per second. Each frame is 80x80 in size. There are 2 circles in the videos, one big and one small, and these circles move linearly in the randomly determined direction. The Circle dataset can be obtained by running `Database/create_videos.ipynb`.

### Moving Mnsit
While training the models we also used Moving Mnist dataset beside circle dataset and the reason for this is to compare our results with other studies that usually uses the Moving Mnsit dataset.
In the pix2pix model only one video is used to train the model and normally Moving Mnist Dataset has 10.000 videos each containing 20 frames so we have to use 10 frames for training and 10 frames for testing. After getting the results we thought 10 frames can be so small for training and we create a video with 60 frames of Mnist data. The moving mnist dataset can download via this link: [Moving Mnist Dataset](http://www.cs.toronto.edu/~nitish/unsupervised_video/)

## Performance Analysis
In this chapter, the performances of every model we trained in both of the performance metrics we used for our project and our comments on the results are mentioned.
### Moving Mnist Dataset
For the Moving Mnsit dataset, we trained 2 conv_lstm models and the pix2pix model with different sizes of sets and epochs. In the table below RMSE values of models we trained with Moving Mnist are given.

| Model | Frame 1 | Frame 2 | Frame 4 | Frame 6 | Frame 8 | Frame 10 |
| --- | --- | --- | --- | --- | --- | --- |
Conv_lstm model1 | 39.53 | 81.72 | 77.04 | 75.71 | 91.09 | 96.11 | 
Conv_lstm model2 | 48.32 | 45.30 | 51.26 | 51.62 | 56.27 | 59.87 | 
Pix2Pix 20 Frame 50 epoch | 40.79 | 113.04 | 119.47 | 121.13 | 123.81 | 117.02 |
Pix2Pix 20 Frame 100 epoch | 18.20 | 115.91 | 119.33 | 119.24 | 126.00 | 124.30 |
Pix2Pix 20 Frame 200 epoch | 9.71 | 117.58 | 130.27 | 128.62 | 122.97 | 135.52 |
Pix2Pix 60 Frame 50 epoch | 108.61 | 120.86 | 128.99 | 126.75 | 128.52 | 124.35 |
Pix2Pix 60 Frame 100 epoch | 100.43 | 112.21 | 125.16 | 126.61 | 109.64 | 109.60 |
Pix2Pix 60 Frame 200 epoch | 98.42 | 106.87 | 126.14 | 120.12 | 121.02 | 120.75 |

If we look at the results for the first frames pix2pix model that trained with 10 frames and for 200 epochs is the most successful but if we look at overall success conv_lstm model 2 is the best one.

While looking at the results we can say the worst ones are pix2pix trained with 50 frames but normally we expected them to be better than pix2pix models that are trained with 10 frames, the purpose of training the model with more frames was to overcome overfitting if there is overfitting but we see that training with more frames didn’t work for this model.

In the table below SSIM values of models we trained with Moving Mnist are given.

| Model | Frame 1 | Frame 2 | Frame 4 | Frame 6 | Frame 8 | Frame 10 |
| --- | --- | --- | --- | --- | --- | --- |
| Conv_lstm model1 | 0.39 | 0.07 | 0.10 | 0.17 | 0.24 | 0.25 |
| Conv_lstm model2 | 0.23 | 0.30 | 0.44 | 0.63 | 0.66 | 0.66 |
| Pix2Pix 20 Frame 50 epoch | 0.92 | 0.62 | 0.57 | 0.56 | 0.53 | 0.53 |
| Pix2Pix 20 Frame 100 epoch | 0.97 | 0.63 | 0.60 | 0.60 | 0.56 | 0.55 |
| Pix2Pix 20 Frame 200 epoch | 0 .98 | 0.61 | 0.53 | 0.55 | 0.55 | 0.49 |
| Pix2Pix 60 Frame 50 epoch | 0.68 | 0.57 | 0.48 | 0.49 | 0.48 | 0.59 |
| Pix2Pix 60 Frame 100 epoch | 0.72 | 0.64 | 0.53 | 0.49 | 0.58 | 0.67 |
| Pix2Pix 60 Frame 200 epoch | 0.74 | 0.68 | 0.56 | 0.61 | 0.59 | 0.62 |

If we look at the results unlike RMSE values pix2pix model trained with 10 frames is better than any other model. The Pix2pix model is visibly way better than the conv_lstm models. The reason might be because SSIM is a similarity metric that compares the images in the human way of looking.

In the figure below examples of predicted frames are given.

![image](https://user-images.githubusercontent.com/46672488/218314455-2a7d3696-2ef2-4b7f-a291-c5c300d0557f.png)

### Circle Dataset
For the circle dataset, we trained 2 conv_lstm models, multiscale gan and pix2pix model with different epochs. In the table below RMSE values of models we trained with the circle dataset that we created are given.
| Model | Frame 1 | Frame 2 | Frame 4 | Frame 6 | Frame 8 | Frame 10 |
| --- | --- | --- | --- | --- | --- | --- |
| Conv_lstm model1 | 33.24 | 38.47 | 62.00 | 80.59 | 92.57 | 102.87 |
| Conv_lstm model2 | 24.04 | 23.96 | 23.41 | 20.98 | 15.10 | 12.38 |
| Multiscale GAN | 18.44 | 21.09 | 29.84 | 37.17 | 43.23 | 47.03 |
| Pix2pix 50 epoch | 9.15 | 12.69 | 21.82 | 29.38 | 37.44 | 43.01 |
| Pix2pix 100 epoch | 8.43 | 9.30 | 17.10 | 24.17 | 29.19 | 35.44 |
| Pix2pix 200 epoch | 4.87 | 9.05 | 21.44 | 31.76 | 43.59 | 48.43 |

If we look at the results for the first frames pix2pix model that trained with 60 frames and for 200 epochs is the most successful but if we look at overall success conv_lstm model 2 is the best one.

While in the pix2pix model in every frame, the accuracy is getting lower but in the conv_lstm model2 accuracy getting higher and when we look at the predicted images we see that it is because in the first frame, the color is more blurry and it gets better with time but the movements are true so if we consider that we can say conv_lstm model2 is better than other models.

In general, pix2pix is good at predicting the first frames.

In the table below SSIM values of the models we trained with the Circle dataset are given.

| Model | Frame 1 | Frame 2 | Frame 4 | Frame 6 | Frame 8 | Frame 10 |
| --- | --- | --- | --- | --- | --- | --- |
| Conv_lstm model1 | 0.37 | 0.34 | 0.31 | 0.43 | 0.43 | 0.38 |
| Conv_lstm model2 | 0.97 | 0.97 | 0.97 | 0.97 | 0.98 | 0.97 |
| Multiscale GAN | 0.97 | 0.96 | 0.95 | 0.95 | 0.95 | 0.95 |
| Pix2pix 50 epoch | 0.98 | 0.97 | 0.95 | 0.94 | 0.94 | 0.94 |
| Pix2pix 100 epoch | 0.99 | 0.98 | 0.97 | 0.95 | 0.96 | 0.96 |
| Pix2pix 200 epoch | 0.99 | 0.99 | 0.98 | 0.97 | 0.95 | 0.94 |

When we look at results other than conv_lstm model1 other models have similarity values that are higher than 0,95 which is quite good.

For conv_lstm model2 the RMSE values were getting better but because SSIM is calculated in gray level it is not the case here.

In the figures below examples of predicted frames of the circle dataset are given.

![image](https://user-images.githubusercontent.com/46672488/218317186-3f4a1df7-7011-4770-8d74-5cbd28503acc.png)

![image](https://user-images.githubusercontent.com/46672488/218317204-b60222ec-219b-4b4f-b2c6-e72a3f686a95.png)

## Result
In our study, we observed that the pix2pix model was more successful in the circle data set than the moving mnist data set.

When we look at the results of conv_lstm model1 and conv_lstm model2 for the circle data set, there is an increase in the success in conv_lstm model2 contrary to expectations. However, when the two models are compared, conv_lstm model2 seems to be more successful. When we compare these two models for the moving mnist dataset, it is seen that although conv_lstm model1 predicts the first predicted frame more successfully, conv_lstm model2 is more successful when looking at the success until the last predicted frame.

We cannot make a comparison for the multiscale gan model as we do with the other models because the prediction results for the moving mnist dataset were not correct. However, it can be said that it is a successful model for the circle data set.

Considering the success of all models tried for the Moving mnist dataset, it can be said that the most successful one is the conv_lstm model2.

Considering the success of all models tried for the Circle dataset, it can be said that the pix2pix model is quite successful in predicting the first frames, while the conv_lstm model2 is more successful in predicting the final frames.
