# Task-1

In this task we will detect emotions from static facial images. More specfically, we will implement the Xception
architecture for detecting emotions and understand its efficacy as an image recognition pipeline. We will
make use of the Facial Expression Recognition (FER2013) dataset. The dataset consists of 35,685 examples of
48x48 pixel grayscale images of faces. Images are categorized based on the emotion shown in the facial
expressions (happiness, neutral, sadness, anger, surprise, disgust, fear).

# Task-2

In this task we will detect Facial Action Units (FAUs) from static facial images consisting of varying level of emotions. FAUs are specific regions of the face which allow one to estimate the intensity and type of emotion one is experiencing in a given image. Utility of FAUs has seen a tremendous adoption in research and animation as a result of their unique action coding system. Each facial movement in the FAU coding system is encoded by its appearance on the face. For instance, the AU code for *cheek raise* (AU6) and *lip corner pull* (AU12) when combined together give the *happiness* (AU6+12) emotion. As another example, the AU codes *inner brow raise* (AU1), *brow low* (AU4) and *lip corner depress* (AU15) together give the *sadness* (AU1+4+15) emotion. AU codes start from 0 and end at 28 with the 0th code corresponding to *neutral face*. A complete list of AU codes and their corresponding facial expressions can be found on [Wikipedia](https://en.wikipedia.org/wiki/Facial_Action_Coding_System).

We will utilize FAUs in this task to estimate the intensity of facial expressions and try to link them with their corresponding emotions. However, we will do so without making use of a dataset and only use a handful of images (often referred to as few-shot tuning). For this purpose, we will adopt a slightly modified version of the deep ResNet architecture which is pretrained on a FAU dataset.
