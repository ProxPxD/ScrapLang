# AI Language Detection

## Project Motivations and Objectives

ScrapLang has been evolving towards ever greater comfort of specifying arguments. With the aim of supplying even greater handiness of the language detection surpassing a mere special-character recognition, a functionality based on a machine learning model has been decided to be implemented.
ScrapLang has already accommodated for the special-character detection and last used language retrival, but the user experience demonstrated that translating between two languages is problematic due to a language misassuming despite the fact that some words are often easily categorizable even for a user not yet much familiar with a language.
The model is not intended to be overtly large.

The project is not yet terminated. The results are unsatisfactory, but it has been a great support in refreshing and expanding the knowledge and to practice solving real-scenario problems in a non-artificial environment.

## Data-Related Problematics
The application is able to identify the intended language based on the special characters, so the model will have to take care of the remaining cases where there are no unique characters based on the words previously entered and successfully scrapped by the user. As some strings or extracted underlying features can be encountered in multiple languages, the model should be able to predict many labels, desirably with a lower confidence.  

### 1. Initial Selection
As the data is gathered automatically the scope of selection data for training is limited. The application saves the examples for which more than one translation is possible to early filter the samples towards more representative ones. Then in the data processing only data of high enough length and high enough non-unique characters proportion are selected for training.

### 2. Inclusion of Unique Characters 
The application's database ought to be continually growing and so some characters formerly deemed unique might appear in the model's input. A question of a similar nature is that the input dataset deprived of samples featuring special characters is destined to be underrepresentative and to impair the model. So to make full use of the data and to make it resilient to new characters, the model ought to be trained on data with the special-characters-featuring samples masked appropriately. To make the model further resilient and avoid shortcutting the learning by relying on the token representing masked characters, the data ought to be augmented for this token to appear randomly in places where the languages do not necessarily feature their respective special characters. 

### 3. Label Imbalance in Multilabel Classification
Real-life experience proved that a large imbalance between labels is expected to arise. An imbalance of positive and negative classes is also expected. In both cases this should be weighted to make the model's predictions reliable both for the large and smaller classes. Other methods should be undertaken including redesigning the model to make it possible to do well in all the cases. Another case of imbalance is the one which arises from the samples being truly multi-label, i.e., those that have more than one correct label. They will also necessarily favour the dominant languages. This signifies that the model should not be overdiscouraged from assigning some predictions to words that look like a specific language, but are not annotated as such. It should focus much more on assigning greater certainty for the positive records than on punishing the falsely negative labels. 

## Model Architecture Overview
The model consists of Experts which are dedicated to scripts and equipped with masks preventing their predictions from spilling into foreign domains. An expert is created exclusively when more than one language are employing the same script. 
The initial and core architecture of an expert was a three-layer CNN with an FNN, later extended with an attention module in between in order to capture more distant relations. The features are pooled down before being passed to the FNN module.

## Experimentation
The project’s architecture is experimental and exploratory. Numerous runs have been executed to compare the impact of various regularization techniques and their placements. Different strategies of connecting feature flows have also been tested, including the use of the attention module as a gate versus a residual, as well as rewriting the CNN module to function as a residual component.

At one stage, a mask was introduced, which significantly improved the model’s ability to focus on the relevant segments. However, in some cases, the mask degraded performance; this issue was mitigated through additional regularization. Applying the mask directly to the attention mechanism, however, irreparably worsened the performance that could not be resolved neither through further regularization nor architectural modifications. This aspect requires further investigation.

Another noteworthy observation is that even minor changes in the distribution of training and validation datasets can lead to substantial shifts in model performance, despite using a fixed random seed. Since this application continuously collects data, freezing the dataset during the development was not considered. This finding is valuable for evaluating the model’s performance, learning dynamics, and overall stability. Improving the quality of supervision has therefore become a key objective.

## Roadmap
- Investigate why padding mask does not yield better results in the attention mechanism
- Having worked out the final classification layer and earlier pooling, there is a suspicion that they can be improved
- Explore whether normalization placements can be improved
- Experiment with the loss to test whether better results can be drawn if model is assigned a small penalty for some certainty in wrong classes - as long as it improves the prediction of the correct ones
- Experiment with soft loss


