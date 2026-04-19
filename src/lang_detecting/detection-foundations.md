# AI Language Detection

## Project Motivations and Objectives

ScrapLang has been evolving towards ever greater comfort of specifying arguments. With the aim of supplying even greater handiness of the language detection surpassing a mere special-character recognition, a functionality based on a machine learning model has been decided to be implemented.
ScrapLang has already accommodated for the special-character detection and last used language retrival, but the user experience demonstrated that translating between two languages is problematic due to a language misassuming despite the fact that some words are often easily categorizable even for a user not yet much familiar with a language.

## Problematics
The application is able to identify the intended language based on the special characters, so the model will have to take care of the remaining cases where there are no unique characters based on the reliable words previously used by the user. As some strings or similar structures can be encountered in multiple languages, the model should predict many desirably with a lower confidence.

### 1. Inclusion of Unique Characters 
The application's database ought to be continually growing and so some characters formerly deemed unique might appear in the model's input. A question of a similar nature is that the input dataset deprived of samples featuring special characters is destined to be underrepresentative and to impair the model. So to make full use of the data and to make it resilient to new characters, the model ought to be trained on data with the special-characters-featuring samples masked appropriately. To make the model further resilient and avoid shortcutting the learning by relying on the token representing masked characters, the data ought to be augmented for this token to appear randomly in places where the languages do not necessarily feature their respective special characters. 

### 2. Label Imbalance in Multilabel Classification

Real-life experience proved that a large imbalance between labels is expected to arise. An imbalance of positive and negative classes is also expected. In both cases this should be weighted to make the model's predictions reliable both for the large and smaller classes. Other methods should be undertaken including redesigning the model to make it possible to do well in all the cases. Another case of imbalance is that which arises from the samples being true multi-label, so those that have more than one correct label. They will also necessarily favour the dominant languages. This signifies that the model should not be overdiscouraged from assigning some predictions to words that look like a specific language, but are not annotated as such. It should focus much more on assigning greater certainty for the records it has than on punishing the falsely negative labels. 
