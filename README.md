Goal: create a weather prediction model that provides actionable daily intelligence, translating complex atmospheric shifts into simple, decision-ready categories: Rain, Cloudy, and Clear.

This is the first iteration
Overall flow: merger -> cleaner -> engineer -> rf_baseline

Some notes about this iteration: 
	The 3 pipelines(non PCA classification and regression, PCA classification, or in other words, non PCA weather condition prediction and temperature prediction, PCA weather condition prediction) here are considered finalised.
	This iteration's 3 pipelines should receive no more architectural changes, only fixes and additional results gathering and data analysis are expected

## 1. Merge dataset files
we've got 2 dataset files from visualcrossing, we need to merge them into 1 file
code responsible for this step: merger.py

## 2. Initial data cleaning
remove columns from the dataset that we definitely wont need like snow, snowdepth, location name, etc
filled missing values in the preciptype column with noprecip
removed all features with 0 variance as per the slides(there were no such features)

responsible python script: cleaner.py

## 3. Feature engineering
From date and time we created 2 features day_cosine and day_sine;
day_cosine: 1st day starts at 1
day_sine: 1st day starts at 0
we need both since with just cosine, its value is the same for both 0 at the end of spring and start of winter for example, which will be difficult for the model to differentiate

we didnt just create a Date Of Year feature as representing day of year as cosine and sine values here preserve the cyclical nature of dates and time(december 30th being very close to january 1st)

also we keep datetime feature even though we've encoded it into date of year, it will come in handy later

also do the encoding for winddir(north is 0* and east is 90* but this doesnt really matter)

consolidated:
- Original: Rain ➔ Rain
    
- Original: Rain, Overcast ➔ Rain
    
- Original: Rain, Partially cloudy ➔ Rain

- Original: `Overcast` ➔ **Cloudy**
    
- Original: `Partially cloudy` ➔ **Cloudy**

We consolidate the rain types into 1 type mainly to align the model with helping end users on whether to bring an umbrella or plan their transportation accordingly

The consolidation of cloudy types of weather is mainly to increase statistical significance of the samples and reduce class imbalance vs the Rain type
The original dataset exhibited extreme minority classes, specifically "Overcast" with only 59 samples. Consolidating "Cloudy" types provides the algorithm with enough support to identify reliable patterns rather than being forced to ignore rare events to optimize global accuracy.

Rain without enough clouds to even trigger the "partially cloudy" is also problematic, it only happens once every 1 or 2 years which is too rare for a model learn considering that the sample size is only about 5.5 years, 


proof of class imbalances mentioned above:
conditions
Rain, Partially cloudy    1065
Partially cloudy           447
Rain, Overcast             317
Clear                      106
Overcast                    59
Rain                         5

We dont do data normalisation as random forests are scale invariant, they break feature space down to 1D questions

Now we split the pipeline into 3 flows: classification and regression non PCA, classification with PCA

classification and regression non PCA:
	1. We check if 2 features are over 90% correlated, we drop the one that has a lower Mutual Information with the respective target(temp for regression flow and conditions for classification flow). We use MI for this final check instead of another correlation check as weather features often have non-linear relationships with the condition
	2. Perform lagging(details below)

for classification with PCA, we only perform lagging and not the correlation check. As for why it was done this way:
	1. mr Quang did not mention how to integrate correlation checks(filter feature selection methods) with PCA
	2. It felt counterproductive 
		- Correlation check is used to drop features that are highly correlated
		- PCA collapses highly correlated features into 1 Principal Component also to reduce dimensionality

We separated non PCA and PCA because of random forests's nature, it seems like it would be robust against the curse of dimensionality, we try pca vs non pca to verify this(we should explain further random forest's nature and how it suggests robustness against curse of dimensionality)

responsible python file: engineer.py
## 4. Lagging
we use a 3 day lag, collapsing 4 rows of daily data into 1, with data from that row's date + data from 3 previous days

we start lagging from the 4th day in the dataset onwards
exclude_cols = ['datetime', 'day_of_year_sin', 'day_of_year_cos']
we exclude time related features from the lagging step(they are just redundant, not sure how to word this)

responsible python script: engineer.py

## 5. Training
we use scikit-learn's random forest classifier and regressor for the weather type and average temperature predictions

random forest cuz its covered in the course material + tree based algos are robust against curse of dimensionality which is even more important with this small sample size

Sorting by time using the datetime column, we designate the first 80% of the dataset to training, last 20% for validation. We do it like this instead of randomly selecting rows from the dataset to ensure that we only validate the model for predicting the future instead of past values.

To ensure that this model acts as a forecaster instead of nowcaster and prevent data leakage, for this step we drop all current day sensor data, forcing the model to only rely on sensor data from the last 3 days.

automated hyperparameter tuning was performed on all pipelines
pca pipeline also has variance threshold tuning, selected threshold was 0.99

Temperature prediction(regression) results:
	- MAE: 1.07
		Mean Absolute Error of 1.07 means that the average temperature prediction is only 1.07 celsius off the actual temperature. This is practical and usable in real world applications
	- RMSE: 1.3982
		RMSE is basically MAE but penalises bigger prediction errors, this shows our model is very consistent
	- R Squared: 0.9223
		This means our model describes 92.23% of variance in true temperatures, its capturing overall trends and variability well

Non PCA weather condition prediction:
	Accuracy: 0.7575
	Weighted F1: 0.7427
	Macro F1: 0.5663
	t luoi gthich phan nay ngl

Weather condition prediction with PCA:
	Macro F1: 0.5199
		This matches our hypothesis that PCA wont be of help to our random forest model
		Future related work can ignore PCA


also there are some confusion matrices in the same folder as the training script
python script: rf_baseline.py







