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

we didnt just create a Date Of Year feature as representing the day of the year as sine and cosine waves allows the model to understand that December 31st and January 1st are mathematically close, rather than at opposite ends of a linear scale.

also we keep datetime feature even though we've encoded it into date of year, it will come in handy later

also do the encoding for winddir(north is 0* and east is 90* but this doesnt really matter)

Some more feature engineering:
rolling_precip_3day: Sum of daily precipitation over the previous 3 days) \
Accounts for soil moisture feedback. Wet ground evaporates moisture back into the boundary layer, making subsequent days more likely to be cloudy or experience pop-up showers.

diurnal_temp_range(temp_max_daily - temp_min_daily) \
A wide temperature range implies clear skies (sun heats the ground efficiently by day, lack of clouds allows heat to escape efficiently by night). A narrow range strongly suggests persistent overcast conditions.

pressure_change_24h (pressure_daily_today - `pressure_daily_yesterday) \
Captures larger, synoptic-scale waves. Sustained multi-day pressure signals a shift in the air mass.

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

we start lagging from the 4th day in the dataset onwards\
exclude_cols = ['datetime', 'day_of_year_sin', 'day_of_year_cos']\
we exclude time related features from the lagging step(they are just redundant, not sure how to word this)

responsible python script: engineer.py

## 5. Training
we use scikit-learn's random forest classifier and regressor for the weather type and average temperature predictions

random forest cuz its covered in the course material + tree based algos are robust against curse of dimensionality which is even more important with this small sample size

Sorting by time using the datetime column, we designate the first 80% of the dataset to training, last 20% for validation. We do it like this instead of randomly selecting rows from the dataset to ensure that we only validate the model for predicting the future instead of past values.

To ensure that this model acts as a forecaster instead of nowcaster and prevent data leakage, for this step we drop all current day sensor data, forcing the model to only rely on sensor data from the last 3 days.

automated hyperparameter tuning was performed on all pipelines
pca pipeline also has variance threshold tuning, selected threshold was 0.99

Explanations of baseline models:\
Last Observation means predicting temperature tomorrow = temperature today and conditions tomorrow = conditions today\
Majority class means always predicting tomorrow's conditions the same as the majority condition in the dataset

Temperature prediction(regression) results:\
Baseline (Last Observation):\
RMSE: 1.5870\
MAE:  1.2060\
R2:  0.8999

Model MAE: 1.061 \
		Mean Absolute Error of 1.061 means that the average temperature prediction is only 1.061 celsius off the actual temperature. This is practical and usable in real world applications\
Model RMSE: 1.3715 \
		RMSE is basically MAE but penalises bigger prediction errors, this shows our model is very consistent\
Model R Squared: 0.9253 \
		This means our model describes 92.53% of variance in true temperatures, its capturing overall trends and variability well
--> our model can both handle typical temperature changes better than baseline and handle bigger temperature changes(as suggested by the rmse improvement)


Non PCA weather condition prediction:\
Baseline (Majority Class):\
  Accuracy: 0.6700\
  Macro F1: 0.2675

Baseline (Last Observation):\
  Accuracy: 0.7075\
  Macro F1: 0.5844

Model Accuracy: 0.7650\
Model Weighted F1: 0.7485\
Model Macro F1: 0.5705
```text
Classification Report:
              precision    recall  f1-score   support

       Clear       0.58      0.18      0.27        39
      Cloudy       0.52      0.58      0.55        93
        Rain       0.86      0.91      0.89       268

    accuracy                           0.77       400
   macro avg       0.66      0.56      0.57       400
weighted avg       0.75      0.77      0.75       400
```

--> this highlights the fact that our model has learned temporal weather patterns, strongly improving from the majority class baseline but could not meaningfully improve on the persistence model baseline. Compared to persistence model baseline, our model is overall more accurate while exhibiting a small but measurable increase in class imbalance

Weather condition prediction with PCA:\
	Macro F1: 0.5134\
		This matches our hypothesis that PCA wont be of help to our random forest model
		future related work can ignore PCA


also there are some confusion matrices in the same folder as the training script
python script: rf_baseline.py

## Some conclusions(subject to change)
our models have meaningfully learned and improved upon the baselines.\
exception is our classification vs persistence baseline

the temperature model, having improved over the baseline model on all 3 metrics validates our data preparation and training methodology for both models.\
that combined with the fact that visual crossing classifies any day with precipitation > 0 as rain(light 0.2mm precipitation drizzle also counts as rain) suggests that our classification model's inability to improve over the baseline migbt be due to dataset limitations, either temporal granularity/range or especially the lack of spatial/atmospheric context.\
the model is currently blind to oncoming weather events, say a rain band moving towards hanoi or a cold front reaching our city by tomorrow noon, mixing with hot humid air producing heavy rainfall.








