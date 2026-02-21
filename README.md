## Iteration 2
In iteration 1 we suggested that either we lacked temporal granularity in our data or spatial information. To further investigate this, we decided to do a second iteration of our models with more granular data. Additional spatial was not feasible given API cost constraints and the added complexity in pre-processing compared to our time budget(dk if we should include this part, im adding it just in case)

This iterations adds sensor data of 10 11PM and 12AM of each day, on top of the daily weather data used in iteration 1

This has allowed us to do some more feature engineering to highlight the transition period from today to tomorrow


- **`pressure_trend_overnight`** (`pressure_12AM` - `pressure_10PM`)
    
    - **Justification:** This measures pressure tendency. A rapid pressure drop in this 2-hour window is a strong indicator of an approaching low-pressure system or cold front, significantly increasing the probability of "Rain" or "Cloudy" tomorrow. Rising pressure indicates subsidence (sinking air) and clears out clouds, pointing toward "Clear."
                


- **`dewpoint_depression_12AM`** (`temp_12AM` - `dew_12AM`)
    
    - **Justification:** The smaller this number, the closer the air is to 100% relative humidity. A depression nearing 0 at midnight strongly suggests the boundary layer is saturating. This often leads to morning fog or stratus clouds, dictating a "Cloudy" or "Rain" start to the next day.
- **`humidity_momentum`** (`humidity_12AM` - `humidity_10PM`)
    
    - **Justification:** If relative humidity is spiking late at night independent of temperature drops, moisture advection is occurring (moist air is blowing in). This is a strong precursor to precipitation.

        
- **`wind_shift_overnight`** (Absolute angular difference between `winddir_10PM` and `winddir_12AM`)
    
    - **Justification:** A sudden shift in wind direction (e.g., > 45 degrees) within two hours is the hallmark signature of a frontal passage. If a front passes at midnight, tomorrow's weather will be completely different from today's.
- same for 12AM and t-1


- **`cloud_cover_trend`** (`cloudcover_12AM` - `cloudcover_10PM`)
    
    - **Justification:** Nocturnal cloud development. If the sky is rapidly clouding over between 10 PM and midnight, it traps outgoing longwave radiation and indicates incoming mid-level moisture, usually preceding a rainy or overcast day.
        


so after all the data processing steps, our data has:
today's data + tomorrow's 12AM + engineered features from iteration 1 + more granular engineered features enabled by more data in this iteration

--- Regression Results ---\
MAE: 0.9651\
RMSE: 1.2741\
R-squared: 0.9356

Measurable improvements over the iteration 1 models(and furthering improving upon the baseline model) can be explained by: 

Late-evening snapshots (10–12 PM) capture cooling trends and short-term inertia\
The model learns slope + level, not just yesterday’s level

--- Classification Results ---\
Accuracy: 0.7318\
Weighted F1: 0.7360\
Macro F1: 0.6010

Accuracy dropped and macro f1 improved, both just slightly. The model became fairer across minority classes but worse at raw hit rate.


Even with the significant increase in temporal granularity of our data, our regression model only improved modestly and the classification model only exhibited an improvement in predictive behavior and not predictive power. This strengthens our suspicion/hypothesis that the lack of spatial data is our main limitation. Therefore future work should incorporate both temporal and spatial data to better describe the overall atmospheric context.
