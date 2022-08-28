# Weather Forecast

### Using time-series data

### Problem Definition

Predict weather forecast based on the date-time provided. Our model provides prediction regarding: <ol>
 <li>Humidity
 <li>Wind Speed
 <li>Mean Temperature
 <li>Mean Pressure </ol>

### Data

Data provided publicly on Kaggle.

### Evaluation 
Trained on k-nearest-neighbors.

Our model will be evaluated on Coefficient of determination (r2_score)metric which is the proportion of the variation in the dependent variable that is predictable from the independent variable.<br>
r2_score will be used to assess the accuracy of our predictions from the date-time column.

### Used Libraries
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_rows', 100)
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings("ignore")
from plotly.subplots import make_subplots
import plotly.graph_objs as go
```

### r2_score for our Trained KNN Model:
```
r2 score for meantemp : 0.788603433535022
r2 score for humidity : 0.19744456248221431
r2 score for wind_speed : 0.0538139649829803
r2 score for meanpressure : 0.5176042636922933
```

### Conclusion
<br>We need to adapt different models to improve our r2 score or experiment with hyperparameter tuning with our current KNN model. 
