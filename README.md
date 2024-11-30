# ML-Model-Flight-price-Prediction
Business Understanding
Objective:
The primary business goal is to understand the factors influencing airline ticket prices. Airlines,
travel agencies, and customers benefit from optimized pricing strategies. Airlines can maximize
profits and seat occupancy, while customers can book flights at the best possible prices. The focus
of this project is to build a predictive model that forecasts airline ticket prices based on several
attributes, such as the airline, source and destination cities, number of stops, flight duration, class
of travel, and time before departure.
Key Questions:
• What are the main factors that influence airline ticket prices?
• Can we predict the price of an airline ticket based on certain features?
• How can airlines or travel agencies optimize their pricing strategies using these
predictions?
The primary deliverable is a robust predictive model that airlines and travel platforms can use to
forecast ticket prices in real-time.
Data Understanding
The dataset contains 300,153 rows and 12 columns related to airline ticket prices. The features
include:
• Categorical Variables: airline, source_city, destination_city, departure_time,
arrival_time, stops, class.
• Numerical Variables: duration (flight duration in hours), days_left (days between
booking and departure), price (target variable).
Initial Observations:
• The dataset is well-organized, with no missing values.
• Categorical Variables: Some columns, like departure_time and arrival_time, are
considered categorical but may reflect time-based effects (e.g., night vs. morning flights).
• Numerical Variables: duration, days_left, and price need to be properly explored to
understand their distributions.
2
Data Preparation
Transformations Applied:
• Encoding Categorical Variables: Categorical variables like airline, source_city, and
class were one-hot encoded.
• Time-Based Features: departure_time and arrival_time were categorized into time
intervals (e.g., morning, afternoon, evening) to account for time-based pricing effects.
• Scaling: Numerical features (duration, days_left) were scaled using Standard Scaler for
better model performance.
Outlier Treatment:
• Outliers were particularly identified in price and duration, mainly representing premium
classes and long-haul flights, respectively. These were either handled or retained based on
whether they represented significant business cases (e.g., premium pricing).
The dataset was prepped for modeling, with balanced, encoded, and scaled data ready for
machine learning algorithms.
Modelling
Several machine learning models were trained to predict the target variable (price). These
include:
• Linear Regression
• Ridge Regression
• Lasso Regression
• Elastic Net Regression
• Random Forest
• KNN Classification
• Time Series Forecasting (Holt-Winters)
Data Split:
• The dataset was split into 80% training and 20% testing sets. Cross-validation was used
to ensure the models generalize well on unseen data.
3
Evaluation
The following metrics were used to evaluate model performance:
• Root Mean Squared Error (RMSE)
• Mean Absolute Error (MAE)
• R-squared (R²)
Regression Model Evaluation:
• Linear Regression & Ridge Regression: Both models performed exceptionally well, with
an R² of 0.911 and low RMSE (6768), meaning 91% of the price variability was explained
by the model.
• Lasso Regression: Performed similarly to Ridge but with slightly higher RMSE.
• Elastic Net Regression: Underperformed with high RMSE and low R² (0.511).
Classification Models:
• KNN Classification: Achieved a moderate accuracy of 57.83%, indicating limited
performance in classifying ticket classes.
• Random Forest Classifier: This model struggled with an accuracy of only 51.2%,
suggesting further feature engineering or rebalancing of the class distribution is required.
Time-Series Forecasting:
• Holt-Winters Model: Poor performance with a highly negative R² (-16.42), suggesting
that this model was not appropriate for forecasting price trends in this context.
Deployment
Once the best model (e.g., Ridge or Linear Regression) has been selected, it can be integrated
into a travel application or system. The model can be used to:
• Predict real-time airline ticket prices for customers based on their input criteria (e.g.,
airline, travel date, number of stops, and class).
• Allow airlines to optimize their pricing strategies dynamically by adjusting fares based on
booking lead time, travel class, and demand.
4
The deployment stage may involve integrating the model into an API, providing a user-friendly
interface for end-users (travelers, airlines).
5
Exploratory Data Analysis
The exploratory data analysis (EDA) focuses on three critical numerical variables: duration,
days_left, and price.
The distribution of duration reveals that most flights last under 10 hours, with a prominent peak
between 5 and 10 hours. Longer flights are rare, as indicated by the tail of the distribution. The
kernel density estimation (KDE) line suggests a multi-modal distribution, potentially reflecting
different types of flights, such as short-haul and long-haul.
In the case of days_left, the distribution appears relatively uniform, with a slight increase between
10 to 40 days. A sharp decline is evident after 40 days, indicating fewer tickets booked more than
40 days before departure. The KDE line shows minor fluctuations, possibly due to pricing
strategies or demand changes based on the proximity of the flight date.
The price distribution is skewed to the right, with most prices concentrated in the lower range. A
sharp spike is observed in the ₹0 to ₹20,000 range, with a secondary peak between ₹50,000 and
₹60,000. The KDE line suggests significant price variation, which may correspond to different
ticket types, such as economy and business class.
6
-To detect outliers
From the boxplots, it can be concluded that most flight durations are under 30 hours, with long-
haul flights contributing to outliers. Booking times are evenly distributed between 15 and 40 days
before departure, with no significant outliers. Ticket prices, however, display a highly skewed
distribution, with most falling between ₹10,000 and ₹40,000. The numerous price outliers, often
above ₹100,000, likely represent premium or last-minute bookings. These visualizations reinforce
insights from the earlier histograms.
7
Relationship between price and other factors
1. Price vs Airline
The "Price vs Airline" boxplot reveals distinct price distributions across different airlines. Among
the airlines, Vistara and Air India show significantly higher price variability, with Vistara having
the highest maximum fares. Vistara's ticket prices have a wide range, including higher-priced
outliers, indicating premium services or first-class travel. In contrast, Indigo, SpiceJet, and
AirAsia have much lower ticket prices, with smaller interquartile ranges, suggesting they are
budget carriers with relatively lower and more consistent pricing. The plot indicates that the choice
of airline is a major factor affecting the price, with legacy and premium airlines like Vistara charging
significantly more than low-cost carriers like SpiceJet.
8
2. Price vs Source City
The "Price vs Source City" plot shows that Delhi, Mumbai, and Bangalore are source cities
where ticket prices exhibit significant variability. Mumbai has some of the highest prices, while
Chennai and Hyderabad have relatively lower prices and smaller price ranges. This suggests
that the origin city plays a considerable role in influencing ticket prices, possibly due to demand
variations, airport taxes, or the prominence of certain cities as international or domestic hubs. The
presence of more expensive flights originating from cities like Delhi and Mumbai may also reflect
higher demand or more frequent long-haul flights from these locations.
3. Price vs Destination City
Similarly, the "Price vs Destination City" plot displays significant variability in ticket prices based
on the flight's destination. Delhi and Mumbai are destinations with wider price ranges, while
Kolkata and Chennai have more moderate and consistent prices. These observations suggest
that certain destinations, likely due to their status as major metropolitan areas or business hubs,
may experience higher and more varied demand, influencing ticket prices. The size of the
interquartile ranges indicates that destinations like Delhi may have both budget and premium
travel options available, whereas cities like Hyderabad have a more constrained price range.
4. Price vs Departure Time
The "Price vs Departure Time" plot highlights that flight prices tend to vary based on the time of
departure. Night flights exhibit the highest variability in price, with several outliers indicating
higher-priced tickets. Late-night flights, on the other hand, have the lowest prices and variability,
suggesting these flights are often discounted or fall into off-peak periods. Morning, afternoon,
and evening flights show moderate price distributions. The variation in price according to
departure time may be related to factors such as convenience, demand, and time preferences of
travelers, with night flights potentially catering to business travelers willing to pay higher premiums
for flexible schedules.
5. Price vs Stops
The "Price vs Stops" plot shows a stark contrast in ticket prices based on the number of stops in
a flight. Non-stop flights have lower median prices, but there are more expensive options
available, particularly in premium airlines. Flights with one stop have a much wider range of
prices and a higher median, suggesting that adding a stop can significantly impact the price, likely
due to longer travel times or layover costs. Flights with two or more stops are consistently
cheaper, suggesting that passengers are often willing to pay less for the inconvenience of multiple
layovers. The number of stops appears to be a significant factor in influencing price, with direct
flights generally being more expensive but desirable for convenience.
6. Price vs Class
The "Price vs Class" plot clearly demonstrates that ticket prices in Business class are
substantially higher than those in Economy class. The boxplot shows a wide range of prices for
Business class, with numerous outliers representing very high fares. Conversely, Economy class
prices are concentrated within a much smaller range, with lower overall ticket prices and fewer
outliers. This disparity is expected, as Business class offers more premium services and
9
amenities, justifying the significantly higher prices. This finding highlights that class of service is
the most significant factor influencing ticket price, with Business class fares being considerably
higher across the board.
Correlation of features with Target Variable Price
10
The correlation heatmap reveals that class has the strongest negative correlation with price (-0.94),
indicating that the price rises significantly as the travel class increases (e.g., business or first class).
Other factors like stops (-0.2) and duration (-0.14) also exhibit negative correlations, implying that
flights with fewer stops or shorter durations tend to have higher prices. In contrast, features like
airline (0.24) and flight (0.3) show moderate positive correlations with price, meaning these
attributes influence price but to a lesser degree. The bar chart further highlights these insights, with
class being the most dominant factor affecting price, followed by variables like airline, flight, and
duration. In contrast, others, like arrival_time and source_city, have minimal impact.
11
Principal Component Analysis (PCA)
The Principal Component Analysis (PCA) conducted on the dataset reveals insightful patterns in
relation to airline pricing and the underlying features, specifically flight duration, days left until
departure, and ticket price. This analysis serves as a crucial step in reducing the dimensionality of
the data while retaining the essential variance necessary for making accurate predictions about
airline prices.
Component Importance
The first principal component accounts for slightly over 40% of the total variance, which is a
substantial proportion. This finding suggests that a significant portion of the variability in the
dataset—encompassing ticket price, flight duration, and days left until departure—can be
attributed to this single component. Essentially, this first principal component captures the most
influential relationships and patterns within the dataset, making it an essential feature for
understanding the underlying structure of the data. The significance of this component implies that
it may represent a key factor influencing airline ticket prices, likely involving a combination of the
two predictors: days left until departure and flight duration.
Reduction in Variance Across Components
Following the first principal component, the second and third components explain progressively
less variance in the data, with the second accounting for around 30% and the third about 25%. This
diminishing contribution suggests that while these components remain important, their individual
contributions to explaining variability are less significant than the first component. Together, the
first three components explain a cumulative variance of nearly 100%, implying that the majority
12
of the dataset’s variability is captured by these three principal components.
Relationships Captured by PCA
The relationships between the key variables, such as flight duration, days left until departure, and
ticket price, are likely reflected in the principal components. The first principal component likely
captures the combined effect of days left and duration on price. Specifically, as tickets purchased
closer to the departure date are generally more expensive, and longer flights are typically priced
higher, these two variables contribute significantly to price variability. The second and third
components further refine these relationships, capturing more specific variations that are not
entirely explained by the first component. However, their reduced importance, as indicated by their
lower explained variance, means that they contribute less to the overall model's predictive power.
Cumulative Variance Growth
The cumulative variance explained by each successive component illustrates a clear pattern of
diminishing returns. While the first component explains about 40% of the variance, adding the
second raises the cumulative explained variance to over 70%. By the time the third component is
added, nearly 100% of the variance is explained. This pattern confirms that the majority of the
critical information is contained within the first few components, further supporting the decision
to reduce dimensionality for future modeling efforts.
Actionable Insights for Airline Pricing Models
The PCA results indicate that focusing on a few key variables—particularly days left and flight
duration—will yield robust models for predicting airline prices. By reducing the dataset to three
principal components, predictive models can be developed that are both efficient and effective.
This dimensionality reduction will streamline the process of building models, reduce
computational costs, and improve interpretability while maintaining high predictive accuracy. PCA
thus proves to be an invaluable tool for capturing the essential patterns in the dataset, making it
highly suitable for the airline pricing context.
In conclusion, PCA has demonstrated its utility in identifying the key factors that drive variance
in the dataset. The principal components derived from the analysis effectively capture the
relationships between days left, flight duration, and price, thereby providing a foundation for
dimensionality reduction without sacrificing predictive quality.
13
Analysis and Comparison of Regression Models for Airline Price Prediction
In this analysis, various regression models were applied to predict price values based on a set of
categorical and numerical features. The features included categorical variables such as 'airline,'
'source_city,' 'destination_city,' 'departure_time,' 'stops,' and 'class,' while numerical variables
included 'duration' and 'days_left.' The code processed these features using a combination of One-
Hot-Encoder for categorical variables and Standard-Scaler for numerical variables.
The dataset was split into training and testing sets, using 80% of the data for training and the
remaining 20% for testing. A Column-Transformer was employed to preprocess the data by
standardizing numerical features and encoding categorical ones. The preprocessed data was then
passed through a series of regression models: Linear Regression, Ridge Regression, Lasso
Regression, and Elastic Net.
1. Preprocessing:
• The dataset was first split into feature variables (X) and target variable (y), with price
being the target.
• Categorical and numerical variables were preprocessed using One-Hot-Encoder and
Standard-Scaler respectively within a Column--Transformer.
2. Model Training:
• Linear Regression, Ridge Regression, Lasso Regression, and Elastic Net models
were created using a Pipeline that combined preprocessing and the regression model.
• Each model was trained on the training set and subsequently evaluated using the testing
set.
Image: Linear Regression
14
Image: Ridge Regression
Image: Lasso Regression
Image: Elastic Net Regression
15
3. Evaluation Metrics:
• Predictions were made for both the training and testing sets using each model.
• Root Mean Squared Error (RMSE) and R² scores were computed to evaluate model
performance.
4. Conclusions:
• Both Linear Regression and Ridge Regression models performed exceptionally well,
with the lowest RMSE values for both training and testing datasets, approximately
6768. This indicates that their predictions closely match the actual values, making them
highly reliable. Both models also reported the highest R² values of 0.911, meaning that
91.1% of the variance in the price data is explained by these models. The small
confidence interval (CI) range difference between training and testing data
(approximately 3737 to 3739) further highlights the consistency and accuracy of these
models.
• The Lasso Regression model performed similarly, with slightly higher RMSE values
(6786) but maintained the same high R² value of 0.911. The slight increase in the CI
range difference (3752) suggests marginally higher variance but still demonstrates
strong model performance.
• In contrast, the Elastic Net model significantly underperformed, with much higher
RMSE values (15860 for training and 15889 for testing). Its R² values were much
lower, at approximately 0.511, meaning that only 51.1% of the variance in the data is
explained by the model. The large CI range difference (57186) indicates substantial
variability in its predictions, making it less suitable for this dataset without further
tuning.
Image: Different Model Evaluation Metrics
16
Classification
The model in this task is focused on classifying airline ticket classes as either "Business" or
"Economy" based on various features using a Random-Forest-Classifier. The code follows a
structured pipeline approach to preprocess the data and train the classification model. The
pipeline combines feature preprocessing and classification in an integrated manner, making the
workflow both streamlined and reproducible.
1. Feature Selection and Target Variable:
• The features selected for this classification task include airline, source city, destination
city, departure time, stops, duration, and days left. These variables are expected to
influence the classification of tickets into either "Business" or "Economy" class.
• The target variable is the class of the ticket, represented as "class," which holds two
values: Business and Economy. The exclusion of "price" ensures that the classification
model is trained based solely on features that are typically available at the time of
booking.
2. Data Splitting:
• The data is split into training (80%) and testing (20%) sets using train_test_split. This
ensures that the model is trained on one portion of the dataset and evaluated on a different
set of unseen data, reducing the risk of overfitting and providing a more accurate
assessment of the model's generalization performance.
3. Preprocessing Steps:
• The Column Transformer is used to handle both categorical and numerical features. For
numerical features like duration and days_left, a Standard-Scaler is applied to normalize
the data. For categorical features like airline, source_city, and stops, a One-Hot-Encoder
is used to convert these categories into dummy variables. This ensures that categorical
variables are properly encoded for the machine learning model, and numerical features
are standardized, preventing features with larger ranges from dominating the model.
4. Model Training and Prediction:
• A Random-Forest-Classifier is applied after preprocessing, utilizing the ensemble
learning approach to classify the ticket classes. Random forests use multiple decision
trees and aggregate their predictions to improve accuracy and prevent overfitting.
17
• After training on the processed dataset, the model makes predictions on the test set,
providing results for evaluation.
Image: Different Model Evaluation Metrics
5. Model Performance Evaluation:
• The current model's accuracy of 51.2% and the unbalanced performance between classes
suggest that the existing feature set is insufficient for accurately classifying airline ticket
classes. To enhance the model's ability to differentiate between Business and Economy
tickets, additional feature engineering should be considered.
• Addressing the class imbalance, where Economy tickets greatly outnumber Business
tickets, is essential. Techniques such as oversampling the minority class (Business) or
under sampling the majority class (Economy) could help improve model performance.
• Additionally, exploring other classification algorithms like gradient boosting or support
vector machines may offer better results compared to the current RandomForest
approach. These methods may handle class imbalance more effectively and provide
improved accuracy.
18
• Incorporating more detailed features, such as booking behavior or ticket flexibility, may
also contribute to better class distinction, enhancing the model's predictive power.
Image: Model Evaluation Metrics (Random Forest)
19
Time Series Forecasting
1. Data Preparation:
o The data was grouped by the days_left variable, and the mean prices were
calculated to create a price_by_time series. This approach assumes a time-related
change in prices.
2. Train-Test Split:
o The data was divided into a train (80%) and test (20%) set based on the
chronological order of prices. The training data is used to fit the model, and the test
data is used for evaluation.
3. Model Application:
o The Exponential Smoothing (Holt-Winters method) was applied, accounting for
both trend and seasonality with an additive model. The model was fitted on the
training data to forecast the test period.
20
4. Forecasting and Visualization:
o The forecasted values were compared against the actual test data, and a visual plot
was generated, showing the train, test, and forecast series.
5. Forecasting Accuracy:
o An R² score of -16.42 was calculated to assess forecast accuracy. A negative R²
score suggested that the model performs poorly and does not accurately predict
price trends.
Conclusions:
The model’s forecasting accuracy is poor, as indicated by the highly negative R² score (-16.42).
This suggests that the Holt-Winters Exponential Smoothing model did not capture the price
trend effectively, likely due to the lack of actual continuous time data and potential seasonality in
the dataset. The negative score indicates that the forecast was worse than a simple mean prediction
of the test data, implying the need for a better model fit or more appropriate time-related data.
21
KNN Classification
In this analysis, the K-Nearest Neighbors (KNN) classification algorithm was implemented to
predict the target variable using a pipeline for data preprocessing and classification. The pipeline
was constructed with the following steps:
1. Data Preprocessing:
• The preprocessor handles both numerical and categorical variables using:
▪ Standard-Scaler for scaling numerical columns.
▪ One-Hot-Encoder for transforming categorical columns.
2. Model Training:
• The KNN classifier was applied with 5 neighbors (n_neighbors=5). The model
was trained on the training dataset using the transformed data.
3. Prediction:
o After training, the model was used to make predictions on the test dataset.
22
4. Model Evaluation:
o The accuracy score of the KNN classifier was calculated to measure its
performance. The reported accuracy was 57.83%.
Conclusion:
The KNN classifier achieved an accuracy of 57.83% on the test data, which indicates a moderate
level of performance. The accuracy suggests that the model correctly classified approximately
58% of the test instances. Given this result, the KNN model demonstrates some predictive
capability but may not be highly reliable for this particular dataset.
Further refinement may be needed to improve performance. This could involve experimenting
with different values of K (n_neighbors), tuning hyperparameters, or trying other classification
algorithms that might be better suited to the dataset’s characteristics. Additionally, improving
feature selection or scaling the data differently may contribute to better acc
