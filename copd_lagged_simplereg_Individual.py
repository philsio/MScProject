import numpy as np
import pandas as pd
import GPy
import pylab as pl
import matplotlib.pyplot as plt
import pickle

print('Importing .csv Data...')
# First we Import the .csv Data Files
patient_data = pd.read_csv('data/patients_area_a_Admission.csv', delimiter=',', header=0)
pollutant_data = pd.read_csv('data/air_pollution_area_a.csv', delimiter=',', header=0)
print('Done.')

print('Preprocessing Data...')
total_patients = patient_data.shape[0]

# Hand-Set since the .csv file is a bit messy and was extracting more than actually available
total_pollutant_measurements = 413

# Define a list of the applicable dates, so we can then match dates to indices
dates = pd.date_range(start='2014-12-1', end='2016-1-31', freq='D')

total_dates = dates.shape[0]

patient_data = (patient_data[patient_data['MEDICAL_ORG_CODE'] == '45090340X'])
current_total_patients = patient_data.shape[0]

# Count the number of patients admitted per day
patients_per_day = np.zeros(total_dates)
for patient in range(0, current_total_patients):
    admission_date = patient_data.iat[patient, 14]
    if admission_date in dates:
        date_index = dates.get_loc(admission_date)
        patients_per_day[date_index] += 1


pollutant_measurements = np.zeros([total_dates, pollutant_data.shape[1]-1])

### Add a dummy column that points to the relevant dates (date index approach):
dates_range_indices = np.asarray(list(range(1, total_dates+1)))
dates_range_indices = np.expand_dims(dates_range_indices, axis=1)

pollutant_measurements = np.concatenate((dates_range_indices, pollutant_measurements), axis=1)

for measurement in range(0, total_pollutant_measurements):
    measurement_date = pollutant_data.iat[measurement, 0]
    date_index = dates.get_loc(measurement_date)
    for element in range(1, 9):
        pollutant_measurements[date_index, element] = pollutant_data.iat[measurement, element]

# Cover missing values using the previous available measurement (used in lagged variable approach):
for measurement in range(1, total_dates):
    if (pollutant_measurements[measurement, :] == np.zeros(pollutant_data.shape[1]-1)):
        pollutant_measurements[measurement, :] = pollutant_measurements[measurement - 1, :]
# Finally, get rid of NaN values
all_measurements = np.nan_to_num(pollutant_measurements)


# Normalisation of the pollutant data
input_variables_num = all_measurements.shape[1]
for variable in range(1, input_variables_num):
    temp_column = all_measurements[:, variable]
    current_mean = np.mean(temp_column)
    current_std = np.std(temp_column) + 1e-10
    temp_column = (temp_column - current_mean) / current_std
    all_measurements[:, variable] = temp_column

# Normalisation of the Patient Count data:
## Important to store necessary information to invert at the end
temp_column = patients_per_day
patient_mean = np.mean(temp_column)
patient_std = np.std(temp_column) + 1e-10
temp_column = (temp_column - patient_mean) / patient_std
patients_per_day = temp_column

# expand patients per day array dimension to match with measurements:
patients_per_day = np.expand_dims(patients_per_day, axis=1)

### Columns that do not contain useful information (as seen from the plots)
# 'Pressure' : Only a few, sparse spikes
# 'Wind' : Flat-line
all_measurements = np.delete(all_measurements, [7, 9], axis=1)

print('Done.')

########################################################################################
# Creation of lagged dataset
LAGS = 21 # number of lagged variables, i.e. previous days included as input variables

# Extract only patient count and PM2.5: these are the variables we are interested in
pm_25 = np.expand_dims(all_measurements[:, 6], axis=1)
lagged_dataset = np.concatenate((patients_per_day, pm_25), axis=1)
temp_dataset = lagged_dataset

for lag in range(1, LAGS):
    current_lag = np.roll(lagged_dataset, lag, axis=0)
    vars_num = temp_dataset.shape[1]
    # Data is added so as to be availalbe as [All_Patient_Counts, All_PM25_counts]
    temp_dataset = np.concatenate((temp_dataset[:, :vars_num//2], current_lag, temp_dataset[:, vars_num//2:]), axis=1)

lagged_dataset_X = temp_dataset[LAGS-1:-1]
lagged_dataset_Y = lagged_dataset[LAGS:]
########################################################################################

## GPy implementation of Multi-task approach - ROLLING WINDOW PREDICTION VERSION:


all_predictions = []  # Store predictions for plotting versus actual values in the end
all_rmses = []  # Store rmses for each prediction window to get average at the end
all_maes = []
all_vs = []
all_likelihoods = []
all_Betas = []
square_errors = []
correct_hits = []
mean_Beta = np.zeros((2, 2))

STARTING_DAYS = 281 - LAGS # Number of days considered "known" at the beginning
PREDICTION_WINDOW = 7 # Number of days to be predicted in each iteration

current_known_number = STARTING_DAYS

# Focusing on "Stable" part of dataset
lagged_dataset_X = lagged_dataset_X[15:330, :]
lagged_dataset_Y = lagged_dataset_Y[15:330, :]

while current_known_number < lagged_dataset_X.shape[0]:
    print("Currently considering as known days up to: ", current_known_number)
    # Set the current "known days", create relevant datasets
    X_train_main = lagged_dataset_X[:current_known_number, :LAGS]
    X_train_PM25 = lagged_dataset_X[:current_known_number, LAGS:]
    X_train_all = lagged_dataset_X[:current_known_number, :]

    Task_main = np.expand_dims(lagged_dataset_Y[:current_known_number, 0], axis=1)
    Task_PM25 = np.expand_dims(lagged_dataset_Y[:current_known_number, 1], axis=1)

    # "Test Set" refers to the measurements for the days to be predicted:
    #       Thus we start at the day after current_known_number day,
    #       and stop PREDICTION_WINDOW days later:
    test_start = current_known_number + 1
    test_end = test_start + PREDICTION_WINDOW
    # Overflow Check:
    if test_end > lagged_dataset_Y.shape[0]:
        test_end = lagged_dataset_Y.shape[0]
    X_days_test = lagged_dataset_X[test_start:test_end, :]

    Task_main_test = lagged_dataset_Y[test_start:test_end, 0]

    # Relevant datasets chosen, moving on to actual prediction of measurements,
    # using multi-task approach:

    K_RBF_1 = GPy.kern.RBF(X_train_main.shape[1])

    K1 = K_RBF_1

    model = GPy.models.GPRegression(X_train_main,
                                    Task_main,
                                    kernel=K1)

    model.optimize_restarts(num_restarts=5)

    print(model)

    all_likelihoods.append(model.log_likelihood())

    # Prediction of patient counts for the required days:

    # GPy requires the addition of a column of indexes which point to the variable the corresponding X's refer to.
    # In this case, all refer to the main task (index 0).

    next_predict_X = X_days_test[0, :]
    predicted_Xs_main = []
    predicted_Xs_extra = []
    predicted_vs_main = []
    predicted_vs_extra = []

    #print("First next_predict_X: ", next_predict_X)
    for day in range(0, Task_main_test.shape[0]):
        # Predict Next Patient count
        # Using the single entry next_predict_X, predict one day ahead
        newX = np.expand_dims(next_predict_X[:LAGS], axis=1)
        current_prediction_main, current_v_main = model.predict(Xnew=newX.T)
        current_prediction_main = current_prediction_main[0][0].astype(float)
        current_v_main = current_v_main[0][0].astype(float)
        #print("Predicted Main value for day ", day, " is ", current_prediction_main)
        predicted_Xs_main.append(current_prediction_main)
        predicted_vs_main.append(current_v_main)

        #print("Predicted_Xs_Main list: ", predicted_Xs_main)

        # In the creation of the lagged dataset, the Format of the lagged list was as follows:
        # [ Previous_day_count_value, Day_before_previous_count_value..., LAG_previous_day_count_value,
        #   LAG_previous_day_PM25_value, ... , Day_before_previous_PM25_value, Previous_day_PM25_value]

        # we exploit this format by reversing the prediction lists, so the first item of the prediction list,
        # which is the MOST RECENT predicted day, always appears first, and so on.

        # Check if we have reached the end of the test set
        if day < Task_main_test.shape[0]-1:
            # Set next_predict_X to contain the dataset values, i.e. for now it also contains information
            # it is not supposed to have (about the PREDICTION_WINDOW days)
            next_predict_X = X_days_test[day + 1, :]
            predicted_Xs_main_reversed = predicted_Xs_main[::-1]
            #print("Unmodified next_predict_X: ", next_predict_X)

            # In case the rolling window size is larger than the number of lags,
            # we shouldn't go over the number of lags in the following loop
            next_end = min(day+1, X_days_test.shape[1] // 2)
            for next_day in range(0, next_end):
                # overwrite the future days with the predicted values, exploiting the dataset's format
                # ...most recent patient count prediction goes at the front of the list
                next_predict_X[next_day] = predicted_Xs_main_reversed[next_day]
            #print("Modified next_predict_X: ", next_predict_X)
    # Store all predicted values for Accuracy Metrics and Plotting
    Task_main_predicted = np.asarray(predicted_Xs_main[:PREDICTION_WINDOW])

    vs_main = np.asarray(predicted_vs_main[:PREDICTION_WINDOW])

    for sample in range(len(predicted_Xs_main)):
        current_uc_bound = predicted_Xs_main[sample] + 1.96 * (np.sqrt(predicted_vs_main[sample]))
        current_lc_bound = predicted_Xs_main[sample] - 1.96 * (np.sqrt(predicted_vs_main[sample]))
        if (Task_main_test[sample] <= current_uc_bound) and (Task_main_test[sample] >= current_lc_bound):
            correct_hits.append(1)
        else:
            correct_hits.append(0)


    # Restore Predicted values to actual scale:
    Task_main_predicted = (Task_main_predicted * patient_std) + patient_mean
    Task_main_test = (Task_main_test * patient_std) + patient_mean
    vs_main =(vs_main * patient_std) + patient_mean
    print(Task_main_predicted)
    print(Task_main_test)

    current_rmse = np.sqrt(np.mean((Task_main_predicted-Task_main_test)**2))
    current_mae = np.mean(np.abs(Task_main_predicted-Task_main_test))
    all_rmses.append(current_rmse)
    all_maes.append(current_mae)
    all_vs.extend(vs_main)
    square_errors.extend(((Task_main_predicted-Task_main_test)**2).tolist())
    print("RMSE of patient count prediction: ", current_rmse)

    # Append predictions to total prediction list:
    all_predictions.extend(Task_main_predicted)

    # Finally, set the known days for the next iteration:
    current_known_number += PREDICTION_WINDOW

print("Average RMSE over all windows predicted: ", np.mean(all_rmses))
print("Average MAE over all windows predicted: ", np.mean(all_maes))
print("MAE over Data Mean ratio: ", np.mean(all_maes)/np.mean(lagged_dataset_Y[:, 0]))
print("Average negative log-likelihood over all models created: ", -np.mean(all_likelihoods))

print("Percentage of values within Predicted Confidence Interval:", sum(correct_hits)/len(correct_hits))

# Upper and lower 95% confidence bounds for predictions
uc_bound = np.asarray(all_predictions) + 1.96 * (np.sqrt(np.asarray(all_vs)))
lc_bound = np.asarray(all_predictions) - 1.96 * (np.sqrt(np.asarray(all_vs)))

### Prediction Inspection plot

pl.plot(range(len(all_predictions)), all_predictions, 'r')
# Invert normalisation for plotting:
patients_per_day = (lagged_dataset_Y[:, 0] * patient_std) + patient_mean
pl.plot(range(len(all_predictions)), patients_per_day[STARTING_DAYS+1:], 'b')
pl.xlabel('Day Index')
pl.ylabel('Patient Count')
pl.xlabel('Day Index', fontsize=25)
pl.ylabel('Patient Count', fontsize=25)
pl.xticks(np.arange(0, len(all_predictions), 5))
pl.tick_params(axis='both', which='major', labelsize=23)
pl.tick_params(axis='both', which='minor', labelsize=20)
pl.fill_between(range(len(all_predictions)), lc_bound, uc_bound, color='0.75')
pl.show()

