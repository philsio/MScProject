import numpy as np
import pandas as pd
import GPy
import pylab as pl
import matplotlib.pyplot as plt

print('Importing .csv Data...')
# First we Import the .csv Data Files
patient_data = pd.read_csv('data/patients_area_a_Admission_all_J_codes.csv', delimiter=',', header=0)
pollutant_data = pd.read_csv('data/air_pollution_area_a.csv', delimiter=',', header=0)
print('Done.')

print('Preprocessing Data to form usable for the approach decided...')
total_patients = patient_data.shape[0]

#total_pollutant_measurements = pollutant_data.shape[0]
total_pollutant_measurements = 413

start_date = pollutant_data.iat[0, 0]
end_date = pollutant_data.iat[-1, 0]

# Define a list of the applicable dates, so we can then match dates to indices
dates = pd.date_range(start='2014-12-1', end='2016-1-31', freq='D')
#dates = pd.date_range(start=start_date, end=end_date, freq='D')
patient_data = (patient_data[patient_data['MEDICAL_ORG_CODE'] == '45090340X'])
current_total_patients = patient_data.shape[0]
total_dates = dates.shape[0]

# Count the number of patients admitted per day
patients_per_day = np.zeros(total_dates)
for patient in range(0, current_total_patients):
    admission_date = patient_data.iat[patient, 14]
    if admission_date in dates:
        date_index = dates.get_loc(admission_date)
        patients_per_day[date_index] += 1


pollutant_measurements = np.zeros([total_dates, pollutant_data.shape[1]-1])

### Add a dummy column that points to the relevant dates:
dates_range_indices = np.asarray(list(range(1, total_dates+1)))
dates_range_indices = np.expand_dims(dates_range_indices, axis=1)

pollutant_measurements = np.concatenate((dates_range_indices, pollutant_measurements), axis=1)

for measurement in range(0, total_pollutant_measurements):
    measurement_date = pollutant_data.iat[measurement, 0]
    date_index = dates.get_loc(measurement_date)
    for element in range(1, 9):
        pollutant_measurements[date_index, element] = pollutant_data.iat[measurement, element]

# Get rid of NaN values
pollutant_measurements = np.nan_to_num(pollutant_measurements)
# Drop missing measurements:
to_remove = []
for measurement in range(1, total_dates):
    if all(pollutant_measurements[measurement, 1:] == np.zeros(pollutant_data.shape[1]-1)):
        to_remove.append(measurement)
pollutant_measurements = np.delete(pollutant_measurements, to_remove, axis=0)
all_measurements = pollutant_measurements
print('Done.')

# Remove relevant measurements from patient array
patients_per_day = np.delete(patients_per_day, to_remove, axis=0)

# Normalisation of the pollutant data
input_variables_num = all_measurements.shape[1]
for variable in range(0, input_variables_num):
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


print(all_measurements.shape)
print(patients_per_day.shape)

########################################################################################
# Creation of "Aggregation Variables" corresponding to each variable of the original dataset.
#
# These will average the values of a number of previous days (e.g. a week) in an attempt to
# incorporate the "looking back" approach that seems to be appropriate for this problem.

meas = all_measurements.shape[1] - 1
dates = all_measurements.shape[0]
aggregation_array = np.zeros([dates, meas])
aggregation_count = 7  # Number of days to look back over

current_count = 0
for day in range(0, dates):
    while (current_count <= aggregation_count) and (day - current_count >= 0):
        for variable in range(1, all_measurements.shape[1]):
            aggregation_array[day, variable-1] += all_measurements[day-current_count, variable]
        current_count += 1
    aggregation_array[day, :] /= current_count
    current_count = 0

all_measurements = np.concatenate((all_measurements, aggregation_array), axis=1)
print(all_measurements.shape)

########################################################################################

## GPy implementation of Multi-task approach - ROLLING WINDOW PREDICTION VERSION:

# Reference list for all_measurements array:
# - Patient Count (the actual goal of this problem)
# - CO: index 1 of the revised array
# - NO2: index 3 of the revised array
# - O3: index 4 of the revised array
# - PM10: index 5 of the revised array - 12 for aggregated
# - PM2.5: index 6 of the revised array - 13 for aggregated
# - SO2: index 7 of the revised array

all_predictions = []  # Store predictions for plotting versus actual values in the end
all_vs = []
all_rmses = []  # Store rmses for each prediction window to get average at the end
all_maes = []
all_Betas = []
all_likelihoods = []
TRAIN_SIZE = 280
STARTING_DAYS = 267 # Number of days considered "known" at the beginning
PREDICTION_WINDOW = 7 # Number of days to be predicted in each iteration

current_known_number = STARTING_DAYS

# Focus on "stable" period of Hospital 340X
all_measurements = all_measurements[15:337, :]
patients_per_day = patients_per_day[15:337]

while current_known_number < all_measurements.shape[0]:
    print("Currently considering as known days up to: ", current_known_number)
    # Set the current "known days", create relevant datasets
    X_days = all_measurements[:current_known_number, 0]
    X_days = np.expand_dims(X_days, axis=1)

    Task_main = patients_per_day[:current_known_number]
    # Access to "basic" variables
    #Task_CO = np.expand_dims(all_measurements[:current_known_number, 1], axis=1)
    #Task_NO2 = np.expand_dims(all_measurements[:current_known_number, 3], axis=1)
    #Task_O3 = np.expand_dims(all_measurements[:current_known_number, 4], axis=1)
    #Task_PM10 = np.expand_dims(all_measurements[:current_known_number, 5], axis=1)
    #Task_PM25 = np.expand_dims(all_measurements[:current_known_number, 6], axis=1)
    #Task_SO2 = np.expand_dims(all_measurements[:current_known_number, 7], axis=1)

    # Access to "aggregated" variables
    Task_CO = np.expand_dims(all_measurements[:current_known_number, 8], axis=1)
    Task_NO2 = np.expand_dims(all_measurements[:current_known_number, 10], axis=1)
    Task_O3 = np.expand_dims(all_measurements[:current_known_number, 11], axis=1)
    Task_PM10 = np.expand_dims(all_measurements[:current_known_number, 12], axis=1)
    Task_PM25 = np.expand_dims(all_measurements[:current_known_number, 13], axis=1)
    Task_SO2 = np.expand_dims(all_measurements[:current_known_number, 14], axis=1)


    # "Test Set" refers to the measurements for the days to be predicted:
    #       Thus we start at the day after current_known_number day,
    #       and stop PREDICTION_WINDOW days later:
    test_start = current_known_number + 1
    test_end = test_start + PREDICTION_WINDOW
    # Overflow Check:
    if test_end > patients_per_day.shape[0]:
        test_end = patients_per_day.shape[0]
    X_days_test = all_measurements[test_start:test_end, 0]
    X_days_test = np.expand_dims(X_days_test, axis=1)

    Task_main_test = patients_per_day[test_start:test_end]

    # Relevant datasets chosen, moving on to actual prediction of measurements,
    # using multi-task approach:

    K_RBF = GPy.kern.RBF(1)
    K_perexp = GPy.kern.PeriodicExponential(input_dim=1)
    K_Matern = GPy.kern.PeriodicMatern32(1)
    K_poly_bi = GPy.kern.Poly(1, order=2, bias=1)
    K_poly_high = GPy.kern.Poly(1, order=3)

    #K = K_perexp + K_poly_bi -> RMSE 27.10
    #K = K_perexp + K_poly_high # -> RMSE 26.87 for order 3
    #K = K_Matern + K_poly_bi -> 26.05 # Not very informative plot
    #K = K_Matern + K_poly_high -> 26.69 # Not very informative plot (order 5)
    #K = K_perexp
    K = K_Matern + K_RBF

    model = GPy.models.GPRegression(X=X_days, Y=Task_main,kernel=K)

    model.optimize_restarts(num_restarts=5)
    print(model)

    all_likelihoods.append(model.log_likelihood())

    # Prediction of patient counts for the required days:

    newX = np.arange(test_start, test_end)[:, None]
    Task_main_predicted, Task_main_vs = model.predict(Xnew=newX)

    # Restore Predicted values to actual scale:
    Task_main_predicted = (Task_main_predicted * patient_std) + patient_mean
    Task_main_test = (Task_main_test * patient_std) + patient_mean
    print(Task_main_predicted)
    print(Task_main_test)

    Task_main_vs = (Task_main_vs * patient_std) + patient_mean

    current_rmse = np.sqrt(np.mean((Task_main_predicted-Task_main_test)**2))
    current_mae = np.mean(np.abs(Task_main_predicted-Task_main_test))
    all_rmses.append(current_rmse)
    all_maes.append(current_mae)
    print("RMSE of patient count prediction: ", current_rmse)

    # Append predictions to total prediction list:
    all_predictions.extend(Task_main_predicted)
    all_vs.extend(Task_main_vs)

    # Finally, set the known days for the next iteration:
    current_known_number += PREDICTION_WINDOW

print("Average RMSE over all windows predicted: ", np.mean(all_rmses))
print("Average MAE over all windows predicted: ", np.mean(all_maes))
print("Average negative log-likelihood over all models created: ", -np.mean(all_likelihoods))

# Upper and lower 95% confidence bounds for predictions
uc_bound = np.asarray(all_predictions) + 1.96 * (np.sqrt(np.asarray(all_vs)))
lc_bound = np.asarray(all_predictions) - 1.96 * (np.sqrt(np.asarray(all_vs)))

# Invert normalisation for plotting:
patients_per_day = (patients_per_day * patient_std) + patient_mean

correct_hits = []
for sample in range(len(all_predictions)):
    correct_value = patients_per_day[STARTING_DAYS+1+sample]
    if (correct_value <= uc_bound[sample]) and (correct_value >= lc_bound[sample]):
        correct_hits.append(1)
    else:
        correct_hits.append(0)
print("Percentage of values within Predicted Confidence Interval:", sum(correct_hits)/len(correct_hits))

### Prediction Inspection plot
pl.plot(range(len(all_predictions)), all_predictions, 'r')
pl.plot(range(len(all_predictions)), patients_per_day[STARTING_DAYS+1:], 'b')
pl.xlabel('Day Index')
pl.ylabel('Patient Count')
pl.xlabel('Day Index', fontsize=18)
pl.ylabel('Patient Count', fontsize=18)
pl.xticks(np.arange(0, len(all_predictions), 5))
pl.tick_params(axis='both', which='major', labelsize=14)
pl.tick_params(axis='both', which='minor', labelsize=10)
pl.fill_between(range(len(all_predictions)), lc_bound[:, 0], uc_bound[:, 0], color='0.75')
pl.show()

#for i in range(0, len(all_Betas)):
#    Beta = all_Betas[i]
#    plt.imshow(Beta, cmap='autumn')
#    plt.show()
exit()
