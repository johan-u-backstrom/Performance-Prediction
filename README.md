# Performance Prediction

Performance Prediction provides CD (cross directional) performance predictions for 
CD-MPC based multivariable (multi beam) cross directional control for Honeywell's
QCS 4.0 web application

## Project Structure

The application has the following project (directory) structure:

```
|- Performance-Prediction
    |- data
    |- source
    |- test
```


## Class Structure

This is an object oriented application with the following class structure:

```
|- CDPerformancePrediction
    |- CDSystem
    |- cd_measurements
    |- cd_actuators
    |- CDProcessModel
    |- CDMPC
        |- QP
```

Here ```cd_measurements``` is a list of ```cd_measurement``` objects which are instances of the ```CDMeasurement``` class. Similarly, ```cd_actuators``` is a list of ```cd_actuator``` objects, which are instances of the ```CDActuator``` class.

Each class has its own module (file), with the follwing convention: 

The ```CDActuator``` class is implmented in the Python module ```cd_actuator.py```.

## Application Testing

Application testing is done by executing the ```test_run_script.py``` in the application root directory ```Performance-Prediction```. The test script in turn execute test cases in ```\test\test_cases.py```.

A typical test case pattern is as follows, an excerpt from ```test_case_22.py```:

``` Python
# Load the Matlab generated CDActuators struct
data_file = 'cdActuators.json'
data_file_path = data_dir + '/' + data_file
with open(data_file_path, 'r') as f:
    cd_actuators_matlab = json.load(f)

# Load the input data for the CDPerformancePrediction Class
[system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data] = load_performance_prediction_data()

# Create a cd_performance_prediction object
cd_performance_prediction = CDPerformancePrediction(system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data)

# Get the optimal setpoints u(k) from the cd_actuator objects
# (instances of the CDActuator class)    
k = 0
for cd_actuator in cd_performance_prediction.cd_actuators:
    u = cd_actuator.u
    u_matlab = cd_actuators_matlab[k].get('finalProfile')
    k += 1
```
