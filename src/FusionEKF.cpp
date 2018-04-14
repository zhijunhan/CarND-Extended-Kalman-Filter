#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
  			  0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
  			  0, 0.0009, 0,
  			  0, 0, 0.09;

  //Set the process and measurement noises
  H_laser_ << 1, 0, 0, 0,
  			  0, 1, 0, 0;

  Hj_ << 1, 1, 0, 0,
  		 1, 1, 0, 0,
  		 1, 1, 1, 1; 
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    //Initialize states for first time
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;
    //Initialize the state covariance matrix P_
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
    		   0, 1, 0, 0,
    		   0, 0, 1000, 0,
    		   0, 0, 0, 1000;
    //Initialize the system transition matrix F_
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ << 1, 0, 1, 0,
    		   0, 1, 0, 1,
    		   0, 0, 1, 0,
    		   0, 0, 0, 1;
    ekf_.Q_ = MatrixXd::Zero(4, 4);

    // states placeholders
    double px = 0;
    double py = 0;
    double vx = 0;
    double vy = 0;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      double rho = measurement_pack.raw_measurements_[0];
      double phi = measurement_pack.raw_measurements_[1];
      double rho_dot = measurement_pack.raw_measurements_[2];

      px = rho * cos(phi);
      py = rho * sin(phi);
      vx = rho_dot * cos(phi);
      vy = rho_dot * sin(phi);
    }

    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      px = measurement_pack.raw_measurements_[0];
      py = measurement_pack.raw_measurements_[1];
    }

    ekf_.x_ << px, py, vx, vy;

    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;

    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  /**
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  //compute the time elapsed between the current and previous measurements
  double current_timestamp = measurement_pack.timestamp_;
  double dt = (current_timestamp - previous_timestamp_) / 1000000.0;
  //update timestamp cursor
  previous_timestamp_ = measurement_pack.timestamp_;
  
  double dt_2 = dt * dt;
  double dt_3 = dt_2 * dt;
  double dt_4 = dt_3 * dt;

  //update process transition matrix F_ based off time differentiation
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  //Set the noise
  double noise_ax = 9.0;
  double noise_ay = 9.0;
  //update the process noise covariance matrix Q_
  MatrixXd Qv = MatrixXd::Zero(2, 2);
  Qv(0,0) = noise_ax;
  Qv(1,1) = noise_ay;
  MatrixXd G = MatrixXd::Zero(4, 2);
  G(0, 0) = dt_2/2.0;
  G(1, 1) = dt_2/2.0;
  G(2, 0) = dt;
  G(3, 1) = dt;
  ekf_.Q_ = G * Qv * G.transpose();

  // Predict
  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
