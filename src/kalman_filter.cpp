#include <iostream>
#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /*
    A priori estimation and process covariance prediction 
  */
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /*
    A posterior update and correction
  */
  // Innovation or measurement pre-fit residual
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  // Take care of the angle
  if(y[1] > M_PI) y[1] -= 2.0*M_PI;
  else if(y[1] < -M_PI) y[1] += 2.0*M_PI;
  // Innovation covariance
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  // Optimal Kalman gain
  MatrixXd K = P_ * Ht * Si;

  // update states
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /*
    update the state by using Extended Kalman Filter equations
  */ 

  double px = x_[0];
  double py = x_[1];
  double vx = x_[2];
  double vy = x_[3];
  //convert states for Radar EKF update
  double rho = hypot(px, py);
  double phi = atan2(py, px);
  double rho_dot = 0.0;
  //check division by zero
  if (fabs(rho) < 0.0001) rho = 0.0001;
  rho_dot = (px * vx + py * vy) / rho;

  // Innovation or measurement pre-fit residual
  VectorXd z_pred(3);
  z_pred << rho, phi, rho_dot;
  VectorXd y = z - z_pred;

  if(y[1] > M_PI)
    y[1] -= 2.0*M_PI;
  else if(y[1] < -M_PI)
    y[1] += 2.0*M_PI;
  // Innovation covariance
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  // Optimal Kalman gain
  MatrixXd K = P_ * Ht * Si;

  // new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
