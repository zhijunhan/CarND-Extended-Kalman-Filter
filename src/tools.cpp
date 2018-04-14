#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
    * Calculate the RMSE here.
  */
  VectorXd rmse = VectorXd::Zero(4);

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if(estimations.size() != ground_truth.size()
     || estimations.size() == 0){
    cout << "Invalid estimation or ground_truth data" << endl;
    return rmse;
  }

  //accumulate squared residuals
  for(unsigned int i=0; i < estimations.size(); ++i){

    VectorXd residual = estimations[i] - ground_truth[i];

    //coefficient-wise multiplication
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  //calculate the mean
  rmse = rmse/estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //return the result
  return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
    * Calculate a Jacobian here.
  */
  MatrixXd Hj = MatrixXd::Zero(3, 4);
  //recover state parameters
  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);
    
  // pre-compute a set of terms to avoid repeated calculation
  double c1 = px*px + py*py;
  double c2 = sqrt(c1);
  double c3 = c1*c2;

  //check division by zero
  if(c1 < 0.0001)
  {
    cout << "CalculateJacobian has Error - Division by zero" << endl;
    return Hj;
  }
  //compute the Jacobian matrix
  Hj(0,0) = px/c2;
  Hj(0,1) = py/c2;
  Hj(1,0) = -py/c1;
  Hj(1,1) = px/c1;
  Hj(2,0) = py*(vx*py - vy*px)/c3;
  Hj(2,1) = px*(px*vy - py*vx)/c3;
  Hj(2,2) = px/c2;
  Hj(2,3) = py/c2;

  return Hj;
}
