#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using namespace std;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  
  if (estimations.size() == 0) {
    cout << "You must pass in a non-empty vector of estimations" << endl;
    return rmse;
  }
  if (estimations.size() != ground_truth.size()) {
    cout << "You must pass in a vector of true values equal in size to the vector of estimations" << endl;
    return rmse;
  }
  
  
  VectorXd residuals(4);
  residuals << 0,0,0,0;
  
  //accumulate squared residuals
  for(int i=0; i < estimations.size(); ++i) {
    
    VectorXd c = estimations[i] - ground_truth[i];
    residuals = residuals.array() + (c.array() * c.array());
  }
  
  //calculate the mean
  residuals /= estimations.size();
  
  //calculate the squared root
  rmse = residuals.array().sqrt();
  
  //return the result
  return rmse;
}
