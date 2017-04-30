#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd::Zero(5);

  // initial covariance matrix
  P_ = MatrixXd::Zero(5, 5);
  
  // initial laser measurement noise covariance matrix
  R_las_ = MatrixXd::Zero(2, 2);
  
  // initial radar measurement noise covariance matrix
  R_rad_ = MatrixXd::Zero(3, 3);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  /**********
   I calculated this using the data set. It's the standard deviation of all ground truth
   vx and vy changes between successive laser and radar measurements.
  **********/
  std_a_ = 4.2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  /**********
   I tuned this starting with value 1 and decreasing it until I had acceptable results.
  **********/
  std_yawdd_ = 0.3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  
  is_initialized_ = false;

  n_x_ = 5;
  
  n_aug_ = 7;
  
  lambda_ = 3 - n_x_;
  
  Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);
  
  weights_ = VectorXd::Zero(2 * n_aug_ + 1);
  //set weights
  weights_(0) = lambda_/(lambda_+n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {
    weights_(i) = 0.5/(n_aug_+lambda_);
  }
  
  
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;
  
  R_las_ << pow(std_laspx_, 2), 0,
            0, pow(std_laspy_, 2);
  
  R_rad_ << pow(std_radr_, 2), 0, 0,
            0, pow(std_radphi_, 2), 0,
            0, 0, pow(std_radrd_, 2);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (!is_initialized_) {
    // first measurement
    
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
       Convert radar from polar to cartesian coordinates and initialize state.
       */
      float rho = meas_package.raw_measurements_(0);
      float phi = meas_package.raw_measurements_(1);
      float rho_dot = meas_package.raw_measurements_(2);
      
      if (fabs(rho) < min_value_) rho = min_value_;
      
      x_ << rho * cos(phi), rho * sin(phi), 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
       Initialize state.
       */
      float px = meas_package.raw_measurements_(0);
      float py = meas_package.raw_measurements_(1);
      
      if (fabs(px) < min_value_) px = min_value_;
      if (fabs(py) < min_value_) py = min_value_;
      
      x_ << px, py, 0, 0, 0;
    }
    
    time_us_ = meas_package.timestamp_;
    
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }
  
  // update the time
  float delta_t = (meas_package.timestamp_ - time_us_)/1000000.0;
  time_us_ = meas_package.timestamp_;
  
  Prediction(delta_t);
  
  if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
  
  //cout << "State: " << x_ << endl;
  //cout << "Covariance: " << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  
  int n_cols = 2 * n_aug_ + 1;
  
  //create augmented mean vector
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  
  //create augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  
  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, n_cols);
  
  //create augmented mean state
  x_aug.head(n_x_) = x_;
  for (int i = n_x_; i < n_aug_; ++i) {
    x_aug(i) = 0;
  }
  
  //create augmented covariance matrix
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  
  MatrixXd P_process(2,2);
  P_process << std_a_ * std_a_, 0, 0, std_yawdd_ * std_yawdd_;
  
  P_aug.bottomRightCorner(2,2) = P_process;
  
  //create square root matrix
  MatrixXd P_aug_root = P_aug.llt().matrixL();
  
  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  
  MatrixXd Xsig_aug_root = sqrt(lambda_ + n_aug_) * P_aug_root;
  
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(1+i) = x_aug + Xsig_aug_root.col(i);
  }
  
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(1+n_aug_ + i) = x_aug - Xsig_aug_root.col(i);
  }
  
  //predict sigma points
  //avoid division by zero
  //write predicted sigma points into right column
  for (int i = 0; i < n_cols; ++i) {
    VectorXd x_aug = Xsig_aug.col(i);
    VectorXd x = x_aug.head(n_x_);
    
    float delta_t2 = 0.5 * delta_t * delta_t;
    float cos_phi = cos(x_aug(3));
    float sin_phi = sin(x_aug(3));
    
    VectorXd nu_aug(n_x_);
    nu_aug << delta_t2 * cos_phi * x_aug(5),
              delta_t2 * sin_phi * x_aug(5),
              delta_t * x_aug(5),
              delta_t2 * x_aug(6),
              delta_t * x_aug(6);
    
    VectorXd x_aug_dot(n_x_);
    
    if (x_aug(4) == 0) {
      x_aug_dot << x_aug(2) * cos_phi * delta_t,
                   x_aug(2) * sin_phi * delta_t,
                   0,
                   x_aug(4) * delta_t,
                   0;
    } else {
      float ratio = x_aug(2) / x_aug(4);
      
      x_aug_dot << ratio * (sin(x_aug(3) + x_aug(4) * delta_t) - sin(x_aug(3))),
                   ratio * (-cos(x_aug(3) + x_aug(4) * delta_t) + cos(x_aug(3))),
                   0,
                   x_aug(4) * delta_t,
                   0;
    }
    
    Xsig_pred_.col(i) = x + x_aug_dot + nu_aug;
  }
  
  //predict state mean
  x_.fill(0.0);
  for (int i = 0; i < n_cols; i++) {  //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }
  
  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < n_cols; i++) {  //iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  
  // Measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 2;
  
  MatrixXd H = MatrixXd::Zero(n_z, n_x_);
  H << 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0;
  
  VectorXd z_pred = H * x_;
  
  VectorXd z = VectorXd::Zero(n_z);
  z << meas_package.raw_measurements_(0),
       meas_package.raw_measurements_(1);
  
  VectorXd z_diff = z - z_pred;
  
  MatrixXd Ht = H.transpose();
  MatrixXd S = H * P_ * Ht + R_las_;
  MatrixXd S_inv = S.inverse();
  MatrixXd K = P_ * Ht * S_inv;
  
  x_ = x_ + K * z_diff;

  MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
  
  P_ = (I - K * H) * P_;
  
  // NIS Calculation
  NIS_laser_ = (z_diff.transpose() * S_inv * z_diff).sum();
  
  //cout << "NIS laser: " << NIS_laser_ << endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  
  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;
  
  int n_cols = 2 * n_aug_ + 1;
  
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd::Zero(n_z, n_cols);
  
  //mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z);
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z,n_z);
  
  //transform sigma points into measurement space
  for (int i = 0; i < n_cols; ++i) {
    float px = Xsig_pred_(0,i);
    float py = Xsig_pred_(1,i);
    float nu = Xsig_pred_(2,i);
    float phi = Xsig_pred_(3,i);
    float rho = sqrt(pow(px, 2) + pow(py, 2));
    float gamma = atan2(py, px);
    float rho_dot = (px * cos(phi) + py * sin(phi)) * nu / rho;
    Zsig(0,i) = rho;
    Zsig(1,i) = gamma;
    Zsig(2,i) = rho_dot;
  }
  
  //calculate mean predicted measurement
  for (int i = 0; i < n_z; ++i) {
    z_pred(i) = weights_.dot(Zsig.row(i));
  }
  
  MatrixXd delta_z = Zsig.colwise() - z_pred;
  for (int i = 0; i < n_cols; ++i) {
    //angle normalization
    while (delta_z(1,i)> M_PI) delta_z(1,i)-=2.*M_PI;
    while (delta_z(1,i)<-M_PI) delta_z(1,i)+=2.*M_PI;
  }
  
  S.fill(0.);
  for (int i = 0; i < n_cols; ++i) {
    S = S + weights_(i) * delta_z.col(i) * delta_z.col(i).transpose();
  }
  S = S + R_rad_;
  
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
  
  //calculate cross correlation matrix
  MatrixXd delta_x = Xsig_pred_.colwise() - x_;
  for (int i = 0; i < n_cols; ++i) {
    //angle normalization
    while (delta_x(3,i)> M_PI) delta_x(3,i)-=2.*M_PI;
    while (delta_x(3,i)<-M_PI) delta_x(3,i)+=2.*M_PI;
  }
  
  Tc.fill(0.);
  for (int i = 0; i < n_cols; ++i) {
    Tc = Tc + weights_(i) * delta_x.col(i) * delta_z.col(i).transpose();
  }
  
  MatrixXd S_inv = S.inverse();
  
  //calculate Kalman gain K;
  MatrixXd K = Tc * S_inv;
  
  VectorXd z(3);
  z << meas_package.raw_measurements_(0),
       meas_package.raw_measurements_(1),
       meas_package.raw_measurements_(2);
  
  MatrixXd z_diff = z - z_pred;
  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
  
  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
  
  // calculate the radar NIS
  NIS_radar_ = (z_diff.transpose() * S_inv * z_diff).sum();
  
  //cout << "NIS radar: " << NIS_radar_ << endl;
}
