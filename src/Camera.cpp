/*
 * Camera.cpp
 *
 *  Created on: Mar 2, 2012
 *      Author: per
 */

#include "Camera.h"
#include <limits>

Camera::Camera() :
    fov_(45.f),
    aspectRatio_(1.6f),
    nearPlane_(1.f),
    farPlane_(100.f),
    pos_(Vector3(0.f,0.f,0.f)),
    yawDegrees_(0.f),
    yawRadians_(0.f),
    pitchDegrees_(0.f),
    pitchRadians_(0.f),
    shaking_(false),
    shakeTime_(0.f),
    shakeDuration_(0.f),
    shakeMagnitude_(0.f),
    shakeYaw_(0.f),
    shakePitch_(0.f),
    maxYaw_(std::numeric_limits<float>::max()),
    maxPitch_(std::numeric_limits<float>::max()),
    minYaw_(std::numeric_limits<float>::min()),
    minPitch_(std::numeric_limits<float>::min()) 
{
    BuildProjectionMatrix();
    BuildViewMatrix();
}

Camera::Camera(const Vector3 &_pos, float _yaw, float _pitch) :
    fov_(45.f),
    aspectRatio_(1.6f),
    nearPlane_(1.f),
    farPlane_(100.f),
    pos_(_pos),
    yawDegrees_(_yaw),
    yawRadians_(Degree2Radians(_yaw)),
    pitchDegrees_(_pitch),
    pitchRadians_(Degree2Radians(_pitch)),
    shaking_(false),
    shakeTime_(0.f),
    shakeDuration_(0.f),
    shakeMagnitude_(0.f),
    shakeYaw_(0.f),
    shakePitch_(0.f),
    maxYaw_(std::numeric_limits<float>::max()),
    maxPitch_(std::numeric_limits<float>::max()),
    minYaw_(std::numeric_limits<float>::min()),
    minPitch_(std::numeric_limits<float>::min()) 
{
    BuildProjectionMatrix();
    BuildViewMatrix();
}

void Camera::projectionIs(float _fov,
                          float _aspectRatio,
                          float _near,
                          float _far)
{
    fov_ = _fov;
    aspectRatio_ = _aspectRatio;
    nearPlane_ = _near;
    farPlane_ = _far;
    BuildProjectionMatrix();
}

void Camera::fovIs(float _fov)
{
    if (fov_ != _fov){
        fov_ = _fov;
        BuildProjectionMatrix();
    }
}

void Camera::aspectRatioIs(float _aspectRatio)
{
    if (aspectRatio_ != _aspectRatio){
        aspectRatio_ = _aspectRatio;
        BuildProjectionMatrix();
    }
}

void Camera::nearPlaneIs(float _nearPlane)
{
    if (nearPlane_ != _nearPlane){
        nearPlane_ = _nearPlane;
        BuildProjectionMatrix();
    }
}

void Camera::farPlaneIs(float _farPlane)
{
    if (farPlane_ != _farPlane){
        farPlane_ = _farPlane;
        BuildProjectionMatrix();
    }
}

void Camera::positionIs(const Vector3&_pos)
{
    pos_ = _pos;
}

void Camera::rotationIs(float _totalYaw, float _totalPitch)
{
    yawDegrees_ = _totalYaw;
    yawRadians_ = Degree2Radians(yawDegrees_);
    pitchDegrees_ = _totalPitch;
    pitchRadians_ = Degree2Radians(pitchDegrees_);

}

void Camera::yaw(float _degrees)
{
    yawDegrees_ += _degrees;
    if (yawDegrees_ >= maxYaw_) yawDegrees_ -= _degrees;
    else if (yawDegrees_ <= minYaw_) yawDegrees_ -= _degrees;
    yawRadians_ = Degree2Radians(yawDegrees_);
}

void Camera::pitch(float _degrees)
{
    pitchDegrees_ += _degrees;
    if (pitchDegrees_ >= maxPitch_) pitchDegrees_ -= _degrees;
    else if (pitchDegrees_ <= minPitch_) pitchDegrees_ -= _degrees;
    pitchRadians_ = Degree2Radians(pitchDegrees_);
}

void Camera::move(float _dx)
{
    pos_.x -= _dx*sin(yawRadians_);
    pos_.y += _dx*sin(pitchRadians_);
    pos_.z += _dx*cos(yawRadians_);
}

void Camera::strafe(float _dx)
{
    pos_.x -= _dx*cos(yawRadians_);
    pos_.z -= _dx*sin(yawRadians_);
}

void Camera::BuildProjectionMatrix()
{
    float xymax = nearPlane_ * tan(fov_ * PI_OVER_360);
    float ymin = -xymax;
    float xmin = -xymax;

    float width = xymax - xmin;
    float height = xymax - ymin;

    float depth = farPlane_ - nearPlane_;
    float q = -(farPlane_ + nearPlane_) / depth;
    float qn = -2 * (farPlane_ * nearPlane_) / depth;

    float w = 2 * nearPlane_ / width;
    w = w / aspectRatio_;
    float h = 2 * nearPlane_ / height;

    projectionMtx_.m_[0]  = w;
    projectionMtx_.m_[1]  = 0;
    projectionMtx_.m_[2]  = 0;
    projectionMtx_.m_[3]  = 0;

    projectionMtx_.m_[4]  = 0;
    projectionMtx_.m_[5]  = h;
    projectionMtx_.m_[6]  = 0;
    projectionMtx_.m_[7]  = 0;

    projectionMtx_.m_[8]  = 0;
    projectionMtx_.m_[9]  = 0;
    projectionMtx_.m_[10] = q;
    projectionMtx_.m_[11] = -1;

    projectionMtx_.m_[12] = 0;
    projectionMtx_.m_[13] = 0;
    projectionMtx_.m_[14] = qn;
    projectionMtx_.m_[15] = 0;
}

void Camera::BuildViewMatrix()
{
    viewMtx_ = Matrix4::rotate(pitchDegrees_+shakePitch_, 1.f, 0.f, 0.f);
    viewMtx_ = viewMtx_ * Matrix4::rotate(yawDegrees_+shakeYaw_, 0.f, 1.f, 0.f);
    viewMtx_ = viewMtx_ * Matrix4::translate(pos_.x, pos_.y, pos_.z);

    inverseViewMtx_ = Matrix4::translate(-pos_.x, -pos_.y, -pos_.z);
    inverseViewMtx_ = inverseViewMtx_ *
            Matrix4::rotate(-yawDegrees_-shakeYaw_, 0.f, 1.f, 0.f);
    inverseViewMtx_ = inverseViewMtx_ *
            Matrix4::rotate(-pitchDegrees_-shakePitch_, 1.f, 0.f, 0.f);
}

void Camera::shake(float _duration, float _magnitude) {

    shaking_ = true;
    shakeDuration_ = _duration;
    shakeTime_ =_duration;
    shakeMagnitude_ = _magnitude;

    shakeYaw_ = shakeMagnitude_ * Random::randomFloat(-1.f, 1.f);
    shakePitch_ = shakeMagnitude_ * Random::randomFloat(-1.f, 1.f);
}

void Camera::updateShake(float _dt) {
    if (shaking_) {
        float t = 1.f - shakeTime_/shakeDuration_;
        shakeMagnitude_ *= (2.f*t*t*t-3*t*t+1);
        shakeYaw_ = shakeMagnitude_ * Random::randomFloat(-1.f, 1.f);
        shakePitch_ = shakeMagnitude_ * Random::randomFloat(-1.f, 1.f);
        shakeTime_ -= _dt;
        if (shakeTime_ < 0.f) {
            shakeYaw_ = shakePitch_ = 0.f;
            shaking_ = false;
        }
    }
}

void Camera::maxYawIs(float _maxYaw) {
    maxYaw_ = _maxYaw;
}

void Camera::maxPitchIs(float _maxPitch) {
    maxPitch_ = _maxPitch;
}

void Camera::minYawIs(float _minYaw) {
    minYaw_ = _minYaw;
}

void Camera::minPitchIs(float _minPitch) {
    minPitch_ = _minPitch;
}

void Camera::lookAt(Vector3 _eye, Vector3 _center, Vector3 _up)
{
    viewMtx_ = Matrix4::lookAt(_eye, _center, _up);
}

void Camera::BuildOrthoProjection(Vector3 _min, Vector3 _max)
{
    std::cerr << "Building matrix!";
    projectionMtx_.m_[0] = 2.0f / (_max.x - _min.x);
    projectionMtx_.m_[1] = 0.0f;
    projectionMtx_.m_[2] = 0.0f;
    projectionMtx_.m_[3] = 0.0f;
    projectionMtx_.m_[4] = 0.0f;
    projectionMtx_.m_[5] = 2.0f / (_max.y - _min.y);
    projectionMtx_.m_[6] = 0.0f;
    projectionMtx_.m_[7] = 0.0f;
    projectionMtx_.m_[8] = 0.0f;
    projectionMtx_.m_[9] = 0.0f;
    projectionMtx_.m_[10] = 2.0f / (_min.z - _max.z);
    projectionMtx_.m_[11] = 0.0f;
    projectionMtx_.m_[12] = (_max.x + _min.x) / (_min.x - _max.x);
    projectionMtx_.m_[13] = (_max.y + _min.y) / (_min.y - _max.y);
    projectionMtx_.m_[14] = (_max.z + _min.z) / (_min.z - _max.z);
    projectionMtx_.m_[15] = 1.0f;
}


