/*
 * Camera.h
 *
 *  Created on: Mar 2, 2012
 *      Author: per
 */

#ifndef CAMERA_H_
#define CAMERA_H_

#define PI_OVER_180 0.017453292f
#define PI_OVER_360 0.008726646f

#include "MathEngine.h"

class Camera
{
public:
    Camera();
    Camera(const Vector3 &_pos, float _yaw, float _pitch);

    const Matrix4 & projectionMtx() const { return projectionMtx_;};
    void projectionIs(float _fov, float _aspectRatio, float _near, float _far);
    float fov() const { return fov_;};
    void fovIs(float _fov);
    float aspectRatio() const { return aspectRatio_;};
    void aspectRatioIs(float _aspectRatio);
    float nearPlane() const { return nearPlane_;};
    void nearPlaneIs(float _nearPlane);
    float farPlane() const { return farPlane_;};
    void farPlaneIs(float _farPlane);
    void maxPitchIs(float _maxPitch);
    void maxYawIs(float _maxYaw);
    void minPitchIs(float _minPitch);
    void minYawIs(float _minYaw);
    float yaw() const { return yawDegrees_; }
    float pitch() const { return pitchDegrees_; }

    const Matrix4 & viewMtx() const {return viewMtx_;};
    const Matrix4 & inverseViewMtx() const {return inverseViewMtx_;};
    void positionIs(const Vector3&_pos);
    void rotationIs(float _totalYaw, float _totalPitch);
    void yaw(float _degrees);
    void pitch(float _degrees);
    void move(float _dx);
    void strafe(float _dx);
    void BuildViewMatrix();


    Vector3 viewVector() const;
    Vector3 worldPos() const { return worldPos_; }
   

    void shake(float _duration, float _magnitude);
    void updateShake(float _dt);

    //For Light Camera only
    void lookAt(Vector3 _eye, Vector3 _center, Vector3 _up);
    void BuildOrthoProjection(Vector3 _min, Vector3 _max);
private:
    Camera(const Camera&);
    void operator=(const Camera&);

    Matrix4 projectionMtx_;
    float fov_;
    float aspectRatio_;
    float nearPlane_;
    float farPlane_;

    Matrix4 viewMtx_;
    Matrix4 inverseViewMtx_;
    Vector3 pos_;
    Vector3 worldPos_;
    float yawDegrees_;
    float yawRadians_;
    float pitchDegrees_;
    float pitchRadians_;

    float Degree2Radians(const float _degrees) { return _degrees *PI_OVER_180;};
    void BuildProjectionMatrix();

    float maxYaw_;
    float maxPitch_;
    float minYaw_;
    float minPitch_;

    bool shaking_;
    float shakeDuration_;
    float shakeTime_;
    float shakeMagnitude_;
    float shakeYaw_;
    float shakePitch_;

};

#endif /* CAMERA_H_ */
