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
    //todo remove
    static Camera& instance() { static Camera c; return c; };

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

    const Matrix4 & viewMtx() const {return viewMtx_;};
    const Matrix4 & inverseViewMtx() const {return inverseViewMtx_;};
    void positionIs(const Vector3&_pos);
    void rotationIs(float _totalYaw, float _totalPitch);
    void yaw(float _degrees);
    void pitch(float _degrees);
    void move(float _dx);
    void strafe(float _dx);
    void BuildViewMatrix();
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
    float yawDegrees_;
    float yawRadians_;
    float pitchDegrees_;
    float pitchRadians_;

    float Degree2Radians(const float _degrees) { return _degrees *PI_OVER_180;};
    void BuildProjectionMatrix();

};

#endif /* CAMERA_H_ */
