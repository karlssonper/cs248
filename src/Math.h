#ifndef MATH_H
#define MATH_H

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <math.h>
#include <iostream>
#include <iomanip>

class Vector3 {
public:
    Vector3();
    Vector3(const Vector3 &_v);
    Vector3(float _x, float _y, float _z);
    float x() const { return x_; }
    float y() const { return y_; }
    float z() const { return z_; }
    float mag() const;
    Vector3 operator+(const Vector3 &_v) const;
    Vector3 operator-(const Vector3 &_v) const;
    Vector3& operator=(const Vector3 &_v);
    Vector3 operator*(float _f) const;
    Vector3 operator/(float _f) const;
    float dot(const Vector3  &_v) const;
    Vector3 cross(const Vector3 &_v) const;
    Vector3 normalize() const;
    void xIs(float _x);
    void yIs(float _y);
    void zIs(float _z);
    void print() const;
private:
    float x_;
    float y_;
    float z_;
};

class Matrix4 {
public:
    Matrix4();
    Matrix4(const Matrix4 &_m);
    Matrix4(float *_data);
    float operator()(unsigned int _index) const { return m_[_index]; }
    float operator()(unsigned int _index) { return m_[_index]; }
    float* data() { return m_; }
    Matrix4& operator=(const Matrix4 &_m);
    Matrix4 inverse() const;
    Matrix4 operator*(const Matrix4 &_m) const;
    Vector3 operator*(const Vector3 &_v) const;
    void print() const;
    static Matrix4 rotate(float _a, float _x, float _y, float _z);
    static Matrix4 translate(float _tx, float _ty, float _tz);
    static Matrix4 scale(float _sx, float _sy, float _sz);
private:
    float m_[16];
};


/*
Vector3 definitions
*/

inline Vector3::Vector3() : x_(0.f), y_(0.f), z_(0.f) {}

inline Vector3::Vector3(float _x, float _y, float _z) : x_(_x), y_(_y), z_(_z) {}

inline Vector3::Vector3(const Vector3& _v) : x_(_v.x()), y_(_v.y()), z_(_v.z()) {}

inline float Vector3::mag() const {
    return sqrt(x_*x_ + y_*y_ + z_*z_);
}

inline Vector3 Vector3::operator+(const Vector3 &_v) const {
    return Vector3(x_+_v.x(), y_+_v.y(), z_+_v.z()); 
}

inline Vector3 Vector3::operator-(const Vector3 &_v) const {
    return Vector3(x_-_v.x(), y_-_v.y(), z_-_v.z()); 
}

inline Vector3& Vector3::operator=(const Vector3 &_v) {
    if (this == &_v) return *this;
    x_ = _v.x();
    y_ = _v.y();
    z_ = _v.z();
    return *this;
}

inline Vector3 Vector3::operator*(float _f) const {
    return Vector3(x_*_f, y_*_f, z_*_f);
}

inline Vector3 Vector3::operator/(float _f) const {
    float d = 1.f/_f;
    return Vector3(x_*d, y_*d, z_*d);
}

inline float Vector3::dot(const Vector3& _v) const {
    return x_*_v.x() + y_*_v.y() + z_*_v.z();
}

inline Vector3 Vector3::normalize() const {
    Vector3 v(*this);
    return v/v.mag();
}

void Vector3::print() const {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "(" << x_ << ", " << y_ << ", " << z_ << ")" << std::endl;
}

inline Vector3 Vector3::cross(const Vector3 &_v) const {
    return Vector3(y_*_v.z() - z_*_v.y(),
                   z_*_v.x() - x_*_v.z(),
                   x_*_v.y() - y_*_v.x());
}




/*
Matrix4 definitions
*/

inline Matrix4::Matrix4() {
    m_[0]=1.f; m_[4]=0.f; m_[8] = 0.f;  m_[12]=0.f;
    m_[1]=0.f; m_[5]=1.f; m_[9] = 0.f;  m_[13]=0.f;
    m_[2]=0.f; m_[6]=0.f; m_[10] = 1.f; m_[14]=0.f;
    m_[3]=0.f; m_[7]=0.f; m_[11] = 0.f; m_[15]=1.f;
}

inline Matrix4::Matrix4(const Matrix4 &_m) {
    for (int i=0; i<16; ++i) {
        m_[i] = _m(i);
    }
}

inline Matrix4::Matrix4(float *_data) {
    for (int i=0; i<16; ++i) {
        m_[i] = _data[i];
    }
}

inline Matrix4& Matrix4::operator=(const Matrix4 &_m) {
    if (this == &_m) return *this;
        for (int i=0; i<16; ++i) {
        m_[i] = _m(i);
    }
    return *this;
}

inline Matrix4 Matrix4::operator*(const Matrix4 &_m) const {
    float temp[16];
    for (int i=0; i<4; ++i) {
        for (int j=0; j<4; ++j) {
            temp[i*4+j] = 0.f;
            for (int k=0; k<4; ++k) {
                temp[i*4+j] += m_[k*4+j] * _m(i*4+k);
            }
        }
    }
    return Matrix4(temp);
}

inline Vector3 Matrix4::operator*(const Vector3 &_v) const {
    float x, y, z, vx, vy, vz, vw;
    vx = _v.x();
    vy = _v.y();
    vz = _v.z();
    vw = 1.f;
    x = vx*m_[0] + vy*m_[4] + vz*m_[8]  + vw*m_[12];
    y = vx*m_[1] + vy*m_[5] + vz*m_[9]  + vw*m_[13];
    z = vx*m_[2] + vy*m_[6] + vz*m_[10] + vw*m_[14];
    return Vector3(x, y, z);
}

inline Matrix4 Matrix4::translate(float _tx, float _ty, float _tz) {
    float t[16];
    t[0]=1.f; t[4]=0.f; t[8]=0.f;  t[12]=_tx;
    t[1]=0.f; t[5]=1.f; t[9]=0.f;  t[13]=_ty;
    t[2]=0.f; t[6]=0.f; t[10]=1.f; t[14]=_tz;
    t[3]=0.f; t[7]=0.f; t[11]=0.f; t[15]=1.f;
    return Matrix4(t);
}

inline Matrix4 Matrix4::scale(float _sx, float _sy, float _sz) {
    float s[16];
    s[0]=_sx; s[4]=0.f; s[8]=0.f;  s[12]=0.f;
    s[1]=0.f; s[5]=_sy; s[9]=0.f;  s[13]=0.f;
    s[2]=0.f; s[6]=0.f; s[10]=_sz; s[14]=0.f;
    s[3]=0.f; s[7]=0.f; s[11]=0.f; s[15]=1.f;
    return Matrix4(s);
}

inline Matrix4 Matrix4::rotate(float _angle, float _x, float _y, float _z) {
    float norm = sqrt(_x*+_x + _y*_y + _z*_z);
    float u = _x/norm;
    float v = _y/norm;
    float w = _z/norm;
    float uu = u*u;
    float vv = v*v;
    float ww = w*w;
    float rad = _angle * M_PI / 180.f;
    float cosTheta = cos(rad);
    float sinTheta = sin(rad);
    float r[16];
    r[0] = cosTheta + uu*(1.f-cosTheta);
    r[1] = v*u*(1.f-cosTheta)+w*sinTheta;
    r[2] = w*u*(1.f-cosTheta)-v*sinTheta;
    r[3] = 0.f;
    r[4] = u*v*(1.f-cosTheta)-w*sinTheta;
    r[5] = cosTheta + vv*(1.f-cosTheta);
    r[6] = w*v*(1.f-cosTheta)+u*sinTheta;
    r[7] = 0.f;
    r[8] = u*w*(1.f-cosTheta)+v*sinTheta;
    r[9] = v*w*(1.f-cosTheta)-u*sinTheta;
    r[10] = cosTheta + ww*(1.f-cosTheta);
    r[11] = 0.f;
    r[12] = 0.f;
    r[13] = 0.f;
    r[14] = 0.f;
    r[15] = 1.f;
    return Matrix4(r);
}

void Matrix4::print() const {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << m_[0]<<" "<<m_[4]<<" "<< m_[8]<<" "<<m_[12]<<std::endl;
    std::cout << m_[1]<<" "<<m_[5]<<" "<< m_[9]<<" "<<m_[13]<<std::endl;
    std::cout << m_[2]<<" "<<m_[6]<<" "<< m_[10]<<" "<<m_[14]<<std::endl;
    std::cout << m_[3]<<" "<<m_[7]<<" "<< m_[11]<<" "<<m_[15]<<std::endl;
}













#endif