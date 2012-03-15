#ifndef MATHENGINE_H
#define MATHENGINE_H

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#include <math.h>
#include <iostream>
#include <iomanip>
#include <stdlib.h>

class Vector2 {
public:
    Vector2(float _x, float _y);
    Vector2();
    Vector2(const Vector2 &_v);
    float x;
    float y;
};

class Vector3 {
public:
    Vector3();
    Vector3(const Vector3 &_v);
    Vector3(float _x, float _y, float _z);
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
    float x;
    float y;
    float z;
};

class Matrix4 {
public:
    Matrix4();
    Matrix4(const Matrix4 &_m);
    Matrix4(float *_data);
    float operator()(unsigned int _index) const { return m_[_index]; }
    float operator()(unsigned int _index) { return m_[_index]; }
    const float* data() const{ return m_; }
    void makeIdentity();
    Matrix4& operator=(const Matrix4 &_m);
    Matrix4 inverse() const;
    Matrix4 transpose() const;
    Matrix4 operator*(const Matrix4 &_m) const;
    Vector3 operator*(const Vector3 &_v) const;
    void print() const;
    static Matrix4 rotate(float _a, float _x, float _y, float _z);
    static Matrix4 translate(float _tx, float _ty, float _tz);
    static Matrix4 scale(float _sx, float _sy, float _sz);
    float m_[16];
};

class Matrix3 {
public:
    Matrix3();
    Matrix3(const Matrix4 & _m);
    Matrix3 inverse() const;
    Matrix3 transpose() const;
    const float* data() const { return m_;};
    float m_[9];
    void print() const;
};

/* 
Vector2 definitions
*/

inline Vector2::Vector2() : x(0.f), y(0.f) {}
inline Vector2::Vector2(float _x, float _y) : x(_x), y(_y) {}
inline Vector2::Vector2(const Vector2& _v) : x(_v.x), y(_v.y) {}


/*
Vector3 definitions
*/

inline Vector3::Vector3() : x(0.f), y(0.f), z(0.f) {}

inline Vector3::Vector3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

inline Vector3::Vector3(const Vector3& _v) : x(_v.x), y(_v.y), z(_v.z) {}

inline float Vector3::mag() const {
    return sqrt(x*x + y*y + z*z);
}

inline Vector3 Vector3::operator+(const Vector3 &_v) const {
    return Vector3(x+_v.x, y+_v.y, z+_v.z);
}

inline Vector3 Vector3::operator-(const Vector3 &_v) const {
    return Vector3(x-_v.x, y-_v.y, z-_v.z);
}

inline Vector3& Vector3::operator=(const Vector3 &_v) {
    if (this == &_v) return *this;
    x = _v.x;
    y = _v.y;
    z = _v.z;
    return *this;
}

inline Vector3 Vector3::operator*(float _f) const {
    return Vector3(x*_f, y*_f, z*_f);
}

inline Vector3 Vector3::operator/(float _f) const {
    float d = 1.f/_f;
    return Vector3(x*d, y*d, z*d);
}

inline float Vector3::dot(const Vector3& _v) const {
    return x*_v.x + y*_v.y + z*_v.z;
}

inline Vector3 Vector3::normalize() const {
    Vector3 v(*this);
    return v/v.mag();
}

inline void Vector3::print() const {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "(" << x << ", " << y << ", " << z << ")" << std::endl;
}

inline Vector3 Vector3::cross(const Vector3 &_v) const {
    return Vector3(y*_v.z - z*_v.y,
                   z*_v.x - x*_v.z,
                   x*_v.y - y*_v.x);
}

/*
Matrix3 definitions
*/
inline Matrix3::Matrix3()
{
    m_[0] = 1.f;
    m_[1] = 0.f;
    m_[2] = 0.f;
    m_[3] = 0.f;
    m_[4] = 1.f;
    m_[5] = 0.f;
    m_[6] = 0.f;
    m_[7] = 0.f;
    m_[8] = 1.f;
}

inline Matrix3::Matrix3(const Matrix4 & _m)
{
    m_[0] = _m.data()[0];
    m_[1] = _m.data()[1];
    m_[2] = _m.data()[2];
    m_[3] = _m.data()[4];
    m_[4] = _m.data()[5];
    m_[5] = _m.data()[6];
    m_[6] = _m.data()[8];
    m_[7] = _m.data()[9];
    m_[8] = _m.data()[10];

}

inline Matrix3 Matrix3::inverse() const
{
    Matrix3 t;

    float det = m_[0]*m_[4]*m_[8]+m_[1]*m_[5]*m_[6]+
       m_[2]*m_[3]*m_[7]-m_[0]*m_[5]*m_[7]-m_[2]*m_[4]*m_[6]-m_[1]*m_[3]*m_[8];
    if (abs(det) < 0.00001f){
        det = det>0 ? 0.00001f : -0.00001f;
    }

    float one_over_det = 1.f/det;


    t.m_[0] = one_over_det*(m_[4]*m_[8] - m_[7]*m_[5]);
    t.m_[1] = one_over_det*(m_[7]*m_[2] - m_[1]*m_[8]);
    t.m_[2] = one_over_det*(m_[1]*m_[5] - m_[4]*m_[2]);
    t.m_[3] = one_over_det*(m_[6]*m_[5] - m_[3]*m_[8]);
    t.m_[4] = one_over_det*(m_[0]*m_[8] - m_[6]*m_[2]);
    t.m_[5] = one_over_det*(m_[3]*m_[2] - m_[0]*m_[5]);
    t.m_[6] = one_over_det*(m_[3]*m_[7] - m_[6]*m_[4]);
    t.m_[7] = one_over_det*(m_[6]*m_[1] - m_[0]*m_[7]);
    t.m_[8] = one_over_det*(m_[0]*m_[4] - m_[3]*m_[1]);

    return t;
}

inline Matrix3 Matrix3::transpose() const
{
    Matrix3 t;
    t.m_[0] = m_[0];
    t.m_[1] = m_[3];
    t.m_[2] = m_[6];
    t.m_[3] = m_[1];
    t.m_[4] = m_[4];
    t.m_[5] = m_[7];
    t.m_[6] = m_[2];
    t.m_[7] = m_[5];
    t.m_[8] = m_[8];
    return t;
}

inline void Matrix3::print() const
{
    std::cout << std::fixed << std::setprecision(2);
    std::cout << m_[0]<<" "<<m_[3]<<" "<< m_[6]<<std::endl;
    std::cout << m_[1]<<" "<<m_[4]<<" "<< m_[7]<< std::endl;
    std::cout << m_[2]<<" "<<m_[5]<<" "<< m_[8]<<std::endl;
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

inline void Matrix4::makeIdentity(){
    m_[0] = 1.f; m_[4] = 1.f; m_[8] = 1.f; m_[12] = 1.f;
    m_[1] = 0.f; m_[2] = 0.f; m_[3] = 0.f; m_[5] = 0.f;
    m_[6] = 0.f; m_[7] = 0.f; m_[9] = 0.f; m_[12] = 0.f;
    m_[13] = 0.f; m_[14] = 0.f;
}

inline Matrix4 Matrix4::transpose() const
{
    Matrix4 t;
    t.m_[0] = m_[0];
    t.m_[1] = m_[4];
    t.m_[2] = m_[8];
    t.m_[3] = m_[12];
    t.m_[4] = m_[1];
    t.m_[5] = m_[5];
    t.m_[6] = m_[9];
    t.m_[7] = m_[13];
    t.m_[8] = m_[2];
    t.m_[9] = m_[6];
    t.m_[10] = m_[10];
    t.m_[11] = m_[14];
    t.m_[12] = m_[3];
    t.m_[13] = m_[7];
    t.m_[14] = m_[11];
    t.m_[15] = m_[15];
    return t;
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
                temp[i*4+j] += m_[k*4+j] * _m.m_[i*4+k];
            }
        }
    }
    return Matrix4(temp);
}

inline Vector3 Matrix4::operator*(const Vector3 &_v) const {
    float x, y, z, vx, vy, vz, vw;
    vx = _v.x;
    vy = _v.y;
    vz = _v.z;
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

inline void Matrix4::print() const {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << m_[0]<<" "<<m_[4]<<" "<< m_[8]<<" "<<m_[12]<<std::endl;
    std::cout << m_[1]<<" "<<m_[5]<<" "<< m_[9]<<" "<<m_[13]<<std::endl;
    std::cout << m_[2]<<" "<<m_[6]<<" "<< m_[10]<<" "<<m_[14]<<std::endl;
    std::cout << m_[3]<<" "<<m_[7]<<" "<< m_[11]<<" "<<m_[15]<<std::endl;
}

inline Matrix4 Matrix4::inverse() const {

    float det;
    det = m_[0]*m_[5]*m_[10]*m_[15] + m_[0]*m_[9]*m_[14]*m_[7] 
        + m_[0]*m_[13]*m_[6]*m_[11] + m_[4]*m_[1]*m_[14]*m_[11] 
        + m_[4]*m_[9]*m_[2]*m_[15]  + m_[4]*m_[13]*m_[10]*m_[3]
        + m_[8]*m_[1]*m_[6]*m_[15]  + m_[8]*m_[5]*m_[14]*m_[3] 
        + m_[8]*m_[13]*m_[2]*m_[7]  + m_[12]*m_[1]*m_[10]*m_[7] 
        + m_[12]*m_[5]*m_[2]*m_[11] + m_[12]*m_[9]*m_[6]*m_[3]
        - m_[0]*m_[5]*m_[14]*m_[11] - m_[0]*m_[9]*m_[6]*m_[15] 
        - m_[0]*m_[13]*m_[10]*m_[7] - m_[4]*m_[1]*m_[10]*m_[15] 
        - m_[4]*m_[9]*m_[14]*m_[3]  - m_[4]*m_[13]*m_[2]*m_[11]
        - m_[8]*m_[1]*m_[14]*m_[7]  - m_[8]*m_[5]*m_[2]*m_[15]
        - m_[8]*m_[13]*m_[6]*m_[3]  - m_[12]*m_[1]*m_[6]*m_[11]
        - m_[12]*m_[5]*m_[10]*m_[3] - m_[12]*m_[9]*m_[2]*m_[7];

        //std::cout << "det " << det << std::endl;

    if (abs(det) < 0.00001f) {
        //std::cerr << "det very small!" << std::endl;
        if (det > 0 ) { 
            det = 0.00001f;
        } else {
            det = -0.00001f;
        }
    }

    float invDet = 1.f/det;

    float b[16];
    b[0] = invDet*(m_[5]*m_[10]*m_[15]+m_[9]*m_[14]*m_[7]+m_[13]*m_[6]*m_[11]
         - m_[5]*m_[14]*m_[11]-m_[9]*m_[6]*m_[15]-m_[13]*m_[10]*m_[7]);
    b[4] = invDet*(m_[4]*m_[14]*m_[11]+m_[8]*m_[6]*m_[15]+m_[12]*m_[10]*m_[7]
         - m_[4]*m_[10]*m_[15]-m_[8]*m_[14]*m_[7]-m_[12]*m_[6]*m_[11]);
    b[8] = invDet*(m_[4]*m_[9]*m_[15]+m_[8]*m_[13]*m_[7]+m_[12]*m_[5]*m_[11]
         - m_[4]*m_[13]*m_[11]-m_[8]*m_[5]*m_[15]-m_[12]*m_[9]*m_[7]);
    b[12]= invDet*(m_[4]*m_[13]*m_[10]+m_[8]*m_[5]*m_[14]+m_[12]*m_[9]*m_[6]
         - m_[4]*m_[9]*m_[14]-m_[8]*m_[13]*m_[6]-m_[12]*m_[5]*m_[10]);
    b[1] = invDet*(m_[1]*m_[14]*m_[11]+m_[9]*m_[2]*m_[15]+m_[13]*m_[10]*m_[3]
         - m_[1]*m_[10]*m_[15]-m_[9]*m_[14]*m_[3]-m_[13]*m_[2]*m_[11]);
    b[5] = invDet*(m_[0]*m_[10]*m_[15]+m_[8]*m_[14]*m_[3]+m_[12]*m_[2]*m_[11]
         - m_[0]*m_[14]*m_[11]-m_[8]*m_[2]*m_[15]-m_[12]*m_[10]*m_[3]);
    b[9] = invDet*(m_[0]*m_[13]*m_[11]+m_[8]*m_[1]*m_[15]+m_[12]*m_[9]*m_[3]
         - m_[0]*m_[9]*m_[15]-m_[8]*m_[13]*m_[3]-m_[12]*m_[1]*m_[11]);
    b[13]= invDet*(m_[0]*m_[9]*m_[14]+m_[8]*m_[13]*m_[2]+m_[12]*m_[1]*m_[10]
         - m_[0]*m_[13]*m_[10]-m_[8]*m_[1]*m_[14]-m_[12]*m_[9]*m_[2]);
    b[2] = invDet*(m_[1]*m_[6]*m_[15]+m_[5]*m_[14]*m_[3]+m_[13]*m_[2]*m_[7]
         - m_[1]*m_[14]*m_[7]-m_[5]*m_[2]*m_[15]-m_[13]*m_[6]*m_[3]);
    b[6]=  invDet*(m_[0]*m_[14]*m_[7]+m_[4]*m_[2]*m_[15]+m_[12]*m_[6]*m_[3]
         - m_[0]*m_[6]*m_[15]-m_[4]*m_[14]*m_[3]-m_[12]*m_[2]*m_[7]);
    b[10]= invDet*(m_[0]*m_[5]*m_[15]+m_[4]*m_[13]*m_[3]+m_[12]*m_[1]*m_[7]
         - m_[0]*m_[13]*m_[7]-m_[4]*m_[1]*m_[15]-m_[12]*m_[5]*m_[3]);
    b[14]= invDet*(m_[0]*m_[13]*m_[6]+m_[4]*m_[1]*m_[14]+m_[12]*m_[5]*m_[2]
         - m_[0]*m_[5]*m_[14]-m_[4]*m_[13]*m_[2]-m_[12]*m_[1]*m_[6]);
    b[3] = invDet*(m_[1]*m_[10]*m_[7]+m_[5]*m_[2]*m_[11]+m_[9]*m_[6]*m_[3]
         - m_[1]*m_[6]*m_[11]-m_[5]*m_[10]*m_[3]-m_[9]*m_[2]*m_[7]);
    b[7] = invDet*(m_[0]*m_[6]*m_[11]+m_[4]*m_[10]*m_[3]+m_[8]*m_[2]*m_[7]
         - m_[0]*m_[10]*m_[7]-m_[4]*m_[2]*m_[11]-m_[8]*m_[6]*m_[3]);
    b[11]= invDet*(m_[0]*m_[9]*m_[7]+m_[4]*m_[1]*m_[11]+m_[8]*m_[5]*m_[3]
         - m_[0]*m_[5]*m_[11]-m_[4]*m_[9]*m_[3]-m_[8]*m_[1]*m_[7]);
    b[15]= invDet*(m_[0]*m_[5]*m_[10]+m_[4]*m_[9]*m_[2]+m_[8]*m_[1]*m_[6]
         - m_[0]*m_[9]*m_[6]-m_[4]*m_[1]*m_[10]-m_[8]*m_[5]*m_[2]);

    return Matrix4(b);
}

#endif
