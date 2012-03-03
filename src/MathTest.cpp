#include "Math.h"
#include <iostream>

using namespace std;



int main() {

    Matrix4 a;
    a.print();
    std::cout << std::endl;

    float bdata[16];
    bdata[0] = 2.f; bdata[4] = 0.f; bdata[8] = 0.f; bdata[12] = 1.f;
    bdata[1] = 0.f; bdata[5] = 2.f; bdata[9] = 0.f; bdata[13] = 1.f;
    bdata[2] = 2.f; bdata[6] = 0.f; bdata[10] = 2.f; bdata[14] = -1.f;
    bdata[3] = 0.f; bdata[7] = 0.f; bdata[11] = 0.f; bdata[15] = 2.f;

    Matrix4 b(bdata);
    b.print();
    std::cout << std::endl;

    Vector3 v1(1.f, 2.f, 3.f);
    Vector3 v2(b*v1);

    v2.print();
    std::cout << "v2 magnitude " << v2.mag() << std::endl;
    std::cout << std::endl;

    Matrix4 trans(Matrix4::translate(5.f, 6.f, 7.f));
    trans.print();
    std::cout << std::endl;

    Matrix4 rot(Matrix4::rotate(90.f, 1.0, 0.0, 0.0));
    rot.print();
    std::cout << std::endl;

    



   
    return 0;
}