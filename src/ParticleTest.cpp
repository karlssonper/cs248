#include "ParticleSystem.h"

int main() {
    Emitter::EmitterParams params;
    params.numParticles_ = 20;
    params.mass_ = 1.f;
    params.rate_ = 3.f;
    params.startAcc_ = Vector3(0.f, -1.f, 0.f);
    params.startPos_ = Vector3(0.f, 0.f, 0.f);
    params.startVel_ = Vector3(1.f, 1.f, 1.f);
    params.color_ = Vector3(1.f, 1.f, 1.f);
    params.type_ = Emitter::EMITTER_FIRE;
    return 0;
}


