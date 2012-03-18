/*
 * Sound.h
 *
 *  Created on: Mar 17, 2012
 *      Author: per
 */

#ifndef SOUND_H_
#define SOUND_H_

#include "MathEngine.h"
#include <vector>

class Sound
{
public:
    static Sound& instance() { static Sound s; return s; };
    enum SoundEnum{
        THEME = 0,
        IMPACT,
        CANNON,
        NUM_SOUNDS
    };

    void play(SoundEnum _se, Vector3 _pos) const;
    void listenerPositionIs(Vector3 _pos);
private:
    Sound();
    ~Sound();
    Sound(const Sound &);
    void operator=(const Sound &);

    unsigned int buffers_[NUM_SOUNDS];
    unsigned int source_[NUM_SOUNDS];
};
#endif /* SOUND_H_ */
