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

namespace irrklang{
    class ISoundEngine;
    class ISoundSource;
}
class Sound
{
public:
    static Sound& instance() { static Sound s; return s; };
    enum SoundEnum{
        EXPLOSION = 0, NUM_SOUNDS = 1
    };

    void play(SoundEnum _se, Vector3 _pos) const;
    void listenerPositionIs(Vector3 _pos);
private:
    Sound();
    ~Sound();
    Sound(const Sound &);
    void operator=(const Sound &);

    irrklang::ISoundEngine* engine_;
    std::vector<irrklang::ISoundSource*> sounds_;
};
#endif /* SOUND_H_ */
