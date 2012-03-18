/*
 * Sound.cpp
 *
 *  Created on: Mar 17, 2012
 *      Author: per
 */

#include "Sound.h"
#include <irrKlang.h>
#include <string>

static std::string SoundFiles[Sound::NUM_SOUNDS] = {
        "../sound/test.mp3"
};

Sound::Sound()
{
    engine_ = irrklang::createIrrKlangDevice();
    for (unsigned int i = 0; i < NUM_SOUNDS; ++i) {
        sounds_[i] = engine_->addSoundSourceFromFile(SoundFiles[i].c_str());
    }
    engine_->setListenerPosition(irrklang::vec3df(0,1,0),
                                 irrklang::vec3df(1,1,1));
}

Sound::~Sound()
{
    for (unsigned int i = 0; i < NUM_SOUNDS; ++i) {
        sounds_[i]->drop();
    }
}

void Sound::play(SoundEnum _se, Vector3 _pos) const
{
    irrklang::ISound* sound = engine_->play3D(sounds_[_se],
            irrklang::vec3df(_pos.x, _pos.y, _pos.z),
                                    false,
                                    false,
                                    true);
    sound->setMinDistance(0.0f);
}

void Sound::listenerPositionIs(Vector3 _pos)
{
    engine_->setListenerPosition(irrklang::vec3df(_pos.x, _pos.y, _pos.z),
            irrklang::vec3df(1,1,1));
}
