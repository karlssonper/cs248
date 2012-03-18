/*
 * Sound.cpp
 *
 *  Created on: Mar 17, 2012
 *      Author: per
 */

#include <stdlib.h>
#include <stdio.h>

#include "Sound.h"
#ifdef _WIN32
    #include "al.h"
    #include "alc.h"
    #include "alut.h"
#else
    #include <AL/al.h>
    #include <AL/alc.h>
    #include <AL/alut.h>
#endif
#include <string>


static std::string SoundFiles[Sound::NUM_SOUNDS] = {
        "../sound/theme.wav",
        "../sound/impact.wav",
        "../sound/cannon.wav"
};

Sound::Sound()
{
    // Init openAL
    alutInit(0, NULL);
    // Clear Error Code (so we can catch any new errors)
    if (alGetError() != AL_NO_ERROR) std::cerr << "Error in Sound class!\n";

    alGenBuffers(NUM_SOUNDS, buffers_);
    alGenSources(NUM_SOUNDS, source_);
    if (alGetError() != AL_NO_ERROR) std::cerr << "Error in Sound class!\n";

    std::cout << "Loading sound files";
    float zeros[3] = {0.0f, 0.0f, 0.0f};
    for (unsigned int i = 0; i < NUM_SOUNDS; ++i) {
        ALenum     format;
        ALsizei    size;
        ALfloat    freq;
        ALvoid* data = alutLoadMemoryFromFile(
                SoundFiles[i].c_str(), &format, &size, &freq);
        alBufferData(buffers_[i],format,data,size,freq);
        //free(data);
        

        if (alGetError() != AL_NO_ERROR) std::cerr << "Error in Sound class!\n";
        alSourcei(source_[i], AL_BUFFER, buffers_[i]);
        alSourcefv (source_[i], AL_VELOCITY, zeros);
        alSourcefv (source_[i], AL_DIRECTION, zeros);
        std::cout << "..";
    }
    std::cout << "done." << std::endl;


    alListenerfv(AL_POSITION,zeros);
    alListenerfv(AL_VELOCITY,zeros);
    float orientation[6] = {
            1.0f, 0.0f, 1.0f,
            0.0f, 1.0f, 0.0f
            };
    alListenerfv(AL_ORIENTATION,orientation);
}

Sound::~Sound()
{
    alDeleteSources(NUM_SOUNDS, source_);
    alDeleteBuffers(NUM_SOUNDS, buffers_);

    alutExit();

}

void Sound::play(SoundEnum _se, Vector3 _pos) const
{
    alSourcefv (source_[_se], AL_POSITION, &_pos.x);
    alSourcePlay(source_[_se]);
}

void Sound::listenerPositionIs(Vector3 _pos)
{
    alListenerfv(AL_POSITION, &_pos.x);
}
