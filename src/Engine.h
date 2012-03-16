/*
 * Engine.h
 *
 *  Created on: Mar 2, 2012
 *      Author: per
 */

#ifndef ENGINE_H_
#define ENGINE_H_

class Engine
{
public:
    static Engine& instance() { static Engine e; return e; };
    void init(const char * _name, int _width, int _height);
    void loadResources(const char * _file);
    void start();

    int mouseX() const { return mouseX_;};
    void mouseXIs(int x);
    int mouseY() const { return mouseY_;};
    void mouseYIs(int y);
private:
    int mouseX_;
    int mouseY_;

    enum State { NOT_INITIATED, RUNNING, PAUSED};
    State state_;

    Engine();
    Engine(const Engine & );
    void operator=(const Engine & );
};


#endif /* ENGINE_H_ */
