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
};


#endif /* ENGINE_H_ */
