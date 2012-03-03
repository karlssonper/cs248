/*
 * Physics.h
 *
 *  Created on: Mar 2, 2012
 *      Author: per
 */

#ifndef PHYSICS_H_
#define PHYSICS_H_

class Physics
{
public:
    static Physics& instance() { static Physics ep; return p; };
};

#endif /* PHYSICS_H_ */
