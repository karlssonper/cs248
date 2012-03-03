/*
 * Graphics.cpp
 *
 *  Created on: Mar 2, 2012
 *      Author: per
 */

#include "Graphics.h"

Graphics::Graphics()
{

}

Graphics::~Graphics()
{

}

void Graphics::bindGeometry(GLuint                      _VAO,
                            GLuint                      _VBO,
                            GLuint                      _n,
                            GLuint                      _stride,
                            GLuint                      _locIdx,
                            GLuint                      _offset)
{
    glBindVertexArray(_VAO);
    glBindBuffer(_VBO);

    glEnableVertexAttribArray(_locIdx);
    glVertexAttribPointer(_locIdx, _n, GL_FLOAT, 0, _stride,
            BUFFER_OFFSET(_offset));

    glBindVertexArray(0);
    glBindBuffer(0);
}

void Graphics::drawIndices(GLuint _VAO, GLuint _VBO, GLuint _size)
{
    glBindVertexArray(_VAO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _VBO);

    glDrawElements(GL_TRIANGLES, _size, GL_UNSIGNED_INT, 0);

    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

