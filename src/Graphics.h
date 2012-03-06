/*
 * Graphics.h
 *
 *  Created on: Mar 2, 2012
 *      Author: per
 */

#ifndef GRAPHICS_H_
#define GRAPHICS_H_

#include <GL3/gl3w.h>

#include <string>
#include <vector>
#include <map>

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

class Graphics
{
public:
    static Graphics& instance() { static Graphics g; return g; };

    void BuffersNew(const std::string & _name,
                    GLuint & _VAO,
                    GLuint & _geometryVBO,
                    GLuint & _idxVBOO );

    template<class Vertex>
    void geometryIs(GLuint                      _geometryVBO,
                    GLuint                      _indexVBO,
                    const std::vector<Vertex> & _geometryData,
                    const std::vector<GLuint> & _indexData,
                    GLenum                      _usage);

    void bindGeometry(GLuint _VAO,
                      GLuint _VBO,
                      GLuint _n,
                      GLuint _stride,
                      GLuint _locIdx,
                      GLuint _offset);

    void drawIndices(GLuint _VAO, GLuint _VBO, GLuint _size);

    //mat4 projectionMatrix() const { return projectionMatrix; } ;

private:
    Graphics();
    ~Graphics();

    struct VAOData {
        GLuint VAO;
        GLuint geometryVBO;
        GLuint indexVBO;
    };
    std::map<GLuint, VAOData> VAOData_;
    std::map<std::string, GLuint> texture_;

    template<class T>
    void VBODataIs(GLenum                 _target,
                   GLuint                 _VBO,
                   const std::vector<T> & _data,
                   GLenum                 _usage);

    Graphics(const Graphics &);
    void operator=(const Graphics &);
};

template<class Vertex>
void Graphics::geometryIs(GLuint                      _geometryVBO,
                GLuint                      _indexVBO,
                const std::vector<Vertex> & _geometryData,
                const std::vector<GLuint> & _indexData,
                GLenum                      _usage)
{
    VBODataIs(GL_ARRAY_BUFFER, _geometryVBO, _geometryData, _usage);
    VBODataIs(GL_ELEMENT_ARRAY_BUFFER, _indexVBO, _indexData, _usage);
}

template<class T>
void Graphics::VBODataIs(GLenum                 _target,
                         GLuint                 _VBO,
                         const std::vector<T> & _data,
                         GLenum                 _usage)
{
    glBindBuffer(_target, _VBO);
    glBufferData(_target, _data.size() * sizeof(T), _data[0], _usage);
    glBindBuffer(_target, 0);
}


#endif /* GRAPHICS_H_ */
