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
#include "ShaderData.h"

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

enum VBOtype { VBO_STATIC, VBO_DYNAMIC };

class Graphics
{
public:
    static Graphics& instance() { static Graphics g; return g; };

    void buffersNew(const std::string & _name,
                    GLuint & _VAO,
                    GLuint & _geometryVBO,
                    GLuint & _idxVBOO );
    void deleteBuffers(const std::string & _name);
    void deleteBuffers(GLuint _VAO);

    template<class Vertex>
    void geometryIs(GLuint                      _geometryVBO,
                    GLuint                      _indexVBO,
                    const std::vector<Vertex> & _geometryData,
                    const std::vector<GLuint> & _indexData,
                    VBOtype                     _type);

    void bindGeometry(GLuint _VAO,
                      GLuint _VBO,
                      GLuint _n,
                      GLuint _stride,
                      GLuint _locIdx,
                      GLuint _offset);

    void drawIndices(GLuint _VAO,
                     GLuint _VBO,
                     GLuint _size,
                     const ShaderData * _shaderData);

    GLuint texture(const std::string & _img);
    void deleteTexture(const std::string & _img);
    void deleteTexture(unsigned int _texID);
    GLuint shader(const std::string & _shader);
    void deleteShader(const std::string & _shader);
    void deleteShader(unsigned int _shaderID);

private:
    Graphics();
    ~Graphics();

    struct VAOData {
        GLuint VAO;
        GLuint geometryVBO;
        GLuint indexVBO;
    };
    typedef std::map<std::string, VAOData> VAOmap;
    VAOmap VAOData_;
    typedef std::map<std::string, GLuint> TexMap;
    TexMap texture_;

    struct ShaderID
    {
        GLuint vertexShaderId;
        GLuint fragmentShaderID;
        GLuint programID;
    };
    typedef std::map<std::string, ShaderID> ShaderMap;
    ShaderMap shader_;

    template<class T>
    void VBODataIs(GLenum                 _target,
                   GLuint                 _VBO,
                   const std::vector<T> & _data,
                   GLenum                 _usage);

    GLuint LoadShader(const std::string _shader);
    std::vector<char> ReadSource(const std::string _file);

    Graphics(const Graphics &);
    void operator=(const Graphics &);
};

template<class Vertex>
void Graphics::geometryIs(GLuint                      _geometryVBO,
                GLuint                      _indexVBO,
                const std::vector<Vertex> & _geometryData,
                const std::vector<GLuint> & _indexData,
                VBOtype                      _type)
{
    GLenum type;
    switch(_type){
        case VBO_STATIC:
            type = GL_STATIC_DRAW;
            break;
        case VBO_DYNAMIC:
            type = GL_DYNAMIC_DRAW;
            break;
    }

    VBODataIs(GL_ARRAY_BUFFER, _geometryVBO, _geometryData, type);
    VBODataIs(GL_ELEMENT_ARRAY_BUFFER, _indexVBO, _indexData, type);
}

template<class T>
void Graphics::VBODataIs(GLenum                 _target,
                         GLuint                 _VBO,
                         const std::vector<T> & _data,
                         GLenum                 _usage)
{
    glBindBuffer(_target, _VBO);
    glBufferData(_target, _data.size() * sizeof(T), &_data[0], _usage);
    glBindBuffer(_target, 0);
}


#endif /* GRAPHICS_H_ */
